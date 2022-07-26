from re import S
import torch
import os
from Bio import SeqIO
import io
import itertools
import argparse
from typing import Sequence, Tuple, List, Union
import string
import glob
from torch.utils.data import Dataset
from petrel_client.client import Client
import random
from tqdm import tqdm
import random
from multiprocessing import Pool
RawMSA = Sequence[Tuple[str, str]]

class MSADataSet(Dataset):
    def __init__(self,local_file_path=None,remote_file_url=None,src_seq_per_msa=None,total_seq_per_msa=None,num_msa_files=None,src_seq_per_msa_l=None,src_seq_per_msa_u=None,src_path=None,tgt_path=None):
        if src_path and tgt_path :#loading from already processed dataset
            self.src=torch.load(src_path)
            self.tgt=torch.load(tgt_path)
        else: #loadging from raw dataset
            self.translation = self.get_translation()
            if local_file_path:
                msa_data=self.read_from_local(local_file_path,total_seq_per_msa,num_msa_files)
            elif remote_file_url:
                msa_data=self.read_from_ceph(remote_file_url,total_seq_per_msa,num_msa_files)
            else:
                raise ValueError('You must enter either local_path or remote_url to get dataset')
            self.src = [msa[:src_seq_per_msa if isinstance(src_seq_per_msa, int) else random.randint(src_seq_per_msa_l,src_seq_per_msa_u)] for msa in msa_data]
            tgt_seq_num_list=[len(src) for src in self.src]
  
            self.tgt = [msa[tgt_seq_per_msa:total_seq_per_msa] for msa,tgt_seq_per_msa in zip(msa_data,tgt_seq_num_list)]
           

    def __getitem__(self, index):
        return {"src":self.src[index],"tgt":self.tgt[index]}
    def __len__(self):
        return len(self.src)
    def get_translation(self):
        deletekeys = dict.fromkeys(string.ascii_lowercase)
        deletekeys["."] = None
        deletekeys["*"] = None
        return str.maketrans(deletekeys)
    def remove_insertions(self,sequence) :
        """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
        return sequence.translate(self.translation)
    def read_msa(self,filename, nseq) :
        """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
        return [(record.description, self.remove_insertions(str(record.seq)))
                    for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]
    def check_len(self,msa):
        #check if all sequence in a msa has the same length
        l=set([len(x[1]) for x in msa])
        return len(l)==1
    def read_from_ceph(self,remote_file_url,total_seq_per_msa,num_msa_files):
        
        client = Client('~/petreloss.conf')
        target_files=client.get_file_iterator(remote_file_url)
        print('Loading dataset from remote database, Connection sucessfully')
        msa_data=[]
        if num_msa_files is not None:
            pbar=tqdm(desc="Loading Dataset",total=num_msa_files)
        else:
            print('Warnining ! will process all datasets in the url')
        pool=Pool(32)
        for i,(p,k) in enumerate(target_files):
            # show info
            if num_msa_files is not None:
                if len(msa_data)==num_msa_files:
                    break
                pbar.update(1)
            else:
                print('Processing NO. {} file :'.format(i),p)
            # loading from remote
            res=pool.apply_async(func=self.load_ceph_data,args=(p,client,total_seq_per_msa))
            cur_msa=res.get()
            # check 
            if self.check_len(cur_msa) and len(cur_msa)==total_seq_per_msa:
                msa_data.append(cur_msa)
        pool.close()
        pool.join()
        return msa_data
    def load_ceph_data(self,p,client,total_seq_per_msa):
        with io.BytesIO(client.get("s3://"+p,update_cache=True)) as f:
            wrapper = io.TextIOWrapper(f, encoding='utf-8')
            cur_msa=self.read_msa(wrapper,total_seq_per_msa)
        return cur_msa
    def read_from_local(self,local_file_path,total_seq_per_msa,num_msa_files):
        print('Loading dataset from local database')
        if num_msa_files is not None:
            # train on small dataset
            data_path_list=glob.glob(local_file_path+'/*.a3m')[:num_msa_files]
            if not len(data_path_list):
                data_path_list=glob.glob(local_file_path+'/*/*.a3m')[:num_msa_files]
        else:
            print('Warning ! will process all datasets from local folder')
            data_path_list=glob.glob(local_file_path+'/*.a3m')
            if not len(data_path_list):
                data_path_list=glob.glob(local_file_path+'/*/*.a3m')
        msa_data=[self.read_msa(local_file_path,total_seq_per_msa) for local_file_path in data_path_list]
        msa_data=[cur_msa for cur_msa in msa_data if self.check_len(cur_msa) and len(cur_msa)==total_seq_per_msa]
        return msa_data


class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, tokenizer,max_len=512):
        self.max_len=max_len-1
        self.tokenizer = tokenizer
    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)
        seq_encoded_list = [self.tokenizer(self._tokenize(seq_str),truncation=True,max_length=self.max_len+1).input_ids for seq_str in seq_str_list]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        tokens = torch.empty(
            (
                batch_size,
                max_len 
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.tokenizer.pad_token_id)
        labels = []
        strs = []
        for i, (label, seq_str, seq_encoded) in enumerate(
            zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[i,0:len(seq_encoded)] = seq

        return labels, strs, tokens
    def _tokenize(self,sequence):
        return ' '.join(list(sequence)) 
class DataCollatorForMSA(BatchConverter):
    def msa_batch_convert(self, inputs: Union[Sequence[RawMSA], RawMSA]):
        # RawMSA: Sequence[label:str,acid_seq:str]
        if isinstance(inputs[0][0], str):
            # Input is a single MSA
            raw_batch: Sequence[RawMSA] = [inputs]  # type: ignore
        else:
            raw_batch = inputs  # type: ignore

        batch_size = len(raw_batch)
        max_alignments = max(len(msa) for msa in raw_batch) #MSA的数量
        max_seqlen_msa = max(len(msa[0][1]) for msa in raw_batch) # MSA的每个序列长度
        max_seqlen=min(max_seqlen_msa,self.max_len)+1 #加一是为了凑齐每句话结尾有一个/s
        tokens = torch.empty(
            (
                batch_size,
                max_alignments,
                max_seqlen,
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.tokenizer.pad_token_id)
        labels = []
        strs = []

        for i, msa in enumerate(raw_batch):
            msa_seqlens = set(len(seq) for _, seq in msa)
            if not len(msa_seqlens) == 1:
                raise RuntimeError(
                    "Received unaligned sequences for input to MSA, all sequence "
                    "lengths must be equal."
                )
            msa_labels, msa_strs, msa_tokens = super().__call__(msa)
            msa_len=msa_tokens.size(1)
            msa_tokens=msa_tokens[:,:min(msa_len,max_seqlen)]
            labels.append(msa_labels)
            strs.append(msa_strs)
            tokens[i, : msa_tokens.size(0), : msa_tokens.size(1)] = msa_tokens
        return tokens
    def __call__(self,batch): 
        input_ids=self.msa_batch_convert([example["src"] for example in batch])
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id).type_as(input_ids)
        labels=self.msa_batch_convert([example["tgt"] for example in batch])
        decoder_attention_mask=labels.ne(self.tokenizer.pad_token_id).type_as(input_ids)
        labels[labels==self.tokenizer.pad_token_id]=-100
        return {'input_ids':input_ids,'labels':labels,"attention_mask":attention_mask,"decoder_attention_mask":decoder_attention_mask}

def parseargs():
    parser = argparse.ArgumentParser(description="Construct a binary datset of msa files")
    parser.add_argument("--output_dir",type=str,default='./datasets/')
    parser.add_argument("--src_seq_per_msa",type=int,default=3)
    parser.add_argument("--random_src",action="store_true")
    parser.add_argument("--total_seq_per_msa",type=int,default=30,help='number of sequences in one msa')
    parser.add_argument("--src_seq_per_msa_l",type=int,default=2)
    parser.add_argument("--src_seq_per_msa_u",type=int,default=10)
    
    parser.add_argument("--num_msa_files",type=int,default=None,help='number of msa files in the dataset')

    parser.add_argument("--remote_file_url",type=str,default="s3://rmsd_data/msa_dataset/")
    parser.add_argument("--local_file_path",type=str,default=None)
    parser.add_argument("--num_chunk",type=int,default=5,help='number of chunk')
    parser.add_argument("--val_size_ratio",type=float)
    parser.add_argument("--val_size",type=int,default=100)
    parser.add_argument("--split",action="store_true")

    args = parser.parse_args()
    return args

def split(full_list,shuffle=False,ratio=0.05):

    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total==0 or offset<1:
        return [],full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1,sublist_2

def construct_dataset(args):
    msa_dataset=MSADataSet(local_file_path=args.local_file_path,remote_file_url=args.remote_file_url,src_seq_per_msa=args.src_seq_per_msa,total_seq_per_msa=args.total_seq_per_msa,num_msa_files=args.num_msa_files,src_seq_per_msa_l=args.src_seq_per_msa_l,src_seq_per_msa_u=args.src_seq_per_msa_u)
    
    if args.split:
        print('Split train and eval datasets')
        val_src,train_src=split(msa_dataset.src)
        val_tgt,train_tgt=split(msa_dataset.tgt)
    else:
        train_src=msa_dataset.src
        train_tgt=msa_dataset.tgt
    num_msa_files=len(msa_dataset)

    if num_msa_files>1000000:
        msa_files_name='MSA_'+str(round(num_msa_files/1000000,1))+'M'
    elif num_msa_files>1000:
        msa_files_name='MSA_'+str(num_msa_files//1000)+'K'
    else:
        msa_files_name='MSA_'+str(num_msa_files)
    
   

    output_dir=args.output_dir+msa_files_name
    os.makedirs(output_dir,exist_ok=True)


    torch.save(train_src,output_dir+'/'+'train_src.bin')
    torch.save(train_tgt,output_dir+'/'+'train_tgt.bin')
    if args.split:
        torch.save(val_src,output_dir+'/'+'val_src.bin')
        torch.save(val_tgt,output_dir+'/'+'val_tgt.bin')


if __name__=="__main__":
    args=parseargs()
    print('Begin construct datasets')
    if args.random_src:
        print('src_seq_per_msa will random select in {}-{}, total seq per msa is {}'.format(args.src_seq_per_msa_l,args.src_seq_per_msa_u,args.total_seq_per_msa))
        args.src_seq_per_msa='random'
    construct_dataset(args)