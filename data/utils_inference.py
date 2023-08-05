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
import random
from tqdm import tqdm
import random
RawMSA = Sequence[Tuple[str, str]]


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
        # labels[labels==128]=-100
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




if __name__=="__main__":
    args=parseargs()
    if args.random_src:
        print('src_seq_per_msa will random select in {}-{}, total seq per msa is {}'.format(args.src_seq_per_msa_l,args.src_seq_per_msa_u,args.total_seq_per_msa))
        args.src_seq_per_msa='random'
  