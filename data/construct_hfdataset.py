import glob
import os
from Bio import SeqIO
import itertools
import argparse
from datasets import Dataset
import string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
from typing import List, Tuple
from tqdm import tqdm
translation = str.maketrans(deletekeys)
def parseargs():
    parser = argparse.ArgumentParser(description="Construct a huggingface datset of msa files")
    parser.add_argument("--folder_path",type=str,required=True)
    parser.add_argument("--num_seq_per_msa",type=int,default=3,help='the number of row in each example in the dataset')
    parser.add_argument("--num_examples",type=int,default=None,help='the number of row in each example in the dataset')
    parser.add_argument("--test_size",type=float,default=0.1)
    args = parser.parse_args()
    # sanity check

    return args
def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.id, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

def read_batch(batch_file,batch_type='src'):
    batch=[]
    for msa_file in tqdm(batch_file,desc='processing {} file'.format(batch_type)):
        try:
            if batch_type=='src':
                batch.append([' '.join(list(msa[1])) for msa in msa_file[:len(msa_file)//2]])     
            if batch_type=='tgt':
                batch.append([' '.join(list(msa[1])) for msa in msa_file[len(msa_file)//2:]])            
        except ValueError as e:
            print(e,'batch_type can only be src or tgt')
    return batch
def main(args):
    if args.num_examples is not None and isinstance(args.num_examples,int):
        all_file_path=glob.glob(args.folder_path+'/*'+'/*.a3m')[:args.num_examples]
    else:
        all_file_path=glob.glob(args.folder_path+'/*'+'/*.a3m')
    print('total : {} files'.format(len(all_file_path)))
    msa_files=[read_msa(file_path,args.num_seq_per_msa*2) for file_path in all_file_path]
    msa_hfdataset=Dataset.from_dict({'src':read_batch(msa_files),'tgt':read_batch(msa_files,'tgt'),'length':[len(msa[0][1]) for msa in msa_files]})
    # print(msa_hfdataset[0])
    msa_hfdataset=msa_hfdataset.filter(lambda x:(len(x['src'])==args.num_seq_per_msa)&(len(x['tgt'])==args.num_seq_per_msa))
    msa_hfdataset=msa_hfdataset.train_test_split(test_size=args.test_size)
    msa_hfdataset.save_to_disk('msa_{}'.format(args.num_seq_per_msa))

if __name__=='__main__':
    args=parseargs()
    main(args)