import glob
import torch
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset
import string
from Bio import SeqIO
import itertools

class MSADataSet(Dataset):
    def __init__(self,data_dir=None,data_path_list=None,only_poor_msa=None):
        """
        data_dir: read all .a3m file in this dir
        data_path_list: read all .a3m file in this list
        """
        self.setup_traslation()
        if data_path_list is None:
            data_path_list=glob.glob(data_dir+'/*.a3m')

        self.msa_data={msa_file_path.split("/")[-1].split(".")[0]:self.read_msa(msa_file_path,9999) for msa_file_path in data_path_list}
        assert all([self.check_same_len(msa) for msa in self.msa_data.values()]),"all sequence in a msa should have the same length"
        if only_poor_msa:
            self.msa_data={k:v for k,v in self.msa_data.items() if len(v)<only_poor_msa}
        for k,v in self.msa_data.items():
            print(f"{k}: unique seq num:{len(set([x[1] for x in v]))} | total seq num:{len(v)}")
       
    def __getitem__(self, index):
        # if self.is_generation:
        #     return {"all":self.msa_data[index]}
        # return {"src":self.src[index],"tgt":self.tgt[index],"all":self.msa_data[index]}
        if index in self.msa_data:
            return {'msa_name':index,'msa':self.msa_data[index]}
        else:
            key=list(self.msa_data.keys())[index]
            return {'msa_name':key,'msa':self.msa_data[key]}
    def __len__(self):
        return len(self.msa_data)
    def setup_traslation(self):
        deletekeys = dict.fromkeys(string.ascii_lowercase)
        deletekeys["."] = None
        deletekeys["*"] = None
        self.translation = str.maketrans(deletekeys)
    def remove_insertions(self,sequence) :
        """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
        return sequence.translate(self.translation)
    def read_msa(self,filename, nseq) :
        """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
        
        return [(record.description, self.remove_insertions(str(record.seq)))
                    for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]
    def check_same_len(self,msa:List[Tuple[str,str]]):
        #check if all sequence in a msa has the same length
        l=set([len(x[1]) for x in msa])
        return len(l)==1