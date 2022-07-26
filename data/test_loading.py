
import numpy as np
import os
from Bio import SeqIO
from io import StringIO,BytesIO
import io
import itertools
from tqdm import tqdm
from petrel_client.client import Client

client = Client('~/petreloss.conf')
target_url="s3://rmsd_data/msa_dataset/ur50_uc30_msa_pack_1000_2337/"
# target_url="s3://rmsd_data/zfold/zfold_ckpts/m1-384_256_lm4_lp4_md128_mp0.15_gr1_bs128_pld0.3-MSATrans/"
# target_files=client.get_file_iterator(target_url)
# for i,(p,k) in enumerate(target_files):
#     print(p)
#     with io.BytesIO(client.get("s3://"+p,update_cache=True)) as f:
#         wrapper = io.TextIOWrapper(f, encoding='utf-8')

#         list=[(record.description) for record in itertools.islice(SeqIO.parse(wrapper, "fasta"), 5)]
#         print(list)
#         # print(SeqIO.parse(wrapper, "fasta"))
#     if i>3:
#         break

with io.BytesIO(client.get("s3://rmsd_data/zfold/zfold_ckpts/m1-384_256_lm4_lp4_md128_mp0.15_gr1_bs128_pld0.3-MSATrans/model.yaml",update_cache=True)) as f:
    print(f)
    
             