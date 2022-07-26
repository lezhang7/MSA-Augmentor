import numpy as np
import os
from Bio import SeqIO
from io import StringIO
import io
from tqdm import tqdm
from petrel_client.client import Client

client = Client('~/petreloss.conf')
source_files=client.get_file_iterator('s3://rmsd_data/transformer_msa/msa_part3_pack/')
target_url="s3://rmsd_data/msa_dataset/"


for p,k in source_files:
    folder_name=p.split('/')[-1].split('.')[0]
    target_folder_url=target_url+folder_name
    # start from 1000_2191
    # print('target_folder_url ',target_folder_url)
    if not client.isdir(target_folder_url):
        # print(target_folder_url,' is not contrained in this url')
        # break 
        print('processing ',folder_name)
        with io.BytesIO(client.get("s3://"+p,update_cache=True)) as f:
            res=np.load(f)
            a3m_list = res.files
            for a3m in tqdm(a3m_list):
                try:
                    st = res.get(a3m)
                    # print('a3m ',a3m)
                    with StringIO(st.__str__()) as f:
                        tar_seqs = list(SeqIO.parse(f, "fasta"))[:1000] 
                        with io.StringIO() as ft:
                            SeqIO.write(tar_seqs, ft, "fasta-2line")
                            ft.seek(0)                  
                            client.put(target_folder_url+'/'+a3m,io.BytesIO(ft.read().encode('utf8')))
                except Exception as e:
                    print('fail file name:',a3m)
             

