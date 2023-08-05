import numpy as np
import os
from Bio import SeqIO
from io import StringIO

base1 = '/user/sunsiqi/zl/T5/dataset/'
for src in ['/share/zhengliangzhen/transformer_msa/msa_part1_pack/','/share/zhengliangzhen/transformer_msa/msa_part2_pack/','/share/zhengliangzhen/transformer_msa/msa_part3_pack/']:
    
    npz_list = sorted(os.listdir(src))
    mlist = npz_list
    for npz in mlist:
        cpath = src   + npz
        spath = base1 + npz.split('.')[0]
        res = np.load(cpath)
        # os.mkdir(spath)
        a3m_list = res.files
        for a3m in a3m_list:
            st = res.get(a3m)
            with StringIO(st.__str__()) as f:
                tar_seqs = list(SeqIO.parse(f, "fasta"))[:1000]
                # SeqIO.write(tar_seqs, spath+'/'+a3m, "fasta-2line")
        print("finish "+ npz)

