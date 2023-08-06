import subprocess
import logging
import argparse
import os
logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
logger.setLevel(logging.INFO)
lddt = "/GitBucket/Fast_lDDT/Fast_lDDT" #software for computing
seqdir = "./casp15_seq/" # only query
native = "./casp15_predictions/single_chain" # .pdb folder


def eval_iddt(predicted_pdb_root_dir):
    # predicted_pdb_root_dir: ./af2/casp15/orphan/A1T3R1.5/  
   
    for predicted_msa in sorted(os.listdir(predicted_pdb_root_dir)): # T1113
        predicted_msa_dir=os.path.join(predicted_pdb_root_dir,predicted_msa) # .af2/casp15/gold_label/T1151s2
        if 'label' in predicted_pdb_root_dir:
            pdb_path=os.path.join(predicted_msa_dir,'ranked_0.pdb') # .af2/casp15/gold_label/T1151s2/ranked_0.pdb
            if os.path.exists(pdb_path):
                with open(os.path.join(predicted_msa_dir,f"{predicted_msa}.csv"), 'w') as fp: 
                    fp.write("name,result\n")
                    p = subprocess.Popen(
                        [lddt, 
                        '-i', os.path.join(seqdir, "%s.seq"%predicted_msa), 
                        '-n', os.path.join(native, "%s.pdb"%predicted_msa), 
                        '-m', pdb_path,
                        '-v', '1'],
                        #shell=True, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.STDOUT
                    )
                    res = p.stdout.readlines()[-1].decode().split(' ')[2]
                    fp.write("%s,%s\n"%(predicted_msa, res))
                    ret = p.wait()
                logger.info("finish %s"%(predicted_msa))
            else:
                logger.info("no pdb file %s"%(pdb_path))
        else:
            for trial in sorted(os.listdir(predicted_msa_dir)): # generation_0
                if trial.startswith('generation'):
                    pdb_path=os.path.join(predicted_msa_dir,trial,'ranked_0.pdb') # ./af2/casp15/orphan/A1T3R1.5/T1113/generation_0/ranked_0.pdb
                    if os.path.exists(pdb_path):
                        with open(os.path.join(predicted_msa_dir,f"{predicted_msa}_{trial}.csv"), 'w') as fp: 
                            fp.write("name,result\n")
                            p = subprocess.Popen(
                                [lddt, 
                                '-i', os.path.join(seqdir, "%s.seq"%predicted_msa), 
                                '-n', os.path.join(native, "%s.pdb"%predicted_msa), 
                                '-m', pdb_path,
                                '-v', '1'],
                                #shell=True, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.STDOUT
                            )
                            res = p.stdout.readlines()[-1].decode().split(' ')[2]
                            fp.write("%s,%s\n"%(predicted_msa, res))
                            ret = p.wait()
                        logger.info("finish %s"%(predicted_msa))
                    else:
                        logger.info("no pdb file %s"%(pdb_path))
def parse_args():
    parser = argparse.ArgumentParser(description='evaluate lddt')
    parser.add_argument('--predicted_pdb_root_dir', type=str, help='predicted pdb root dir e.g ./af2/casp15/orphan/A1T3R1.5/ or .af2/casp15/gold_label/ ')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    eval_iddt(args.predicted_pdb_root_dir)