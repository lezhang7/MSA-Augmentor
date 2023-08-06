#gold_label results: /user/sunsiqi/zl/T5/af2/casp15/gold_label/T1106s1/T1061s1.csv
#predicted results: /user/sunsiqi/zl/T5/af2/casp15/orphan/A1T3R1.5/T1119/T1119_generation_2.csv
import os
import csv
import json
import logging
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
logger.setLevel(logging.INFO)
def get_lddt_score(csv_path):
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            return row['result']


def ensemble_results(args):
    gold_label_dir = '/user/sunsiqi/zl/T5/af2/casp15/gold_label/'
    predicted_pdb_root_dir = args.predicted_pdb_root_dir
    ensemble_dir = './'
    os.makedirs(ensemble_dir, exist_ok=True)
    all_results={}
    for msa_name in os.listdir(predicted_pdb_root_dir): #T1119

        all_results[msa_name]={}
        orphan_msa_dir = os.path.join(predicted_pdb_root_dir, msa_name)

        if os.path.isdir(orphan_msa_dir):
            for trial_result in os.listdir(orphan_msa_dir):
                if trial_result.endswith('.csv'):
                    orphan_csv_path = os.path.join(orphan_msa_dir, trial_result)
                    orphan_score = get_lddt_score(orphan_csv_path)
                    all_results[msa_name][trial_result]=orphan_score
        
        gold_result_path=os.path.join(gold_label_dir, msa_name, msa_name+'.csv')
        if os.path.exists(gold_result_path):
            gold_score = get_lddt_score(gold_result_path)
            all_results[msa_name]['gold_label']=gold_score
        logging.info(f'ensemble {msa_name} done')
    # save json file
    ablations=args.predicted_pdb_root_dir.split('/')[-2]
    json_file_path = os.path.join(ensemble_dir, f'ensemble_results_{ablations}.json')
    with open(json_file_path, 'w') as f:
        json.dump(all_results, f,indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description='evaluate lddt')
    parser.add_argument('--predicted_pdb_root_dir', type=str, help='predicted pdb root dir e.g ./af2/casp15/orphan/A1T3R1.5/')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    ensemble_results(args)