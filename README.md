# MSA-Augmentor codebase

codebase for paper **Enhancing the Protein Tertiary Structure Prediction by Multiple Sequence Alignment Generation** [arxiv](https://arxiv.org/abs/2306.01824)

# Pretrain

**All the commands are designed for slurm cluster, we use huggingface trainer to pretrain the model, more details could be find [here](https://huggingface.co/docs/transformers/main_classes/trainer)**

   1. Construct local binary dataset ( load training data from cluster is too slow, so it's better to  fisrt construct all your dataset to .bin file as shown in datasets )

      ```
      python utils.py \
         --output_dir ./datasets/ \
         --random_src --src_seq_per_msa_l 5\
         --src_seq_per_msa_u 10 \
         --total_seq_per_msa 25 \
         --local_file_path  path_to_pretrained_dataset 
      ```

   2. install dependency libraries `pip install -r requirements.txt`
   3. `bash run.sh` 

# Inference

1. download [checkpoints](https://drive.google.com/file/d/12cYk3WZDX18j-9xwYK9uu2kaGjmLuowB/view) 
2. run inference by `bash scripts/inference.sh`

*Note: all inference code is in inference.py*

# Evaluation

| DATASET | MSA                               | STRUCTURE                                                    |
| ------- | --------------------------------- | ------------------------------------------------------------ |
| CASP15  | https://zenodo.org/record/8126538 | [google drive](https://github.com/deepmind/alphafold/blob/main/docs/casp15_predictions.zip) |

### Alphafold2 Prediction

1. Please refer to [Alphafold2 GitHub](https://github.com/deepmind/alphafold) to learn more about set up af2.

2. We provide scripts to use alphafold2 to launch protein structure prediction by `bash scripts/run_af2`, one need to modify `msa directory`

### LDDT

1. Install **Fast_lDDT** following [here]()

2. modify these variables in `eval_lddt.py`

   -  `lddt`   *Fast_lDDT software path*
   -  `seqdir` *Path to Directory with only query .seq file (the first seq of each msa file)*
   -  `native` ground truth .pdb folder dir

3. running following code to evaluate lddt score with ground truth .pdb file
   ```python eval_lddt.py --predicted_pdb_root_dir $predicted_pdb_dir # e.g ./af2/casp15/orphan/A1T3R1.5/```

   *note: you should run at least one time with following code to get original results without our augment to compare*

   `python eval_lddt.py --predicted_pdb_root_dir $gold_label_dir #e.g .af2/casp15/gold_label/`

### Ensemble

Directly run following to get .json file of final results.

```python ensemble.py --predicted_pdb_root_dir ./af2/casp15/orphan/A1T3R1.5/```

