

# MSA-Augmentor codebase

codebase for paper **Enhancing the Protein Tertiary Structure Prediction by Multiple Sequence Alignment Generation** [arxiv](https://arxiv.org/pdf/2210.07920.pdf)

# Pretraining steps

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

Please refer to [Alphafold2 GitHub](https://github.com/deepmind/alphafold) to learn more about folding and evaluation with ground truth protein structure

