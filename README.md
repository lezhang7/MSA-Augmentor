# MSA-Augmentor pretraining code

    |_config -> config settings for the model
    |_data -> .bin dataset construct pipeline
	    |_utils.py -> class for MSA dataset and binary dataset construction method
	    |_construct_binary_dataset.sh 
    |_datasets -> .bin dataset
    |_model -> model files
    |_install.sh -> deepspeed install shell script
    |_train_trainer.py -> pertraining file (huggingface trainer)
    |_run.sh -> pretraining shell script     

# Pretraining steps
**All the commands are designed for shlab cluster**

 1. Construct local binary dataset ( load training data from cluster is too slow, so it's better to  fisrt construct all your dataset to .bin file as shown in datasets )
 `srun -p CM2M -n 1 --cpus-per-task 32 python utils.py --output_dir /mnt/lustre/zhangle/projects/msat5/datasets/ --random_src --src_seq_per_msa_l 5 --src_seq_per_msa_u 10 --total_seq_per_msa 25 --local_file_path  /mnt/lustre/zhangle/projects/msat5/datasets/sorted_msa`
 change the parameters above you can construct your own binary dataset
 
 2. Install dependent libraries, including pytorch transformer as well deepspeed.
 3. Start pretraining using run.sh script, you can modify the parameter in the script!
