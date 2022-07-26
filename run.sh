
# deep speed 
export http_proxy=http://172.16.1.135:3128/  
export https_proxy=http://172.16.1.135:3128/  
export HTTP_PROXY=http://172.16.1.135:3128/  
export HTTPS_PROXY=http://172.16.1.135:3128/  
export COMET_API_KEY="Ft5V2tCSrL4a6FkEf0UOQZHKv" 
GPU_NUM=8
BS=1
srun -p Gvlab -N 1 --ntasks-per-node=1 --gres=gpu:$GPU_NUM --job-name=MSAT5_BASE \
    deepspeed --master_port 13245 train_trainer.py \
    --model_name_or_path ./config/base/ \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS \
    --max_steps 6000000 \
    --warmup_ratio 0.01 \
    --learning_rate 5e-5 \
    --max_target_length 512 \
    --dataloader_num_workers 2 \
    --do_train \
    --output_dir ./tmp/msat5-base \
    --overwrite_output_dir True \
    --logging_strategy steps \
    --logging_steps 250 \
    --save_steps 20000 \
    --save_total_limit 2 \
    --bf16 \
    --deepspeed ./config/base/ds_config.json \
    --local_msadataset_path /mnt/lustre/zhangle/projects/msat5/datasets/MSA_2.6M_1 \
 

# 多卡
# export http_proxy=http://172.16.1.135:3128/  
# export https_proxy=http://172.16.1.135:3128/  
# export HTTP_PROXY=http://172.16.1.135:3128/  
# export HTTPS_PROXY=http://172.16.1.135:3128/  
# export COMET_API_KEY="Ft5V2tCSrL4a6FkEf0UOQZHKv" 
# GPU_NUM=1
# BS=6
# srun -p CM2M -N 1 --ntasks-per-node=1 --gres=gpu:$GPU_NUM  \
# python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --master_port 12651 train_trainer.py \
# --model_name_or_path ./config/ \
#     --per_device_train_batch_size $BS \
#     --per_device_eval_batch_size $BS \
#     --num_train_epochs 100 \
#     --warmup_ratio 0.01 \
#     --max_train_samples 100 \
#     --max_eval_samples 100 \
#     --learning_rate 5e-5 \
#     --max_target_length 512 \
#     --dataloader_num_workers 2 \
#     --do_train \
#     --do_eval \
#     --output_dir ./tmp \
#     --overwrite_output_dir True \
#     --logging_strategy steps \
#     --logging_steps 100 \
#     --evaluation_strategy steps \
#     --save_steps 20000 \
#     --fp16 \
#     --local_msadataset_path /mnt/lustre/zhangle/projects/msat5/datasets/MSA_10K \
#     --logging_dir /mnt/lustre/zhangle/projects/msat5/tmp/test1/runs/1

#debug 
# export http_proxy=http://172.16.1.135:3128/  
# export https_proxy=http://172.16.1.135:3128/  
# export HTTP_PROXY=http://172.16.1.135:3128/  
# export HTTPS_PROXY=http://172.16.1.135:3128/  
# export COMET_API_KEY="Ft5V2tCSrL4a6FkEf0UOQZHKv" 
# GPU_NUM=4
# BS=3
# srun -p CM2M -N 1 --ntasks-per-node=1 --gres=gpu:$GPU_NUM  \
# python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --master_port 12651 train_trainer.py \
# --model_name_or_path ./config/ \
#     --per_device_train_batch_size $BS \
#     --per_device_eval_batch_size $BS \
#     --max_steps 20000 \
#     --warmup_ratio 0.01 \
#     --learning_rate 5e-5 \
#     --max_target_length 512 \
#     --do_train \
#     --output_dir /mnt/lustre/zhangle/projects/msat5/tmp/test1 \
#     --overwrite_output_dir True \
#     --logging_strategy steps \
#     --logging_steps 250 \
#     --save_steps 20000 \
#     --fp16 \
#     --local_msadataset_path /mnt/lustre/zhangle/projects/msat5/datasets/MSA_400 \
    


