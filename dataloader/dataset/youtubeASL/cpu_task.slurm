#!/bin/bash


#SBATCH --partition=normal                    
#SBATCH --job-name=youtubeasl                   
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/signLanguageTrans/zlog/%u/%x-%N-%j.out  
#SBATCH --error=/home/rhong5/research_pro/hand_modeling_pro/signLanguageTrans/zlog/%u/%x-%N-%j.err   
#SBATCH --nodes=1
#SBATCH --ntasks=2                 # 需要 4 个独立任务
#SBATCH --mem-per-cpu=20gb          # 每个 CPU 核分配 20GB 内存
#SBATCH --export=ALL 
#SBATCH --time=4-24:00:00            
#SBATCH --cpus-per-task=2            # 每个任务使用 2 个 CPU 核


set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

module load gnu10                           
module load python

source /home/rhong5/py310Torch/bin/activate

cd /home/rhong5/research_pro/hand_modeling_pro/signLanguageTrans/dataloader/dataset/youtubeASL

srun --exclusive python youtubeASLPieces.py --start_idx 86486 --end_idx 157185 &
srun --exclusive python youtubeASLPieces.py --start_idx 157186 --end_idx 227885 &
# srun --exclusive python youtubeASLPieces.py --start_idx 227886 --end_idx 298585 &
# srun --exclusive python youtubeASLPieces.py --start_idx 298586 --end_idx 369285 &
wait  # 确保所有任务完成后 SLURM 任务才结束

# torchrun --nnodes=${NUM_NODES} --nproc_per_node=${GPUS_PER_NODE} train_VAE.py --debug 

# torchrun --nnodes=1 --nproc_per_node=1 train_VAE.py --batch_size 10

# salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-24:00:00
# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-24:00:00