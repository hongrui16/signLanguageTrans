#!/bin/bash


#SBATCH --partition=normal                    
#SBATCH --job-name=mv_youtubeasl                   
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/signLanguageTrans/zlog/%u/%x-%N-%j.out  
#SBATCH --error=/home/rhong5/research_pro/hand_modeling_pro/signLanguageTrans/zlog/%u/%x-%N-%j.err   
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # 需要 4 个独立任务
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

cd /projects/kosecka/hongrui/dataset/youtubeASL


rsync -av --partial --append-verify /scratch/rhong5/dataset/youtubeASL_anno/ ./youtubeASL_anno/
