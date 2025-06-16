#!/bin/bash


#SBATCH --partition=normal                    
#SBATCH --job-name=cleanT1                   
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/signLanguageTrans/dataloader/dataset/youtubeASL/slrum_scripts/zlog/%x-%N-%j.out  
#SBATCH --error=/home/rhong5/research_pro/hand_modeling_pro/signLanguageTrans/dataloader/dataset/youtubeASL/slrum_scripts/zlog/%x-%N-%j.err   
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # 需要 4 个独立任务
#SBATCH --mem-per-cpu=20gb          # 每个 CPU 核分配 20GB 内存
#SBATCH --export=ALL 
#SBATCH --time=4-24:00:00            
#SBATCH --cpus-per-task=2            # 每个任务使用 2 个 CPU 核



# cd /projects/kosecka/hongrui/dataset/youtubeASL


# rsync -av --partial --append-verify /scratch/rhong5/dataset/youtubeASL_anno/ ./youtubeASL_anno/


TARGET_DIR="/projects/kosecka/hongrui/dataset/how2sign/sentence_level"

echo "Deleting files under: $TARGET_DIR"

if [ -d "$TARGET_DIR" ]; then
    # 更快更安全地删除所有文件（保留目录）
    find "$TARGET_DIR" -type f -delete
    echo "Deletion completed."
else
    echo "Directory $TARGET_DIR does not exist."
fi
