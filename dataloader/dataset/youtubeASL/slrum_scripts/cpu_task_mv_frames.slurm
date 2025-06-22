#!/bin/bash


#SBATCH --partition=normal                    
#SBATCH --job-name=cleanup                   
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/signLanguageTrans/dataloader/dataset/youtubeASL/slrum_scripts/zlog/%x-%N-%j.out  
#SBATCH --error=/home/rhong5/research_pro/hand_modeling_pro/signLanguageTrans/dataloader/dataset/youtubeASL/slrum_scripts/zlog/%x-%N-%j.err   
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # 需要 4 个独立任务
#SBATCH --mem-per-cpu=10gb          # 每个 CPU 核分配 20GB 内存
#SBATCH --export=ALL 
#SBATCH --time=4-24:00:00            
#SBATCH --cpus-per-task=1            # 每个任务使用 1 个 CPU 核



# cd /projects/kosecka/hongrui/dataset/youtubeASL


# rsync -av --partial --append-verify /scratch/rhong5/dataset/youtubeASL_anno/ ./youtubeASL_anno/


TARGET_DIR="/scratch/rhong5/dataset/youtubeASL_frames"

echo "🚀 Starting batch deletion under: $TARGET_DIR"

# 每次找出一批文件删除，避免卡死
find "$TARGET_DIR" -type f -print0 | xargs -0 -n 1000 -P 8 rm -f
if [ $? -ne 0 ]; then
    echo "❌ Error occurred during batch deletion."
    exit 1
else
    echo "✅ Batch deletion completed successfully."
fi
