#!/bin/bash

# Define variables for number of nodes and GPUs per node
# V_NUM_NODES=1                # Number of nodes
# V_GPUS_PER_NODE=1            # Number of GPUs per node
# V_TOTAL_CPUS_PER_TASK=4      # Number of CPU cores per task
# V_JOB_NAME="vae"             # Job name
# V_MEM_PER_NODE='40gb'         #  100gb for InterHand26M, memory per CORE; total memory is 1 TB (1,000,000 MB), 
# V_GPU_GROUP="gpuq"  # 'contrib-gpuq' or 'gpuq'
# V_TIME_LIMIT="2-24:00:00"    # Time limit for the job
# V_OUTPUT_DIR="/home/rhong5/research_pro/hand_modeling_pro/InterWild/zlog"  # Output directory
# V_TEST_VAR=1  # 简单的变量


# #SBATCH --partition=${GPU_GROUP}                    # need to set 'gpuq' or 'contrib-gpuq'  partition
# #SBATCH --qos=gpu                           # need to select 'gpu' QOS or other relvant QOS
# #SBATCH --job-name=${JOB_NAME}                   # vqgan, vqvae, vqdiffusion, gaussiandiffusion
# #SBATCH --output=${OUTPUT_DIR}/job_%j.out
# #SBATCH --error=${OUTPUT_DIR}/job_%j.err
# #SBATCH --nodes=${NUM_NODES}      # Use the defined variable for number of nodes
# #SBATCH --ntasks-per-node=${GPUS_PER_NODE}                 # number of cores needed
# #SBATCH --gres=gpu:A100.80gb:${GPUS_PER_NODE}  # Number of GPUs per node
# #SBATCH --mem=${MEM_PER_NODE}                # 100gb for InterHand26M, memory per CORE; total memory is 1 TB (1,000,000 MB), 
# #SBATCH --export=ALL 
# #SBATCH --time=${TIME_LIMIT}                   # set to 24hr; please choose carefully
# #SBATCH --cpus-per-task=${TOTAL_CPUS_PER_TASK}  # Number of CPU cores per task

#SBATCH --partition=contrib-gpuq                    # need to set 'gpuq' or 'contrib-gpuq'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QOS or other relvant QOS
#SBATCH --job-name=unisign                   # vqgan, vqvae, vqdiffusion, gaussiandiffusion
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/signLanguageTrans/Zlog/%u/%x-%N-%j.out   # Output file
#SBATCH --error=/home/rhong5/research_pro/hand_modeling_pro/signLanguageTrans/Zlog/%u/%x-%N-%j.err    # Error file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                # number of cores needed
#SBATCH --gres=gpu:A100.80gb:1                # up to 8; only request what you need # gpu:A100.80gb:1 ; gpu:3g.40gb:1 
#SBATCH --mem=60gb                # 100gb for InterHand26M, memory per CORE; total memory is 1 TB (1,000,000 MB), 
#SBATCH --export=ALL 
#SBATCH --time=4-24:00:00                   # set to 24hr; please choose carefully
#SBATCH --cpus-per-task=4  # Number of CPU cores per task

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

module load gnu10                           
module load python

source /home/rhong5/py310Torch/bin/activate
# source /home/rhong5/env_mmcv/bin/activate
cd /home/rhong5/research_pro/hand_modeling_pro/signLanguageTrans



# Check Python and CUDA
python uniSign.py 2> >(grep -v "absl::InitializeLog" | grep -v "gl_context_egl.cc" | grep -v "gl_context.cc" >&2)

# torchrun --nnodes=${NUM_NODES} --nproc_per_node=${GPUS_PER_NODE} train_VAE.py --debug 

# torchrun --nnodes=1 --nproc_per_node=1 train_VAE.py --batch_size 10

# salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-24:00:00
# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-24:00:00
# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:2g.20gb:1 --mem=40gb -t 0-24:00:00
# salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:2g.20gb:1 --mem=40gb -t 0-24:00:00