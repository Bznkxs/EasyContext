# llama3 cnt pretrain long context (sliding window)
set -ex

export CUDA_DEVICE_MAX_CONNECTIONS=1

export GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# Change for multinode config
PBSNODEFILE=$HOME/hostname.txt
export MASTER_ADDR=$(head -n 1 $PBSNODEFILE)
export MASTER_PORT=10086
NNODES=`wc -l < $PBSNODEFILE`
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
NODEID=$1 #passed in as argument
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MEGATRON_DIR="$SCRIPT_DIR/.."
ACCELERATE_DIR=$HOME/.local/bin/accelerate
# torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$PALS_NODEID --master_addr=$MASTER_ADDR \
#        --master_port=$MASTER_PORT test_multinode.py --backend=nccl
echo "In script: run_sft_distributed for rank $NODEID"
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real

# llama2-7B
$ACCELERATE_DIR launch \
--config_file  accelerate_configs/multi_node.yaml \
--main_process_ip $MASTER_ADDR \
--main_process_port $MASTER_PORT \
--machine_rank $NODEID \
--num_machines $NNODES \
--nproc_per_node $GPUS_PER_NODE \
train.py \
--batch-size 1 \
--gradient-accumulate-every 2 \
--output-dir ./output/7B_64K_bs_1M_rope_5M_step_1000_lr_2e-5 \
--seed 2022 \
--wandb EasyContext \
--max-train-steps 1000  \
--learning-rate 2e-5  \
--dataset yaofu/slimpajama-per-source-length-upsample \
--model /u/mtian8/LLM/model/HF_model/Llama-2-7b-hf  \
--seq-length 131072 \
--rope-theta 5000000 \
--parallel_mode data_parallel
