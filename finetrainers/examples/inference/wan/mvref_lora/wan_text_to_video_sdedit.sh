#!/bin/bash

set -e -x

# Change to finetrainers directory and set PYTHONPATH
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Navigate to finetrainers root directory (4 levels up from examples/inference/wan/mvref_lora/)
FINETRAINERS_DIR="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
cd "$FINETRAINERS_DIR"
# Add parent directory to PYTHONPATH so finetrainers module can be imported
export PYTHONPATH="$(dirname "$FINETRAINERS_DIR"):${PYTHONPATH}"

# export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
# export TORCHDYNAMO_VERBOSE=1
# export WANDB_MODE="offline"
export WANDB_MODE="disabled"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL="DEBUG"

BACKEND="ptd"

NUM_GPUS=1
# CUDA_VISIBLE_DEVICES="0" # 필요시 설정

# Seed 설정 (환경변수 SEED로 설정 가능, 기본값 42)
SEED=42

# Check the JSON files for the expected JSON format
# validation.json을 참고하여 적절한 데이터셋 파일 경로 설정
DATASET_FILE="examples/training/sft/wan/mvref_lora_test/validation.json"

# 체크포인트 디렉토리 설정 (트레이닝 출력 디렉토리)
# 예: "outputs/wan_iclora" 또는 절대 경로
CHECKPOINT_DIR="outputs/wan_iclora_sdedit_r256"

# LoRA 가중치 디렉토리
LORA_WEIGHTS_DIR="${CHECKPOINT_DIR}/lora_weights"

# 자동으로 가장 최신 스텝의 LoRA 가중치 찾기
if [ -d "$LORA_WEIGHTS_DIR" ]; then
    # lora_weights 디렉토리에서 숫자로 된 폴더들을 찾아서 가장 큰 값(최신 스텝) 찾기
    # 디렉토리 이름이 "007000" 형식이거나 "7000" 형식일 수 있으므로 숫자로 변환해서 정렬
    LATEST_STEP_NUM=0
    LATEST_STEP_DIR=""
    
    for step_dir in "$LORA_WEIGHTS_DIR"/*; do
        if [ -d "$step_dir" ]; then
            # 디렉토리 이름에서 숫자 추출 (앞의 0 제거)
            step_name=$(basename "$step_dir")
            if [[ "$step_name" =~ ^[0-9]+$ ]]; then
                # 숫자로 변환 (앞의 0 제거)
                step_num=$((10#$step_name))
                if [ "$step_num" -gt "$LATEST_STEP_NUM" ]; then
                    LATEST_STEP_NUM=$step_num
                    LATEST_STEP_DIR="$step_name"
                fi
            fi
        fi
    done
    
    if [ -n "$LATEST_STEP_DIR" ]; then
        # 원래 디렉토리 이름을 사용 (이미 올바른 형식일 가능성이 높음)
        LORA_PATH="${LORA_WEIGHTS_DIR}/${LATEST_STEP_DIR}"
        
        # pytorch_lora_weights.safetensors 파일이 있는지 확인
        if [ -f "${LORA_PATH}/pytorch_lora_weights.safetensors" ]; then
            echo "Found latest LoRA checkpoint at step ${LATEST_STEP_DIR} (${LATEST_STEP_NUM}): ${LORA_PATH}"
        else
            echo "Error: LoRA weights file not found at ${LORA_PATH}/pytorch_lora_weights.safetensors"
            exit 1
        fi
    else
        echo "Error: No LoRA checkpoint directories found in ${LORA_WEIGHTS_DIR}"
        exit 1
    fi
else
    echo "Error: LoRA weights directory not found: ${LORA_WEIGHTS_DIR}"
    exit 1
fi

# 특정 스텝을 수동으로 지정하려면 아래 주석을 해제하고 LORA_PATH를 직접 설정
# LORA_STEP="005000"  # 원하는 스텝으로 변경 (예: 003500, 004000 등)
# LORA_PATH="${LORA_WEIGHTS_DIR}/${LORA_STEP}"

# Output directory: save to LORA_PATH/output folder
OUTPUT_DIR="${LORA_PATH}/output"
# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Depending on how many GPUs you have available, choose your degree of parallelism and technique!
DDP_1="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 2 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_4="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 4 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_8="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 8 --dp_shards 1 --cp_degree 1 --tp_degree 1"
CP_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 1 --cp_degree 2 --tp_degree 1"
CP_4="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 1 --cp_degree 4 --tp_degree 1"
# FSDP_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 2 --cp_degree 1 --tp_degree 1"
# FSDP_4="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 4 --cp_degree 1 --tp_degree 1"
# HSDP_2_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 2 --dp_shards 2 --cp_degree 1 --tp_degree 1"

# Parallel arguments
parallel_cmd=(
  $DDP_1
)

# Model arguments
# 트레이닝 스크립트와 동일한 모델 사용
model_cmd=(
  --model_name "wan"
  --pretrained_model_name_or_path "Wan-AI/Wan2.1-T2V-14B-Diffusers"
  --use_iclora
  --lora_path "$LORA_PATH"
  --condition_width_pixel 160
  --iclora_mode "sdedit"  # "preserve" (default) or "sdedit"
  # --vanilla  # Uncomment to skip LoRA loading (can be combined with any iclora_mode)
  # --enable_slicing  # 필요시 주석 해제
  # --enable_tiling   # 필요시 주석 해제
)

# Inference arguments
inference_cmd=(
  --inference_type text_to_video
  --dataset_file "$DATASET_FILE"
)

# Attention provider arguments
# Use native attention provider (same as training default)
attn_provider_cmd=(
  --attn_provider native
)

# Torch config arguments
torch_config_cmd=(
  --allow_tf32
  --float32_matmul_precision high
)

# Miscellaneous arguments
miscellaneous_cmd=(
  --seed "$SEED"
  --tracker_name "finetrainers-inference-mvref-lora"
  --output_dir "$OUTPUT_DIR"
  --init_timeout 600
  --nccl_timeout 600
  --report_to "wandb"
)

# Execute the inference script
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=$NUM_GPUS \
  --rdzv_backend c10d \
  --rdzv_endpoint="localhost:0" \
  examples/inference/inference.py \
    "${parallel_cmd[@]}" \
    "${model_cmd[@]}" \
    "${inference_cmd[@]}" \
    "${attn_provider_cmd[@]}" \
    "${torch_config_cmd[@]}" \
    "${miscellaneous_cmd[@]}"

echo -ne "-------------------- Finished executing script --------------------\n\n"

