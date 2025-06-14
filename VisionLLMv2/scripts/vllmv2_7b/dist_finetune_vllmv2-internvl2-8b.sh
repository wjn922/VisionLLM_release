#!/bin/bash

OUTPUT_DIR=$1
PRETRAINED_MODEL=$2
DATASET_CONFIG=$3
GPUS=${GPUS:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-25001}

# stage 2
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS} --master_port=${PORT} \
    visionllmv2/train/train_mem.py \
    --version v1 \
    --multi_dataset True \
    --group_by_data_source True \
    --dataset_config ${DATASET_CONFIG} \
    --model_name_or_path ${PRETRAINED_MODEL} \
    --vis_encoder_path ${PRETRAINED_MODEL} \
    --vl_bridge_type internvl_mlp \
    --vis_output_layer -1 \
    --use_region_encoder False \
    --freeze_vis_encoder True \
    --tune_llm_embed True \
    --use_im_start_end False \
    --image_size 448 \
    --image_max_tile 6 \
    --use_pixelshuffle True \
    --image_aspect_ratio anyres \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --deepspeed scripts/zero3.json \
    --report_to "tensorboard" \
    | tee ${OUTPUT_DIR}/train.log

# e.g.
# s8a bash scripts/vllmv2_7b/dist_finetune_vllmv2-internvl2-8b.sh work_dirs/visionllmv2-internvl2-8b-ft-uninext work_dirs/internvl2-8b visionllmv2/datasets/configs/uninext/uninext_train.py