#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32


CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --use_fast_tokenizer True \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen-1_8B \
    --dataset nhanes_train \
    --template default \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_target "c_attn","c_proj" \
    --output_dir /home/hn/Documents/enar-datafest-2024/fine-tune_1-16-24/datafest/Qwen/1.8B/lora/with_chol \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16 \
    --overwrite_output_dir \
    --cache_dir ./cache/Qwen/1.8B/lora-1

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --use_fast_tokenizer True \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen-1_8B \
    --dataset nhanes_train_no_chol \
    --template default \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_target "c_attn","c_proj" \
    --output_dir /home/hn/Documents/enar-datafest-2024/fine-tune_1-16-24/datafest/Qwen/1.8B/lora/no_chol \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16 \
    --overwrite_output_dir \
    --cache_dir ./cache/Qwen/1.8B/lora-2




CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --use_fast_tokenizer True \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen-7B \
    --dataset nhanes_train \
    --template default \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_target "c_attn","c_proj" \
    --output_dir /home/hn/Documents/enar-datafest-2024/fine-tune_1-16-24/datafest/Qwen/7B/qlora/with_chol \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16 \
    --overwrite_output_dir \
    --cache_dir ./cache/Qwen/7B/qlora-1 \
    --quantization_bit 8

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --use_fast_tokenizer True \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen-7B \
    --dataset nhanes_train_no_chol \
    --template default \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_target "c_attn","c_proj" \
    --output_dir /home/hn/Documents/enar-datafest-2024/fine-tune_1-16-24/datafest/Qwen/7B/qlora/no_chol \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16 \
    --overwrite_output_dir \
    --cache_dir ./cache/Qwen/7B/qlora-2 \
    --quantization_bit 8
