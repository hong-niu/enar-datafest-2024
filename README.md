# ENAR-Datafest-2024

Entry for the 2024 ENAR Datafest. 

## Report

PDF of submission report can be found in /report/. 

## Code dependencies

The main dependencies for this project are LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory), used for the fine-tuning of LLM models, as well as the scikit-learn package to perform comparisons to traditional classification and regression models. 

## Data Preprocessing

The core data comes from the aggregated NHANES survey data from the CardioStatsUSA package (https://github.com/jhs-hwg/cardioStatsUSA/tree/main). From this data, we restricted our analysis to just respondents with confirmed hypertension. Further pre-processing of the final tabular dataset includes converting into natural language input/output prompts for the supervised LLM fine-tuning. 

## Fine-Tuning (for Report)

After preprocessing the input data into the respective JSON files, fine-tuning can be performed in LLaMA-Factory from the LLaMA-Factory directory and running the following: 

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
      --output_dir /output/directory/ \
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
      --cache_dir /cache/directory



## Colab 
For ease of use, we prepared a completely self-contained Colab notebook with a demonstration of loading and querying LLM models using LLaMA-Factory. 
