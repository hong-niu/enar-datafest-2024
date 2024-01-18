# ENAR-Datafest-2024

Entry for the 2024 ENAR Datafest. 

## Report

PDF of submission report can be found in /report/. 

## Code Dependencies and Requirements

The main dependencies for this project are LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory), used for the fine-tuning of LLM models, as well as the scikit-learn package to perform comparisons to traditional classification and regression models. All experiments were performed either on an A100 (40 GB) GPU, which was necessary for training LLM's greater than roughly 7B parameters, or desktop RTX 4080 (16 GB) GPU. Running the code requires cloning the LLaMA-Factory repo and installing the required dependencies. 

## Code Layout

The /code directory contains 4 scripts used for the project as follows: 

    /code/  01-data-preprocessing.R 
            02-run_qwen_train.sh 
            03-use_model.py  
            04-traditional_comps.ipynb

01 Contains preprocessing steps such as restricting to patients with hypertension and generating the natural language prompts for input into LLM fine tuning. 02 contains the input and hyperparameters used for LLaMA-Factory fine-tuning. 03 contains code for prompting the retrained model and 04 contains a notebook for performing comparisons to traditional classification models. 

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

Specific parameters for our experiments can be found in the /code/02-run_qwen_train.sh script. 


