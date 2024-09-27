#!/bin/bash
#SBATCH --job-name=eval_script         
#SBATCH --partition=gpu             
#SBATCH --export=ALL                 
#SBATCH --cpus-per-task=1
##SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=20:00:00
#SBATCH --output=%x-%j.log
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

models=(
    'microsoft/phi-2' 
    'microsoft/phi-1_5' 
    'meta-llama/Meta-Llama-3-8B' 
    'Qwen/Qwen1.5-4B' 
    'Qwen/Qwen1.5-1.8B' 
    'Qwen/Qwen1.5-0.5B' 
    'Qwen/Qwen2-0.5B' 
    'Qwen/Qwen2-1.5B' 
    'Qwen/Qwen2-7B' 
    'meta-llama/Llama-2-7b-hf' 
    'meta-llama/Llama-2-13b-hf' 
    'EleutherAI/pythia-410m-deduped' 
    'EleutherAI/pythia-6.9b-deduped' 
    'EleutherAI/pythia-2.8b-deduped' 
    'EleutherAI/pythia-12b-deduped' 
    'EleutherAI/pythia-70m-deduped' 
    'EleutherAI/pythia-1.4b-deduped' 
    'EleutherAI/pythia-1b-deduped'
    'EleutherAI/pythia-160m-deduped' 
    'google/gemma-2b' 
    'google/gemma-7b' 
    'tiiuae/falcon-7b' 
    'tiiuae/falcon-rw-1b' 
    'deepseek-ai/deepseek-coder-1.3b-base' 
    'deepseek-ai/deepseek-coder-6.7b-base' 
    'huggyllama/llama-7b' 
    'huggyllama/llama-13b'
)


for model in "${models[@]}"; do
    echo "Starting $model"
    /scratch2/jsalle/.conda/envs/p11/bin/python3 mmlu_subtasks_eval.py --model $model 
    echo "Done with $model"
done

/scratch2/jsalle/.conda/envs/p11/bin/python3 add_eval_scores_to_csv.py