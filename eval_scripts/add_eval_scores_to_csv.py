# Adds the eval scores from the mmlu folder to eval_results/base_llm_benchmark_eval_2.csv
import pandas as pd
import os
import sys
sys.path.append('../')
from utils.helper import load_base_llm_benchmark_eval


def get_evals_from_json(json_folder="./mmlu"
):
    from pathlib import Path
    import json

    mmlu_df = pd.DataFrame()

    for filename in Path(json_folder).rglob('*.json'):

        f = open(filename)
        jsonf = json.load(f)
        results = jsonf['results']
        
        temp_df = pd.DataFrame(columns=['Model']+[t for t in results.keys()])

        for task in results.keys():
            if task != 'model_name':
                temp_df.at[0,task] = results[task]['acc,none']

        if 'model_name' in jsonf:
            model_name = jsonf['model_name']
        elif 'model_name' in results:
            model_name = results['model_name']

        temp_df['Model'] = model_name
        mmlu_df = pd.concat([mmlu_df, temp_df], ignore_index=True, axis=0)

    print('mmlu_df', mmlu_df)
    # Group rows by model
    mmlu_df = mmlu_df.groupby('Model').mean()
    mmlu_df.reset_index(inplace=True)

    if 'model_name' in mmlu_df.columns:
        mmlu_df.drop(columns=['model_name'], inplace=True)
    return mmlu_df



def save_full_eval_df():
    # Load base LLM eval results
    base_llm_benchmark_eval = load_base_llm_benchmark_eval(csv_path="../eval_results/base_llm_benchmark_eval.csv")

    # Add MMLU subtasks
    mmlu_df = get_evals_from_json(json_folder="./mmlu")
    merged_eval = pd.merge(base_llm_benchmark_eval, mmlu_df, on='Model', how='outer')


    # Add MBPP if available
    if os.path.isdir("./mbpp") and len(os.listdir("./mbpp")) != 0:
        mbpp_df = get_evals_from_json(json_folder="./mbpp")
        merged_eval = pd.merge(merged_eval, mbpp_df, on='Model', how='left')

    merged_eval.to_csv("../eval_results/base_llm_benchmark_eval_2.csv", index=False)
    return 


if __name__ == "__main__":
    save_full_eval_df()
    print("Saved full eval results to eval_results/base_llm_benchmark_eval_2.csv")