import numpy as np
from sklearn.utils import resample
from scipy.stats import rankdata
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_directory)
os.chdir('..')
current_directory = os.getcwd()

class Bootstrap:

    def __init__(self, 
                 loss,
                 lambda_: float = None,
                 n_iterations: int = 100, 
                 n_samples: int = 50,
                 random_state: int = None,
                 norm_weights: bool = True,
                 bbh_subtasks: bool = True,
                 tasks: list[str] = None):
        """
        Initializes the Bootstrap object.

        Parameters:
        data (array-like): The dataset to bootstrap.
        n_iterations (int): Number of bootstrap iterations.
        random_state (int): Seed for reproducibility.
        """

        assert loss in ['robust', 'ridge', 'lasso']
        self.loss = loss
        self.lambda_ = lambda_
        self.random_state = random_state
        self.bootstrap_samples = []
        self.norm_weights = norm_weights
        self.bbh_subtasks = True
        self.tasks = tasks


    def generate_data(self):
        """ Generates a DataFrame with the weights (on the raw benchmarks) for each BBH subtask and model sublist.
        Step by step process:
        1. Get the full post-training eval scores
        2. Get all lists of models with either 1 or 2 models removed (except for the models used for equivalent scale)
        3. For each BBH subtask, and each sublist of models, fit weights and store them

        Returns:
        all_fit_results - pd.DataFrame [n_bbh_subtasks*n_sublists, n_evals]: DataFrame with the weights for each BBH subtask and model sublist.
        """
        from utils.data import get_full_post_training_df, regression_raw_benchmarks_difference_cot_naive_get_kwargs, regression_raw_benchmarks_difference_cot_naive, get_weights_from_fit_results, regression_pca_safety_benchmarks
        from utils.helper import load_base_llm_benchmark_eval
        from utils.constants import BBH_SUBTASKS

        # 1.
        merged_eval, subtasks = get_full_post_training_df()

        if not self.bbh_subtasks:
            base_llm_benchmark_eval = load_base_llm_benchmark_eval()
            safety_eval = pd.read_csv('/scratch2/jsalle/ObsScaling/eval_results/safety_evals.csv')
            merged_eval = pd.merge(base_llm_benchmark_eval, safety_eval, on='Model')

            for col in ['wmdp', 'wmdp_chem', 'wmdp_cyber', 'wmdp_bio', 'sycophancy_on_nlp_survey', 'sycophancy_on_philpapers2020', 'sycophancy_on_political_typology_quiz']:
                merged_eval[col] = (merged_eval[col] - merged_eval[col].min()) / (merged_eval[col].max() - merged_eval[col].min())

            subtasks = ['ARC-C', 'HellaSwag', 'Winograd', 'TruthfulQA', 'GSM8K', 'XWinograd', 'HumanEval', 'GSM8K']

        models_list = merged_eval['Model'].unique().tolist()

        # 2.
        # The first 2 models are Llama2, used for the equiv scale so they can't be removed
        lists_with_one_model_removed = [models_list[:i] + models_list[i+1:] for i in range(2, len(models_list))] 
        print('Number of lists with one model removed', len(lists_with_one_model_removed))
        lists_with_two_models_removed = [models_list[:i] + models_list[i+1:j] + models_list[j+1:] for i in range(2, len(models_list)) for j in range(i+1, len(models_list))]
        print('Number of lists with two models removed', len(lists_with_two_models_removed))
        all_lists = lists_with_one_model_removed + lists_with_two_models_removed

        # 3.
        all_fit_results = pd.DataFrame()

        if self.bbh_subtasks:
            for bbh_subtask in BBH_SUBTASKS:
                regression_kwargs = regression_raw_benchmarks_difference_cot_naive_get_kwargs(bbh_task=bbh_subtask,
                                                                                            other_subtasks=subtasks,
                                                                                            loss=self.loss,
                                                                                            lambda_=self.lambda_)
                
                for s, model_sublist in enumerate(lists_with_one_model_removed):
                    sub_merged_eval = merged_eval[merged_eval['Model'].isin(model_sublist)]
                    fit_results = regression_raw_benchmarks_difference_cot_naive(base_llm_eval_with_post_training=sub_merged_eval,
                                                                                regression_kwargs=regression_kwargs)
                    weights = get_weights_from_fit_results(fit_results=fit_results,
                                                        regression_kwargs=regression_kwargs,
                                                        norm_weights=self.norm_weights)
                    temp_df = {
                        'subtask': bbh_subtask,
                        'sublist_id': s
                    }
                    for i, weight in enumerate(weights[0]):
                        temp_df[regression_kwargs['metric_list'][i]] = weight
                    all_fit_results = pd.concat([all_fit_results, pd.DataFrame(temp_df, index=[0])], ignore_index=True)

        else:
            for task_to_predict in self.tasks:
                for s, model_sublist in enumerate(lists_with_one_model_removed):
                    sub_merged_eval = merged_eval[merged_eval['Model'].isin(model_sublist)]
                    fit_results = regression_pca_safety_benchmarks(evals=sub_merged_eval,
                                                                task_to_predict=task_to_predict)

                    weights = get_weights_from_fit_results(fit_results=fit_results,
                                                        regression_kwargs={'metric_list': subtasks},
                                                        norm_weights=self.norm_weights)
                    
                    temp_df = {
                        'subtask': task_to_predict,
                        'sublist_id': s
                    }
                    for i, weight in enumerate(weights[0]):
                        temp_df[subtasks[i]] = weight
                    all_fit_results = pd.concat([all_fit_results, pd.DataFrame(temp_df, index=[0])], ignore_index=True)

        return all_fit_results


    def generate_samples(self,
                         subtask: str = 'overall'):
        """
        Generates bootstrap samples. 
        """
        all_fit_results = self.generate_data()

        subtask_data = all_fit_results[all_fit_results['subtask'] == subtask]
        print('subtask_data', subtask_data)
                                           
        sample = subtask_data.drop(columns=['subtask', 'sublist_id']).to_numpy()
        self.bootstrap_samples.append(sample)
        return


    def compute_statistic(self, 
                          statistic_func,
                          subtask: str = 'overall'):
        """
        Computes a statistic over the bootstrap samples.

        Parameters:
        statistic_func (function): Function to compute the statistic.

        Returns:
        list: List of statistic values for each bootstrap sample.
        """
        if not self.bootstrap_samples:
            self.generate_samples(subtask=subtask)
        
        #results = [statistic_func(sample) for sample in self.bootstrap_samples]

        return self.bootstrap_samples[0]


    def summary(self, results):
        """
        Summarizes the bootstrap results.

        Parameters:
        stats (list): List of statistic values.

        Returns:
        dict: Dictionary with mean, standard error, and confidence interval.
        """

        avg_weights = np.mean(results, axis=0)
        std_error = np.std(results, axis=0)

        print('results', results)
        confidence_interval = np.percentile(results, [2.5, 97.5], axis=0)
        
        return {
            'stats': results,
            'mean': avg_weights,
            'std_error': std_error,
            'confidence_interval': confidence_interval,
        }
    

def average_weights(sample):
    mean_sample = np.mean(sample, axis=0)
    confidence_interval_sample = np.percentile(sample, [2.5, 97.5], axis=0)
    zero_in_ci = np.sign(confidence_interval_sample[0]*confidence_interval_sample[1])
    zero_in_ci_mask = zero_in_ci < 0 # If 0 is contained in the confidence interval, the value will be removed from the mean calculus
    return mean_sample, zero_in_ci_mask


def plot_raw_benchmarks_weights(weights: np.ndarray,
                                ci: np.ndarray,
                                metric_list: list,
                                vmax=1,
                                df: pd.DataFrame = None
):
    
    fig, ax = plt.subplots(figsize=(18,5))
    sns.heatmap(weights.reshape((1,-1)), annot=True, fmt='.2f', 
                cmap='coolwarm', vmin=-vmax, vmax=vmax, 
                ax=ax, annot_kws={"fontsize":8})
    
    for i in range (len(metric_list)):
        if ci[0][i] * ci[1][i] < 0: # O is in CI
            metric_list[i] = metric_list[i] + '*'

    #if df is not None:
        

    ax.set_xticks(np.arange(len(metric_list)) + 0.5)
    ax.set_xticklabels(metric_list, rotation=90, ha="center")
    ax.set_yticklabels([''])
    fig.tight_layout()
    return fig


def plot_weights_ranks(weights: np.ndarray,
                       metric_list: list
):
    fig, ax = plt.subplots(figsize=(18,5))
    reshaped_weights = weights.reshape((1,-1))
    absolute_values = np.abs(reshaped_weights)
    ranks = rankdata(absolute_values, method='dense')
    ranked_array = ranks.reshape(reshaped_weights.shape)
    sns.heatmap(ranked_array.reshape((1,-1)), annot=True, fmt='.2f', 
                cmap='coolwarm', ax=ax, annot_kws={"fontsize":8})
    
    ax.set_xticks(np.arange(len(metric_list)) + 0.5)
    ax.set_xticklabels(metric_list, rotation=90, ha="center")
    ax.set_yticklabels([''])
    fig.tight_layout()
    return fig


def plot_weights_avg_ranks(masked: np.ndarray,
                           ci: np.ndarray,
                           metric_list: list
):
    fig, ax = plt.subplots(figsize=(18,5))
    absolute_values = np.abs(masked)
    ranks = rankdata(absolute_values, method='dense', axis=1)

    avg_ranks = np.ma.mean(ranks, axis=0)
    # Highest importance correspond to highest rank but we want the opposite
    avg_ranks = np.ma.max(avg_ranks) + 1 - avg_ranks

    for i in range (len(metric_list)):
        if ci[0][i] * ci[1][i] < 0: # O is in CI
            metric_list[i] = metric_list[i] + '*'

    sns.heatmap(avg_ranks.reshape((1,-1)), annot=True, fmt='.2f', 
                cmap='coolwarm', ax=ax, annot_kws={"fontsize":8})
    
    ax.set_xticks(np.arange(len(metric_list)) + 0.5)
    ax.set_xticklabels(metric_list, rotation=90, ha="center")
    ax.set_yticklabels([''])
    fig.tight_layout()
    return fig




if __name__ == "__main__":
    from utils.constants import BBH_SUBTASKS

    loss = 'robust'
    lambda_ = 1e-2
    bbh_subtasks = False
    
    
    #### FOR BBH SUBTASKS
    if bbh_subtasks:
        for subtask in tqdm(BBH_SUBTASKS):

            bootstrap = Bootstrap(loss=loss,
                                lambda_=lambda_,
                                random_state=42)

            stats = bootstrap.compute_statistic(average_weights,
                                                subtask=subtask)
            summary = bootstrap.summary(stats)


            from utils.data import get_full_post_training_df
            _, subtasks = get_full_post_training_df()

            fig = plot_raw_benchmarks_weights(weights=summary['mean'],
                                            ci=summary['confidence_interval'],
                                            metric_list=['ARC-C', 'HellaSwag', 'Winograd', 'TruthfulQA', 'GSM8K', 'XWinograd', 'HumanEval']+subtasks)
            fig.savefig(f'/scratch2/jsalle/ObsScaling/notebooks/figures/deterministic/weights_{subtask}_{loss}.png')
            plt.close(fig)

            #fig = plot_weights_ranks(weights=summary['mean'],
            #                         metric_list=['ARC-C', 'HellaSwag', 'Winograd', 'TruthfulQA', 'GSM8K', 'XWinograd', 'HumanEval']+subtasks)

            fig = plot_weights_avg_ranks(masked=summary['stats'],
                                        ci=summary['confidence_interval'],
                                        metric_list=['ARC-C', 'HellaSwag', 'Winograd', 'TruthfulQA', 'GSM8K', 'XWinograd', 'HumanEval']+subtasks)
            fig.savefig(f'/scratch2/jsalle/ObsScaling/notebooks/figures/deterministic/avgranks_{subtask}_{loss}.png')   
            plt.close(fig)


    #### FOR SAFETY TASKS
    else:
        safety_tasks = ['wmdp', 'sycophancy_on_nlp_survey', 'sycophancy_on_philpapers2020', 'sycophancy_on_political_typology_quiz', 'wmdp_bio', 'wmdp_chem', 'wmdp_cyber']
        
        for subtask in tqdm(safety_tasks):
            bootstrap = Bootstrap(loss=loss,
                                  lambda_=lambda_,
                                  random_state=42,
                                  bbh_subtasks=False,
                                  tasks=subtask)

            stats = bootstrap.compute_statistic(average_weights,
                                                subtask=subtask)
            summary = bootstrap.summary(stats)

            subtasks = ['ARC-C', 'HellaSwag', 'Winograd', 'TruthfulQA', 'GSM8K', 'XWinograd', 'HumanEval']

            fig = plot_raw_benchmarks_weights(weights=summary['mean'],
                                            ci=summary['confidence_interval'],
                                            metric_list=['ARC-C', 'HellaSwag', 'Winograd', 'TruthfulQA', 'GSM8K', 'XWinograd', 'HumanEval'])
            fig.savefig(f'/scratch2/jsalle/ObsScaling/notebooks/figures/deterministic/weights_{subtask}_{loss}.png')
            plt.close(fig)
