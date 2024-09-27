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
                 norm_weights: bool = True):
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
        self.n_iterations = n_iterations
        self.n_samples = n_samples
        self.random_state = random_state
        self.bootstrap_samples = []
        self.norm_weights = norm_weights


    def generate_data(self):
        """ Generates a DataFrame with the weights (on the raw benchmarks) for each BBH subtask and model sublist.
        Step by step process:
        1. Get the full post-training eval scores
        2. Get all lists of models with either 1 or 2 models removed (except for the models used for equivalent scale)
        3. For each BBH subtask, and each sublist of models, fit weights and store them

        Returns:
        all_fit_results - pd.DataFrame [n_bbh_subtasks*n_sublists, n_evals]: DataFrame with the weights for each BBH subtask and model sublist.
        """
        from utils.data import get_full_post_training_df, regression_raw_benchmarks_difference_cot_naive_get_kwargs, regression_raw_benchmarks_difference_cot_naive, get_weights_from_fit_results
        from utils.constants import BBH_SUBTASKS

        # 1.
        merged_eval, subtasks = get_full_post_training_df()
        models_list = merged_eval['Model'].unique().tolist()
        print('Models', models_list)

        # 2.
        # The first 2 models are Llama2, used for the equiv scale so they can't be removed
        lists_with_one_model_removed = [models_list[:i] + models_list[i+1:] for i in range(2, len(models_list))] 
        print('Number of lists with one model removed', len(lists_with_one_model_removed))
        lists_with_two_models_removed = [models_list[:i] + models_list[i+1:j] + models_list[j+1:] for i in range(2, len(models_list)) for j in range(i+1, len(models_list))]
        print('Number of lists with two models removed', len(lists_with_two_models_removed))
        all_lists = lists_with_one_model_removed + lists_with_two_models_removed

        # 3.
        all_fit_results = pd.DataFrame()
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

        return all_fit_results


    def generate_samples(self,
                         bbh_subtask: str = 'overall'):
        """
        Generates bootstrap samples. 
        """
        current_directory = os.getcwd()
        from utils.constants import BBH_SUBTASKS
        assert bbh_subtask in BBH_SUBTASKS

        all_fit_results = self.generate_data()

        np.random.seed(self.random_state)
        for _ in range(self.n_iterations):
            subtask_data = all_fit_results[all_fit_results['subtask'] == bbh_subtask]
            data_array = subtask_data.drop(columns=['subtask', 'sublist_id']).to_numpy()
            
            sample = resample(data_array, n_samples=self.n_samples, replace=True)
            sample = sample.reshape((self.n_samples, data_array.shape[1]))
            self.bootstrap_samples.append(sample)


    def compute_statistic(self, 
                          statistic_func,
                          bbh_subtask: str = 'overall'):
        """
        Computes a statistic over the bootstrap samples.

        Parameters:
        statistic_func (function): Function to compute the statistic.

        Returns:
        list: List of statistic values for each bootstrap sample.
        """
        if not self.bootstrap_samples:
            self.generate_samples(bbh_subtask=bbh_subtask)
        
        results = [statistic_func(sample) for sample in self.bootstrap_samples]
        return results


    def summary(self, results):
        """
        Summarizes the bootstrap results.

        Parameters:
        stats (list): List of statistic values.

        Returns:
        dict: Dictionary with mean, standard error, and confidence interval.
        """
        stats = [result[0] for result in results]
        stats_array = np.vstack(stats)
        zero_in_ci_mask = [result[1] for result in results]
        mask_array = np.vstack(zero_in_ci_mask)
        masked = np.ma.array(stats_array, mask=mask_array)

        mean_stat = np.ma.mean(masked, axis=0)
        std_error = np.ma.std(masked, axis=0)

        # Check for non-masked data before calculating the confidence interval
        if masked.count() > 0:
            # There are valid data points
            confidence_interval = np.percentile(masked.compressed(), [2.5, 97.5], axis=0)
        else:
            # Handle the case where all data points are masked
            print('All data points are masked')
            confidence_interval = np.array([np.nan, np.nan])

        # Count the number of times 0 is in the confidence interval
        false_counts = np.sum(mask_array, axis=0)

        
        return {
            'stats': masked,
            'mean': mean_stat,
            'std_error': std_error,
            'confidence_interval': confidence_interval,
            'false_counts': false_counts
        }
    

def average_weights(sample):
    mean_sample = np.mean(sample, axis=0)
    confidence_interval_sample = np.percentile(sample, [2.5, 97.5], axis=0)
    zero_in_ci = np.sign(confidence_interval_sample[0]*confidence_interval_sample[1])
    zero_in_ci_mask = zero_in_ci < 0 # If 0 is contained in the confidence interval, the value will be removed from the mean calculus
    return mean_sample, zero_in_ci_mask


def plot_raw_benchmarks_weights(weights: np.ndarray,
                                metric_list: list,
                                vmax=1
):
    
    fig, ax = plt.subplots(figsize=(18,5))
    sns.heatmap(weights.reshape((1,-1)), annot=True, fmt='.2f', 
                cmap='coolwarm', vmin=-vmax, vmax=vmax, 
                ax=ax, annot_kws={"fontsize":8})
    
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
                           metric_list: list
):
    fig, ax = plt.subplots(figsize=(18,5))
    absolute_values = np.ma.abs(masked)

    ranks = np.ma.masked_all_like(absolute_values)
    # Apply rankdata along the specified axis, ignoring masked values
    for i in range(absolute_values.shape[0]):
        row = absolute_values[i, :]
        ranks[i, ~row.mask] = rankdata(row[~row.mask], method='dense')
    # Compute weights as the mean of ranks across axis 0
    avg_ranks = np.ma.mean(ranks, axis=0)

    # Highest importance correspond to highest rank but we want the opposite
    avg_ranks = np.ma.max(avg_ranks) + 1 - avg_ranks

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
    
    
    for bbh_subtask in tqdm(BBH_SUBTASKS[4:8]):

        bootstrap = Bootstrap(loss=loss,
                            lambda_=lambda_,
                            random_state=42)

        stats = bootstrap.compute_statistic(average_weights,
                                            bbh_subtask=bbh_subtask)
        summary = bootstrap.summary(stats)
        print('Number of times 0 is in the confidence interval', summary['false_counts'])


        from utils.data import get_full_post_training_df
        df, subtasks = get_full_post_training_df()
        df.to_csv('evals_scores.csv', index=False)

        fig = plot_raw_benchmarks_weights(weights=summary['mean'],
                                        metric_list=['ARC-C', 'HellaSwag', 'Winograd', 'TruthfulQA', 'GSM8K', 'XWinograd', 'HumanEval']+subtasks)
        fig.savefig(f'/scratch2/jsalle/ObsScaling/notebooks/figures/raw_weights/weights/weights_{bbh_subtask}_{loss}.png')
        plt.close(fig)

        #fig = plot_weights_ranks(weights=summary['mean'],
        #                         metric_list=['ARC-C', 'HellaSwag', 'Winograd', 'TruthfulQA', 'GSM8K', 'XWinograd', 'HumanEval']+subtasks)

        fig = plot_weights_avg_ranks(masked=summary['stats'],
                                    metric_list=['ARC-C', 'HellaSwag', 'Winograd', 'TruthfulQA', 'GSM8K', 'XWinograd', 'HumanEval']+subtasks)
        fig.savefig(f'/scratch2/jsalle/ObsScaling/notebooks/figures/raw_weights/avgranks/avgranks_{bbh_subtask}_{loss}.png')
        plt.close(fig)