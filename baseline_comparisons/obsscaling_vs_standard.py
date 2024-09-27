import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm as tqdm
import sys
sys.path.append('../')
from utils import *




class ScalingLawsComparison():

    def __init__(self,
                 benchmarks_to_predict_dict: dict,
                 base_benchmarks: list[str],
                 thresholds: list[float],
                 standard_vs_obsscaling: bool = False,
                 lambdas_: list[float] = [0.001, 0.01, 0.05,  0.1, 1.0, 3.0],
                 savefigs: bool = False,
    ):
        
        self.benchmarks_to_predict_dict = benchmarks_to_predict_dict
        self.benchmarks_to_predict = list(benchmarks_to_predict_dict.keys())
        self.base_benchmarks = base_benchmarks
        self.thresholds = thresholds
        self.standard_vs_obsscaling = standard_vs_obsscaling
        self.lambdas_ = lambdas_
        self.errors = pd.DataFrame()
        self.savefigs = savefigs
        self.evals = self.get_eval_data()
        return


    def get_eval_data(self,
                      eval_file: str= '../eval_results/safety_evals.csv',
    ) -> pd.DataFrame:
        
        base_llm_benchmark_eval = load_base_llm_benchmark_eval(csv_path="../eval_results/base_llm_benchmark_eval.csv")

        try:
            safety_eval = pd.read_csv(eval_file)
        except FileNotFoundError:
            print(f"File {eval_file} not found")
            return None
        
        evals = pd.merge(base_llm_benchmark_eval, safety_eval, on='Model')
        for col in ['wmdp', 'wmdp_chem', 'wmdp_cyber', 'wmdp_bio', 'sycophancy_on_nlp_survey', 'sycophancy_on_philpapers2020', 'sycophancy_on_political_typology_quiz']:
            evals[col] = (evals[col] - evals[col].min()) / (evals[col].max() - evals[col].min())

        # TODO: add MBPP

        # Check that all necessary benchmarks are present in the evals DataFrame
        required_columns_exist = all([col in evals.columns for col in self.benchmarks_to_predict+self.base_benchmarks])
        #print([col in evals.columns for col in self.benchmarks_to_predict+self.base_benchmarks], self.benchmarks_to_predict+self.base_benchmarks)
        if not required_columns_exist:
            print("Some columns are missing in the DataFrame.")

        # TODO: check missing values
        return evals
    

    def get_scaling_kwargs(self, 
                           benchmark_to_predict: str, 
                           benchmark_to_predict_alias: str,
                           cutoff_threshold: float = 84,
                           loss: str = "robust",
                           lambda_: float = 0.1,
                           apply_pca: bool = True
    ) -> tuple[list[str], dict, dict, dict, dict]:

        pca_preprocess_kwargs = copy.deepcopy(DEFAULT_PCA_PREPROCESS_KWARGS)
        pca_preprocess_kwargs["imputation_metrics"] = copy.deepcopy(self.base_benchmarks)
        pca_preprocess_kwargs["pca_metrics"] = copy.deepcopy(self.base_benchmarks)

        if benchmark_to_predict in self.base_benchmarks:
            pca_preprocess_kwargs["imputation_metrics"].remove(benchmark_to_predict)
            pca_preprocess_kwargs["pca_metrics"].remove(benchmark_to_predict)

        scaling_setup_kwargs = {
            # Data preprocessing: PCA imputation and extraction
            **pca_preprocess_kwargs,
            
            # Non-lineariy: sigmoid with parametrized scale and shift
            "nonlinearity": "sigmoid", 

            # Cutoff: 8.4E22 FLOPs corresponding to LLama-2 7B
            "split_method": "cutoff_by_FLOPs (1E21)",
            "cutoff_threshold": cutoff_threshold,

            # Model families: include all we have evaled
            "df_filter": lambda x: x['Model Family'].isin(EVAL_BASE_MODEL_FAMILIES),   
            "df_groupby": 'Model Family',  # group markers by model family

            # Regression: ordinary least squares
            "reg_method": loss,  
            "reg_kwargs": {"delta": 1.0,
                           "lambda": lambda_},  # huber loss with delta=1.0 for normalized target within [0, 1] reduces to OLS
        }

        if not apply_pca:
            scaling_setup_kwargs['apply_pca'] = False

        metric_map = {
           benchmark_to_predict: benchmark_to_predict_alias,
        }
        scaling_orig_metric_map = {
            **metric_map
        }
        scaling_metrics = list(metric_map.keys())
        scaling_metrics = list(metric_map.keys())
        color_palette = sns.color_palette()

        scaling_color_map = {
            benchmark_to_predict: color_palette[1]
        }


        setup_specific_kwargs = {}
        for e in scaling_metrics:
            setup_specific_kwargs[e] = {
                "plot_adjust_kwargs": {"title": scaling_orig_metric_map[e], "ylabel": "Accuracy"},
            }

        return scaling_metrics, scaling_setup_kwargs, setup_specific_kwargs, scaling_orig_metric_map, scaling_color_map


    def generate_size_based_scaling_law(self,
                                        benchmark_to_predict: str,
                                        benchmark_to_predict_alias: str,
                                        cutoff_threshold: float = 84
    ):

        [
            scaling_metrics, 
            scaling_setup_kwargs, 
            setup_specific_kwargs, 
            scaling_orig_metric_map, 
            scaling_color_map
        ] = self.get_scaling_kwargs(benchmark_to_predict=benchmark_to_predict, 
                                    benchmark_to_predict_alias=benchmark_to_predict_alias,
                                    cutoff_threshold=cutoff_threshold)

        fig, all_fit_results, metric = plot_scaling_comparison_multi_metrics(
            eval_df=self.evals, 
            y_metric_list=scaling_metrics, 
            x_metrics_list=[MODEL_SIZE_METRIC], 
            analysis_setup_kwargs=scaling_setup_kwargs, 
            y_metric_specific_kwargs=setup_specific_kwargs,
            ymetric2title_map=scaling_orig_metric_map, 
            ymetric2color_map={benchmark_to_predict: sns.color_palette()[0]},
            plot_title="Size Based Scaling Law",
            legend_offset=0.8
        )
        if self.savefigs:
            plt.savefig(f'./figures/threshold_{cutoff_threshold}/{benchmark_to_predict}_size.png')

        # Add MAE and MSE to self.errors
        errors_temp = pd.DataFrame({'benchmark_to_predict': benchmark_to_predict,
                                  'law': 'Model Size',
                                  'threshold': cutoff_threshold,
                                  'mae_train': metric['mae_train'],
                                  'mae_test': metric['mae_test'],
                                  'mse_train': metric['mse_train'],
                                  'mse_test': metric['mse_test']}, index=[0])
        self.errors = pd.concat([self.errors, errors_temp], axis=0)
        return


    def generate_flops_based_scaling_law(self,
                                        benchmark_to_predict: str,
                                        benchmark_to_predict_alias: str,
                                        cutoff_threshold: float = 84
    ):
    
        [
            scaling_metrics, 
            scaling_setup_kwargs, 
            setup_specific_kwargs, 
            scaling_orig_metric_map, 
            scaling_color_map
        ] = self.get_scaling_kwargs(benchmark_to_predict=benchmark_to_predict, 
                                    benchmark_to_predict_alias=benchmark_to_predict_alias,
                                    cutoff_threshold=cutoff_threshold)

        fig, all_fit_results, metric = plot_scaling_comparison_multi_metrics(
            eval_df=self.evals, 
            y_metric_list=scaling_metrics, 
            x_metrics_list=[TRAINING_FLOPS_METRIC], 
            analysis_setup_kwargs=scaling_setup_kwargs, 
            y_metric_specific_kwargs=setup_specific_kwargs,
            ymetric2title_map=scaling_orig_metric_map, 
            ymetric2color_map={benchmark_to_predict: sns.color_palette()[1]},
            plot_title="FLOPs Based Scaling Law",
            legend_offset=0.8
        )
        if self.savefigs:
            plt.savefig(f'./figures/threshold_{cutoff_threshold}/{benchmark_to_predict}_flops.png')

        # Add MAE and MSE to self.errors
        errors_temp = pd.DataFrame({'benchmark_to_predict': benchmark_to_predict,
                                  'law': 'FLOPs',
                                  'threshold': cutoff_threshold,
                                  'mae_train': metric['mae_train'],
                                  'mae_test': metric['mae_test'],
                                  'mse_train': metric['mse_train'],
                                  'mse_test': metric['mse_test']}, index=[0])
        self.errors = pd.concat([self.errors, errors_temp], axis=0)
        return


    def generate_obs_based_scaling_law(self,
                                       benchmark_to_predict: str,
                                       benchmark_to_predict_alias: str,
                                       cutoff_threshold: float = 84,
                                       loss: str = "robust",
                                       lambda_: float = 0.1,
                                       apply_pca: bool = True
    ):

        [
            scaling_metrics, 
            scaling_setup_kwargs, 
            setup_specific_kwargs, 
            scaling_orig_metric_map, 
            scaling_color_map
        ] = self.get_scaling_kwargs(benchmark_to_predict=benchmark_to_predict, 
                                    benchmark_to_predict_alias=benchmark_to_predict_alias,
                                    cutoff_threshold=cutoff_threshold,
                                    loss=loss,
                                    lambda_=lambda_,
                                    apply_pca=apply_pca)

        fig, all_fit_results, metric = plot_scaling_comparison_multi_metrics(
            eval_df=self.evals, 
            y_metric_list=scaling_metrics, 
            x_metrics_list=[PC_METRIC_NUM_3] if apply_pca else [self.base_benchmarks], 
            analysis_setup_kwargs=scaling_setup_kwargs, 
            y_metric_specific_kwargs=setup_specific_kwargs,
            ymetric2title_map=scaling_orig_metric_map, 
            ymetric2color_map={benchmark_to_predict: sns.color_palette()[2]},
            plot_title="Observational Scaling Law",
            legend_offset=0.8
        )
        if self.savefigs:
            plt.savefig(f'./figures/threshold_{cutoff_threshold}/{benchmark_to_predict}_obs_scaling_{loss}_{lambda_}.png')

        # Add MAE and MSE to self.errors
        errors_temp = pd.DataFrame({'benchmark_to_predict': benchmark_to_predict,
                                  'law': 'Observational',
                                  'threshold': cutoff_threshold,
                                  'lambda_': lambda_,
                                  'loss': loss,
                                  'mae_train': metric['mae_train'],
                                  'mae_test': metric['mae_test'],
                                  'mse_train': metric['mse_train'],
                                  'mse_test': metric['mse_test']}, index=[0])
        self.errors = pd.concat([self.errors, errors_temp], axis=0, ignore_index=True)

        return


    def plot_mae_mse_distributions(self):

        thresholds = self.errors['threshold'].unique().tolist()
        for threshold in thresholds:
            errors_threshold = self.errors[self.errors['threshold'] == threshold]
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            # Plot MAE histograms for each law on the left subplot
            sns.kdeplot(data=self.errors, x='mae_test', hue='law', fill=True, ax=axs[0])
            axs[0].set_title('MAE Test')
            axs[0].set_xlabel('MAE')
            axs[0].set_xlim(left=0)

            # Plot MSE histograms for each law on the right subplot
            sns.kdeplot(data=self.errors, x='mse_test', hue='law', fill=True, ax=axs[1])
            axs[1].set_title('MSE Test')
            axs[1].set_xlabel('MSE')
            axs[1].set_xlim(left=0)

            plt.tight_layout()

            if self.savefigs:
                plt.savefig(f'./figures/threshold_{threshold}/mae_mse_distributions.png')
                print(f'Saved MAE and MSE distributions at ./figures/threshold_{threshold}/mae_mse_distributions.png')
            plt.show()

        return


    def plot_multi_thresold_histograms(self,
                                       mae_or_mse: str = 'mae'):
        plt.figure(figsize=(8, 6))
        sns.barplot(data=self.errors, x='threshold', y=f'{mae_or_mse}_test', hue='law')
        plt.xlabel('Cutoff Threshold (FLOPs)')
        ylabel = 'MAE' if mae_or_mse == "mae" else "MSE"
        plt.ylabel(f'{ylabel}')
        plt.legend(title='Scaling Law Type')

        plt.savefig(f'./figures/{mae_or_mse}_multi_threshold_histograms.png')
        print(f'Saved multi-threshold histograms at ./figures/{mae_or_mse}_multi_threshold_histograms.png')
        return
    

    def plot_multi_reg_histograms(self,
                                  mae_or_mse: str = 'mae'):
        plt.figure(figsize=(8, 6))
        sns.barplot(data=self.errors, x='lambda_', y=f'{mae_or_mse}_test', hue='loss')
        plt.xlabel('Regularization Parameter (lambda)')
        ylabel = 'MAE' if mae_or_mse == "mae" else "MSE"
        plt.ylabel(f'{ylabel}')
        plt.legend(title='Regularization')

        plt.savefig(f'./figures/{mae_or_mse}_reg_histograms.png')
        print(f'Saved multi-regularization histograms at ./figures/{mae_or_mse}_reg_histograms.png')
        return
    

    def run_comparison_standard_vs_obsscaling(self):

        for threshold in tqdm(self.thresholds, desc='Thresholds'):

            # Create dir to save figures if savefigs and not exists
            if self.savefigs:
                os.makedirs(f'./figures/threshold_{threshold}', exist_ok=True)

            for benchmark_to_predict in self.benchmarks_to_predict:
                
                args = {
                    'benchmark_to_predict': benchmark_to_predict,
                    'benchmark_to_predict_alias': self.benchmarks_to_predict_dict[benchmark_to_predict],
                    'cutoff_threshold': threshold
                }

                self.generate_size_based_scaling_law(**args)
                self.generate_flops_based_scaling_law(**args)
                self.generate_obs_based_scaling_law(**args)

            self.plot_mae_mse_distributions()
            self.errors.to_csv(f'./figures/threshold_{threshold}/errors.csv', index=False)

        self.plot_multi_thresold_histograms('mae')
        self.plot_multi_thresold_histograms('mse')
        return


    def run_comparison_obsscaling_regularization(self,
                                                 apply_pca: bool = True):

        for threshold in tqdm(self.thresholds, desc='Thresholds'):

            if self.savefigs:
                os.makedirs(f'./figures/threshold_{threshold}', exist_ok=True)
        
            for lambda_ in self.lambdas_:

                for benchmark_to_predict in self.benchmarks_to_predict:

                    args = {
                        'benchmark_to_predict': benchmark_to_predict,
                        'benchmark_to_predict_alias': self.benchmarks_to_predict_dict[benchmark_to_predict],
                        'cutoff_threshold': threshold,
                        'loss': 'lasso',
                        'lambda_': lambda_,
                        'apply_pca': apply_pca
                    }
                    self.generate_obs_based_scaling_law(**args)

                    args.update({'loss': 'ridge'})
                    self.generate_obs_based_scaling_law(**args)
                
                self.plot_mae_mse_distributions()
            
            self.plot_multi_reg_histograms('mae')
            self.plot_multi_reg_histograms('mse')
        return


    def run_comparison(self,
                       reg_apply_pca: bool = True):

        if self.standard_vs_obsscaling:
            self.run_comparison_standard_vs_obsscaling()
        else: # Compare obsscaling with different regularization methods
            self.run_comparison_obsscaling_regularization(apply_pca=reg_apply_pca)


        return




if __name__ == "__main__":

    base_benchmarks = ['MMLU', 'ARC-C', 'HellaSwag', 'Winograd', 'TruthfulQA', 'XWinograd', 'HumanEval', 'GSM8K']

    benchmarks_to_predict_dict = {
        'MMLU': 'MMLU',
        'ARC-C': 'ARC-C',
        'HellaSwag': 'HellaSwag',
        'Winograd': 'Winograd',
        'TruthfulQA': 'TruthfulQA',
        'XWinograd': 'XWinograd',
        'HumanEval': 'HumanEval',
        'GSM8K': 'GSM8K',
        #'mbpp': 'MBPP',
        'wmdp': 'WMDP', 
        'wmdp_cyber': 'WMDP Cyber',
        'wmdp_chem': 'WMDP Chem',
        'wmdp_bio': 'WMDP Bio',
        'sycophancy_on_nlp_survey': 'Sycophancy on NLP Survey',
        'sycophancy_on_philpapers2020': 'Sycophancy on Philpapers2020',
        'sycophancy_on_political_typology_quiz': 'Sycophancy on Political Typology Quiz',
        # TODO: add BBH subtasks
    }

    cutoff_thresholds = [20]

    scaling_laws_comparison = ScalingLawsComparison(
        benchmarks_to_predict_dict=benchmarks_to_predict_dict,
        base_benchmarks=base_benchmarks,
        thresholds=cutoff_thresholds,
        standard_vs_obsscaling=False,
        savefigs=True
    )
    scaling_laws_comparison.run_comparison(reg_apply_pca=False)
