import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F

import importlib
import pandas as pd
from tqdm import tqdm
from functools import lru_cache

from datasets.dataset import *
from utils import *


class PerformanceEvaluator:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @lru_cache(maxsize=None)
    def _get_class(mod_name: str, cls_name: str):
        module = importlib.import_module(mod_name)
        return getattr(module, cls_name)
    
    @staticmethod
    def run(data_list, model_list, seeds, save_path='./analyses/results/'):
        results = []
        header = ['dataset', 'test_domain', 'model_name', 'seed', 'test_accuracy']

        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, 'model_performance.csv')

        for seed in tqdm(seeds, desc='Seeds', leave=False):
            set_seed(seed)

            for dataset_name in tqdm(data_list, desc='Datasets', leave=False):
                if dataset_name in DATA_DISPATCH:
                    num_domains = DATA_DISPATCH[dataset_name]
                    domains = list(range(num_domains))
                else:
                    raise ValueError(f'Unknown dataset {dataset_name}.')
                
                for test_domain in tqdm(domains, desc='Test Domains', leave=False):
                    test_domain = [test_domain]

                    for model_name in tqdm(model_list, desc='Models', leave=False):
                        if model_name in MODEL_DISPATCH:
                            builder_name = MODEL_DISPATCH[model_name]

                            cfg = ModelConfig(model_name)
                            loader_config, model_config, train_config = cfg.resolve(dataset_name)
                            model_config['model_name'] = model_name
                            dataloader = ImageDataLoader(dataset_name, test_domain, loader_config)

                            ModelLoaderClass = PerformanceEvaluator._get_class('methods.builder', builder_name)
                            model_loader = ModelLoaderClass(dataloader, model_config, train_config)
                            model_loader.model.to(model_loader.device)

                            load_model(model_loader.model, model_loader.model_path)
                            model_loader.model.eval()
                            
                            te_acc = model_loader.evaluate_accuracy(model_loader.test_loader)

                            results.append([dataset_name, test_domain[0], model_name, seed, te_acc])

        df = pd.DataFrame(results, columns=header)
        df.to_csv(file_path, index=False)
        print(f'[SAVED] {file_path}')

        df_domain_avg = (df.groupby(['dataset', 'model_name', 'test_domain']).agg(mean_acc=('test_accuracy', 'mean'),
                                                                                  std_acc=('test_accuracy', 'std')).reset_index())
        df_domain_avg['mean_acc_str'] = df_domain_avg['mean_acc'].map(lambda x: f"{x*100:.1f}" if pd.notna(x) else "")
        df_domain_avg['std_acc_str'] = df_domain_avg['std_acc'].map(lambda x: f"{x*100:.1f}" if pd.notna(x) else "")
        df_domain_avg.to_csv(os.path.join(save_path, 'model_performance_per_domain.csv'), index=False)
        print(f'[SAVED] {os.path.join(save_path, "model_performance_per_domain.csv")}')

        df_seed_avg = (df.groupby(['dataset', 'model_name', 'seed'])['test_accuracy'].mean().reset_index())
        avg_performance = (df_seed_avg.groupby(['dataset', 'model_name']).agg(mean_acc=('test_accuracy', 'mean'),
                                                                              std_acc=('test_accuracy', 'std')).reset_index())
        avg_performance['mean_acc_str'] = avg_performance['mean_acc'].map(lambda x: f"{x*100:.1f}" if pd.notna(x) else "")
        avg_performance['std_acc_str'] = avg_performance['std_acc'].map(lambda x: f"{x*100:.1f}" if pd.notna(x) else "")
        avg_performance.to_csv(os.path.join(save_path, 'model_performance_avg.csv'), index=False)
        print(f'[SAVED] {os.path.join(save_path, "model_performance_avg.csv")}')


if __name__ == '__main__':

    data_list = ['VLCS', 'PACS', 'OfficeHome', 'DomainNet', 'TerraIncognita']
    model_list = ['blender', 'diva', 'dirt', 'lfme', 'arith']
    seeds = [41, 42, 43, 44, 45]

    PerformanceEvaluator.run(data_list, model_list, seeds, save_path='./analyses/results/')