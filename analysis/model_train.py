import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import importlib
from tqdm import tqdm
from functools import lru_cache
from methods.arguments import get_args
from datasets.dataset import *
from utils import *

@lru_cache(maxsize=None)
def get_class(mod_name: str, cls_name: str):
    module = importlib.import_module(mod_name)
    return getattr(module, cls_name)

if __name__ == '__main__':
    args = get_args()
    data_list = [args.dataset_name]
    model_list = [args.model_name]
    seeds = [args.seed]

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

                        ModelLoaderClass = get_class('methods.builder', builder_name)
                        model_loader = ModelLoaderClass(dataloader, model_config, train_config)
                        model_loader.train()