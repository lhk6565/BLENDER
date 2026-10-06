import os

import random
import numpy as np
import torch
import json

DATA_DISPATCH = {'ColoredMNIST': 3,
                 'RotatedMNIST': 6,
                 'VLCS': 4,
                 'PACS': 4,
                 'OfficeHome': 4,
                 'DomainNet': 6,
                 'TerraIncognita': 4}

MODEL_DISPATCH = {'blender': 'BlenderBuilder',
                  'diva': 'DivaBuilder',
                  'dirt': 'DirtBuilder',
                  'lfme': 'LFMEBuilder',
                  'arith': 'ArithBuilder'}

def set_seed(seed=42):
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_global_seed():
    return _GLOBAL_SEED

def build_save_path(model_name, dataset_name, test_domain, seed, base_dir= "./analyses/models/"):
    folder_path = os.path.join(base_dir, model_name)
    os.makedirs(folder_path, exist_ok=True)
    return f"{folder_path}/{model_name}_{dataset_name}_{test_domain}_{seed}.pt"

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location='cpu'))

def unpack_batch(batch):
        """Handle both (x, y, d) and (x, y) batches."""
        if len(batch) == 3:
            x, y, d = batch
        elif len(batch) == 2:
            x, y = batch
            d = torch.zeros(x.size(0), dtype=torch.long)
        else:
            raise ValueError('Batch must be a tuple of length 2 or 3.')
        return x, y, d

class ModelConfig:
    def __init__(self, model_name: str):
        config_path = f'./methods/configs/{model_name}.json'

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

    def get_param(self, key: str):
        return self.config.get(key)
    
    def resolve(self, dataset_name):
        config_dataloader = self.config[dataset_name]['dataloader']
        config_model = self.config[dataset_name]['model']
        config_train = self.config[dataset_name]['trainer']
        return config_dataloader, config_model, config_train