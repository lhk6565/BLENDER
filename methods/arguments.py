import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='BLENDER: A Domain Generalization Method for Image Classification')
    parser.add_argument('--model_name', type=str, default='blender', choices=['blender', 'diva', 'dirt', 'lfme', 'arith'])
    parser.add_argument('--dataset_name', type=str, default='VLCS', choices=['RotatedMNIST', 'VLCS', 'PACS', 'OfficeHome', 'DomainNet', 'TerraIncognita'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args