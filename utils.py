import pickle
import random
import os
import torch
import numpy as np
import json
def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def seed_worker(worker_id,seed=3407):
    seed_everything(seed + worker_id)     

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    np.random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
