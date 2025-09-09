from pathlib import Path
import os
import numpy as np
import pickle
import torch
def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

data_path = Path(os.environ.get('TRAIN_DATA_PATH'))
cache_path = Path(os.environ.get('USER_CACHE_PATH'))

emb = data_path/"creative_emb"/"emb_81_32.pkl"

id_list = []
emb_list = []
for k, v in read_pickle(emb).items():
    if isinstance(v, np.ndarray):
        id_list.append(k)
        emb_list.append(torch.as_tensor(v, dtype=torch.float32,device=torch.device('cuda')))
id_tensors = torch.as_tensor(id_list,device=torch.device('cuda'),dtype=torch.int64)
emb_tensors = torch.stack(emb_list)

chunk_size = 1000
top21_list = []
for i in range(0, len(id_list), chunk_size):
    emb_src = emb_tensors[i:i+chunk_size]
    a_id = id_tensors[i:i+chunk_size]
    sim_mat = emb_src @ emb_tensors.T
    
    _, indices = torch.topk(sim_mat, k=21, dim=1)
    top21 = id_tensors[indices].cpu().tolist()
    top21_list.append(top21)
    
for idx, id_ in enumerate(a_id.tolist()):
    if id_ in top21[idx]:
        top21[idx].remove(id_)
        
annoyied_top21 = dict(zip(id_list, top21_list))       
with open(cache_path/"annoyid2top20sim_dict.pkl", "wb") as f:
    pickle.dump(annoyied_top21, f)
print(f"save to {cache_path/'annoyid2top20sim_dict.pkl'}")

