from utils import read_pickle
from pathlib import Path
import torch
import gc
import time
import numpy as np
class BaseMmEmbLoader:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        indexer_file = read_pickle(self.data_path / "indexer.pkl")
        self.redi2annoyid = {v:k for k,v in indexer_file['i'].items()}
        
    def convert_reid_to_anoyid(self, redi):
        if redi in self.redi2annoyid:
            return self.redi2annoyid[redi]
        else:
            return -1
    
    def batch_load_emb(self, reid_list):
        pass



# class Memorymm81Embloader:
#     def __init__(self, data_path):
#         self.data_path = Path(data_path)
#         self.feat_size = 32
#         self.mm_emb_dict = self.load_81emb()
    
#     def load_81emb(self):
#         print("Loading 81 embeddings")
#         start_time = time.time()
#         mm_emb_dict = read_pickle(self.data_path/"creative_emb" / "emb_81_32.pkl")
#         indexer_dict = read_pickle(self.data_path / "indexer.pkl")['i']
#         new_mm_emb_dict = {}
        
#         new_mm_emb_dict[0] = torch.zeros(self.feat_size, dtype=torch.float32)
#         for k, v in mm_emb_dict.items():
#             if k not in indexer_dict:
#                 continue
#             if isinstance(v, np.ndarray):
#                 new_mm_emb_dict[indexer_dict[k]] = torch.as_tensor(v, dtype=torch.float32)
#             else:
#                 new_mm_emb_dict[indexer_dict[k]] = torch.zeros(self.feat_size, dtype=torch.float32)
#         del mm_emb_dict, indexer_dict
#         gc.collect()
#         print(f"Loaded {len(new_mm_emb_dict)} embeddings Successfully in {time.time() - start_time} seconds")
#         return new_mm_emb_dict
        
#     def batch_load_emb(self, reid_list):
#         embeddings = [self.mm_emb_dict.get(reid, self.mm_emb_dict[0] ) for reid in reid_list]
#         return torch.stack(embeddings)

#     def add_mm_emb(self, seq_id, feat, item_mask=None):
#         if item_mask is None:
#             item_mask = torch.ones_like(seq_id, dtype=torch.bool)
        
#         mm_emb = self.batch_load_emb((seq_id * item_mask).reshape(-1).tolist())
#         if seq_id.ndim == 2:
#             feat['81'] = mm_emb.reshape(seq_id.shape[0], seq_id.shape[1], self.feat_size)
#         else:
#             feat['81'] = mm_emb.reshape(seq_id.shape[0], self.feat_size)
#         return feat