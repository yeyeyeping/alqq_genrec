import os
import json
from pathlib import Path
import time
import pickle
import os
import torch
import json
def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
def infer():
    torch.set_grad_enabled(False)
    data_path = Path(os.environ.get('EVAL_DATA_PATH'))
    candidate_path = data_path/ 'predict_set.jsonl'
    indexer = read_pickle(data_path/ "indexer.pkl")['i']
    mm_emb_dict = read_pickle(data_path/"creative_emb" / "emb_81_32.pkl")
    cold_start_item = set()
    cold_start_item_no_emb = set()
    total_item = 0
    with open(candidate_path, 'r') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            creative_id = item['creative_id']
            if creative_id not in indexer:
                cold_start_item.add(creative_id)
            
            if creative_id not in mm_emb_dict:
                cold_start_item_no_emb.add(creative_id)
            total_item += 1
    print(f"cold start item: {len(cold_start_item)} / {total_item}")
    print(f"cold start item no emb: {len(cold_start_item_no_emb)} / {total_item}")
    both_no = cold_start_item & cold_start_item_no_emb
    print(f"both no: {len(both_no)} / {total_item}")
    time.sleep(3000)