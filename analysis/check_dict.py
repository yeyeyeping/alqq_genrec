from pathlib import Path
import os
import pickle
import json
def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
def read_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
data_path = Path(os.environ.get('TRAIN_DATA_PATH'))
user_cache_path = Path(os.environ.get('USER_CACHE_PATH'))

topsim_dict = read_pickle(user_cache_path/"annoyid2top20sim_dict.pkl")
item_feat_dict = read_json(data_path/"item_feat_dict.json")

oov = set()
for key_id in item_feat_dict.keys():
    if int(key_id) not in topsim_dict:
        oov.add(key_id)
print(f"oov: {len(oov)}")