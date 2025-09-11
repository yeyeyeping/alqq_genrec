from pathlib import Path
import jsonlines
import os
import pickle
user_cache_path = Path(os.environ.get('USER_CACHE_PATH'))
data_path = Path(os.environ.get('TRAIN_DATA_PATH'))
MAX_TS = 1748907455

item_id2_time_dict = {}
for interactive_records in jsonlines.open(data_path/"seq.jsonl"):
    for u, i, user_feat, item_feat, action_type, ts  in interactive_records:
        if u and user_feat:
            continue
        
        t = (MAX_TS - ts)/60/60/24
        item_id2_time_dict[i] = min(item_id2_time_dict.get(i, MAX_TS), t)


MEAN_TIME = sum(item_id2_time_dict.values()) / len(item_id2_time_dict)
MAX_TIME = max(item_id2_time_dict.values())
MIN_TIME = min(item_id2_time_dict.values())
print(f"MEAN_TIME: {MEAN_TIME}, MAX_TIME: {MAX_TIME}, MIN_TIME: {MIN_TIME}")

# write pkl to file
with open(user_cache_path/"item_id2_time_dict.pkl", "wb") as f:
    pickle.dump(item_id2_time_dict, f)
            
    
