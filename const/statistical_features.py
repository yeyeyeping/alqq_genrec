import os
import pickle
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np

def get_statistical_features(data_dir: str, cache_dir: str):
    """
    计算或加载统计特征。

    该函数首先检查缓存目录中是否存在预先计算好的统计特征文件 ('statistical_features.pkl')。
    如果存在，则直接加载并返回。
    如果不存在，它将遍历整个数据集 ('seq.jsonl') 来计算以下特征：
    - 全局物品ID ('100') 的热度。
    - 全局物品特征 ('101') 中每个值的热度。
    - 每个用户历史中，物品特征 ('101') 出现最频繁的值。
    - 每个用户历史中，交叉特征 ('101', '102') 出现最频繁的值。
    - 每个用户历史中，物品特征 ('101') 的值分布。

    计算完成后，结果将被保存到缓存目录中，以避免未来重复计算。

    参数:
    data_dir (str): 包含 'seq.jsonl' 的数据目录路径。
    cache_dir (str): 用于存储或读取缓存特征文件的目录路径。

    返回:
    dict: 包含所有计算出的统计特征的字典。
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = Path(cache_dir) / 'statistical_features.pkl'

    if os.path.exists(cache_path):
        print(f"Loading statistical features from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    seq_file = Path(data_dir) / 'seq.jsonl'
    if not os.path.exists(seq_file):
        print(f"Warning: '{seq_file}' not found and cache is empty.")
        print("Returning empty statistical features. This may impact model performance if you are in a training environment.")
        empty_features = {
            'global_id_popularity_100': {},
            'global_value_popularity_101': {},
            'user_most_freq_value_101': {},
            'user_cross_freq_101_102': {},
            'user_value_dist_101': {},
            'cross_freq_101_102_indexer': {},
        }
        print(f"Saving empty statistical features to cache to avoid re-computation: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(empty_features, f)
        return empty_features

    print("Calculating statistical features from scratch...")
    
    # Global stats
    global_id_popularity_100 = Counter()
    global_value_popularity_101 = Counter()

    # User-specific features
    user_most_freq_value_101 = {}
    user_value_dist_101 = {}
    user_cross_freq_101_102 = {}

    # Temporary storage for a single user's history
    current_user_id = None
    user_item_history = []
    line_count = 0

    def process_user_history(user_id, items):
        """Helper function to process features for a single user."""
        if not items:
            return

        # Calculate user_most_freq_value_101 and user_value_dist_101
        values_101 = [item.get('101') for item in items if item.get('101') is not None]
        if values_101:
            counter_101 = Counter(values_101)
            user_most_freq_value_101[user_id] = counter_101.most_common(1)[0][0]
            user_value_dist_101[user_id] = dict(counter_101)

        # Calculate user_cross_freq_101_102
        cross_values = []
        for item in items:
            val_101 = item.get('101')
            val_102 = item.get('102')
            if val_101 is not None and val_102 is not None:
                cross_values.append(f"{val_101}_{val_102}")
        
        if cross_values:
            user_cross_freq_101_102[user_id] = Counter(cross_values).most_common(1)[0][0]

    with open(seq_file, 'r') as f:
        for line in f:
            line_count += 1
            if line_count % 100000 == 0:
                print(f"Processed {line_count} lines...")

            user_seq = json.loads(line)
            if not user_seq:
                continue
            
            # Assuming user_id is consistent within a single line (user_seq)
            user_id = user_seq[0][0]

            # If user changes, process the history of the previous user
            if current_user_id is not None and user_id != current_user_id:
                process_user_history(current_user_id, user_item_history)
                user_item_history = []

            current_user_id = user_id
            
            for record in user_seq:
                u, i, user_feat, item_feat, action_type, ts = record
                if i and item_feat:  # Process only item interactions
                    item_id_100 = item_feat.get('100')
                    item_val_101 = item_feat.get('101')
                    
                    if item_id_100 is not None:
                        global_id_popularity_100[item_id_100] += 1
                    
                    if item_val_101 is not None:
                        global_value_popularity_101[item_val_101] += 1
                    
                    user_item_history.append(item_feat)

    # Process the last user's history after the loop finishes
    if current_user_id is not None and user_item_history:
        process_user_history(current_user_id, user_item_history)

    print("Finished processing all lines.")

    # --- Feature Post-processing and Indexing ---
    
    # 1. Create indexer for user_cross_freq_101_102
    all_cross_freq_values = sorted(list(set(user_cross_freq_101_102.values())))
    cross_freq_101_102_indexer = {val: i + 1 for i, val in enumerate(all_cross_freq_values)}
    user_cross_freq_101_102_indexed = {
        user_id: cross_freq_101_102_indexer.get(val, 0)
        for user_id, val in user_cross_freq_101_102.items()
    }
    
    # 2. Log-transform and bucketize popularity features
    def bucketize_popularity(pop_dict):
        # Using np.log1p for safe log transformation (handles zeros)
        # Adding 1 to bucket index to reserve 0 for padding/unknown
        return {k: int(np.log1p(v)) + 1 for k, v in pop_dict.items()}

    global_id_popularity_100_bucketized = bucketize_popularity(global_id_popularity_100)
    global_value_popularity_101_bucketized = bucketize_popularity(global_value_popularity_101)


    statistical_features = {
        'global_id_popularity_100': global_id_popularity_100_bucketized,
        'global_value_popularity_101': global_value_popularity_101_bucketized,
        'user_most_freq_value_101': user_most_freq_value_101,
        'user_cross_freq_101_102': user_cross_freq_101_102_indexed,
        'user_value_dist_101': user_value_dist_101,
        'cross_freq_101_102_indexer': cross_freq_101_102_indexer,
    }

    print(f"Saving statistical features to cache: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(statistical_features, f)

    return statistical_features

if __name__ == '__main__':
    # Add a simple test case
    # You need to create a dummy data_dir and seq.jsonl for this to run
    # For example: /Users/admin/alqq_genrec/TencentGR_1k
    data_directory = './TencentGR_1k' 
    cache_directory = os.environ.get('CACHE_DIR', './cache')
    
    # Create dummy data for testing if it doesn't exist
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    if not os.path.exists(os.path.join(data_directory, 'seq.jsonl')):
        with open(os.path.join(data_directory, 'seq.jsonl'), 'w') as f:
            # user 1
            f.write(json.dumps([
                [1, 10, {}, {'100': 10, '101': 1, '102': 1}, 1, 1000],
                [1, 11, {}, {'100': 11, '101': 1, '102': 2}, 1, 1001],
                [1, 12, {}, {'100': 12, '101': 2, '102': 1}, 1, 1002],
            ]) + '\n')
            # user 2
            f.write(json.dumps([
                [2, 10, {}, {'100': 10, '101': 1, '102': 2}, 1, 1003],
                [2, 13, {}, {'100': 13, '101': 3, '102': 1}, 1, 1004],
            ]) + '\n')


    features = get_statistical_features(data_directory, cache_directory)
    print("Features computed successfully!")
    # Print some stats to verify
    print(f"Size of cross_freq_101_102 vocab: {len(features['cross_freq_101_102_indexer'])}")
    max_pop_100 = max(features['global_id_popularity_100'].values()) if features['global_id_popularity_100'] else 0
    max_pop_101 = max(features['global_value_popularity_101'].values()) if features['global_value_popularity_101'] else 0
    print(f"Max bucket for global_id_popularity_100: {max_pop_100}")
    print(f"Max bucket for global_value_popularity_101: {max_pop_101}")

    for key, value in features.items():
        if isinstance(value, dict):
            print(f"--- {key} ---")
            # Print first 5 items for brevity
            print(dict(list(value.items())[:5]))