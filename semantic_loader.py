import json
from pathlib import Path
from utils import read_pickle
import torch

class SemanticLoader:
    def __init__(self, data_path, semantic_file_path):
        """
        Initializes the loader for semantic features derived from RQ-VAE.

        Args:
            data_path (str): Path to the main data directory, containing indexer.pkl.
            semantic_file_path (str): Path to the semantic_id_map.json file.
        """
        print("Loading semantic features...")
        indexer = read_pickle(Path(data_path) / 'indexer.pkl')
        self.reid_to_creative_id = {v: k for k, v in indexer['i'].items()}

        with open(semantic_file_path, 'r') as f:
            self.semantic_map = json.load(f)
        print("Semantic features loaded.")
        
        self.default_semantic_features = [0] * 8

    def _get_semantic_features(self, reid):
        """
        Gets the 8 semantic features for a single re-identified item_id.
        """
        creative_id = self.reid_to_creative_id.get(reid)
        if creative_id:
            # The keys in semantic_map might be strings
            return self.semantic_map.get(str(creative_id), self.default_semantic_features)
        return self.default_semantic_features

    def add_semantic_features_to_batch(self, batch_reids, feat_dict):
        """
        Adds the 8 semantic features to the feature dictionary for a batch of items.

        Args:
            batch_reids (torch.Tensor): A tensor of shape (batch_size, seq_len) containing re-identified item_ids.
            feat_dict (dict): The feature dictionary to which the new features will be added.

        Returns:
            dict: The updated feature dictionary.
        """
        # Initialize lists to hold the features for the whole batch
        semantic_features_batch = [[] for _ in range(8)]
        
        # Flatten the batch for easier processing
        reids_list = batch_reids.view(-1).tolist()
        
        # Retrieve features for each reid in the flattened list
        for reid in reids_list:
            features = self._get_semantic_features(reid)
            for i in range(8):
                semantic_features_batch[i].append(features[i])
        
        # Reshape the feature lists back to the original batch shape
        original_shape = batch_reids.shape
        for i in range(8):
            feature_tensor = torch.tensor(semantic_features_batch[i], dtype=torch.int32).view(original_shape)
            feat_dict[f'13{i}'] = feature_tensor
            
        return feat_dict
