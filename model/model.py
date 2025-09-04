import numpy as np
import torch
import torch.nn.functional as F
import const
from torch import nn
from .atten import AttentionDecoder

class BaseTower(nn.Module):
    """
    A base class for feature towers (UserTower, ItemTower).
    It handles the common logic for creating embedding layers and calculating feature dimensions.
    """
    def __init__(self, main_id_name, feature_config, model_param):
        super().__init__()
        self.main_id_name = main_id_name
        self.sparse_feature_ids = feature_config.sparse_feature_ids
        self.array_feature_ids = getattr(feature_config, 'array_feature_ids', [])
        self.dense_feature_ids = feature_config.dense_feature_ids
        self.model_param = model_param
        
        self.sparse_emb = self.setup_embedding_layer()

    def setup_embedding_layer(self):
        """Initializes embedding layers for all sparse and array features."""
        emb_dict = nn.ModuleDict()
        
        # Main ID embedding
        emb_dict[self.main_id_name] = nn.Embedding(
            self.model_param.embedding_table_size[self.main_id_name] + 1, 
            self.model_param.embedding_dim[self.main_id_name], 
            padding_idx=0
        )
        
        # Other sparse and array feature embeddings
        for feat_id in list(self.sparse_feature_ids) + list(self.array_feature_ids):
            emb_dict[feat_id] = nn.Embedding(
                self.model_param.embedding_table_size[feat_id] + 1, 
                self.model_param.embedding_dim[feat_id], 
                padding_idx=0
            )
        return emb_dict

    def get_feature_dim(self):
        """Calculates the total dimension of the concatenated features."""
        dim = self.model_param.embedding_dim[self.main_id_name]
        for feat_id in self.sparse_feature_ids:
            dim += self.model_param.embedding_dim[feat_id]
        for feat_id in self.array_feature_ids:
            # For array features, we use sum pooling, so the dimension is the same as embedding dim
            dim += self.model_param.embedding_dim[feat_id]
        dim += len(self.dense_feature_ids)
        return dim

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Each tower must implement its own forward pass.")


class UserTower(BaseTower):
    """User Tower for processing user features."""
    def __init__(self):
        super().__init__('user_id', const.user_feature, const.model_param)
        self.dnn = nn.Sequential(
            nn.Linear(self.get_user_feature_dim(), const.model_param.user_dnn_units),
            nn.ReLU(),
            nn.Linear(const.model_param.user_dnn_units, const.model_param.hidden_units),
        )
        
    def get_user_feature_dim(self):
        """Calculates the total dimension for user features."""
        num = self.model_param.embedding_dim['user_id']
        for feat_id in self.sparse_feature_ids + self.array_feature_ids:
            num += self.model_param.embedding_dim[feat_id]
        num += len(self.dense_feature_ids)
        return num
        
    def forward(self, seq_id, user_mask, feature_dict):
        """Forward pass for UserTower."""
        # Get embedding and then apply mask to zero out padded positions.
        id_embedding = self.sparse_emb['user_id'](seq_id) * user_mask.unsqueeze(-1)
        
        feat_emb_list = [id_embedding]
        
        for feat_id in self.sparse_feature_ids:
            if feat_id in feature_dict:
                feat_emb_list.append(self.sparse_emb[feat_id](feature_dict[feat_id]))
                feature_dict.pop(feat_id)
        
        for feat_id in self.array_feature_ids:
            if feat_id in feature_dict:
                emb = self.sparse_emb[feat_id](feature_dict[feat_id])
                # Sum pooling for array features.
                feat_emb_list.append(emb.sum(-2))
                feature_dict.pop(feat_id)
        
        for feat_id in self.dense_feature_ids:
            if feat_id in feature_dict:
                feat_emb_list.append(feature_dict[feat_id].unsqueeze(-1))
                feature_dict.pop(feat_id)
        
        user_features = torch.cat(feat_emb_list, dim=-1)
        return self.dnn(user_features)


class ItemTower(BaseTower):
    """Item Tower for processing item features."""
    def __init__(self):
        super().__init__('item_id', const.item_feature, const.model_param)
        
        self.mm_emb_feature_ids = const.item_feature.mm_emb_feature_ids
        # A config to skip certain mm features if needed.
        self.skip_mm_feats = getattr(const.model_param, 'skip_mm_feats', [])

        self.dnn = nn.Sequential(
            nn.Linear(self.get_item_feature_dim(), const.model_param.item_dnn_units),
            nn.ReLU(),
            nn.Linear(const.model_param.item_dnn_units, const.model_param.hidden_units),
        )
        self.mm_liner = self.build_mm_liner()
        
    def build_mm_liner(self):
        """Builds linear layers for multi-modal features."""
        mm_liner = nn.ModuleDict()
        for feat_id in self.mm_emb_feature_ids:
            if feat_id in self.skip_mm_feats:
                continue
            # This assumes mm_emb_dim and embedding_dim for these features are in const
            mm_liner[feat_id] = nn.Linear(const.mm_emb_dim[feat_id], self.model_param.embedding_dim[feat_id])
        return mm_liner
            
    def get_item_feature_dim(self):
        """Calculates the total dimension for item features."""
        num = self.model_param.embedding_dim['item_id']
        for feat_id in self.sparse_feature_ids:
            num += self.model_param.embedding_dim[feat_id]
        
        num += len(self.dense_feature_ids)
        for feat_id in self.mm_emb_feature_ids:
            if feat_id in self.skip_mm_feats:
                continue
            num += self.model_param.embedding_dim[feat_id]
        return num
        
    def forward(self, seq_id, item_mask, feature_dict):
        """Forward pass for ItemTower."""
        # Get embedding and then apply mask to zero out padded positions.
        id_embedding = self.sparse_emb['item_id'](seq_id) * item_mask.unsqueeze(-1)
        
        feat_emb_list = [id_embedding]
        
        for feat_id in self.sparse_feature_ids:
            if feat_id in feature_dict:
                feat_emb_list.append(self.sparse_emb[feat_id](feature_dict[feat_id]))
                feature_dict.pop(feat_id)
        
        for feat_id in self.dense_feature_ids:
            if feat_id in feature_dict:
                feat_emb_list.append(feature_dict[feat_id].unsqueeze(-1))
                feature_dict.pop(feat_id)
            
        for feat_id in self.mm_emb_feature_ids:
            if feat_id in self.skip_mm_feats:
                continue
            # Use dropout from model_param
            feat_emb_list.append(F.dropout(self.mm_liner[feat_id](feature_dict[feat_id]), p=self.model_param.dropout))
            feature_dict.pop(feat_id)
        
        item_features = torch.cat(feat_emb_list, dim=-1)
        return self.dnn(item_features)

class ContextTower(nn.Module):
    """Context Tower for processing contextual features, including user history."""
    def __init__(self, item_embedding):
        super().__init__()
        self.sparse_feature_ids = const.context_feature.sparse_feature_ids
        self.user_history_feature_id = const.context_feature.array_feature_ids[0] # Assumes '210' is the first and only one

        self.sparse_emb = self.setup_embedding_layer()
        self.dnn = nn.Sequential(
            nn.Linear(self.get_context_feature_dim(), const.model_param.context_dnn_units),
            nn.ReLU(),
            nn.Linear(const.model_param.context_dnn_units, const.model_param.hidden_units),
        )
        self.item_embedding = item_embedding
        
    def get_context_feature_dim(self):
        """Calculates the total dimension for context features."""
        dim = 0
        for feat_id in self.sparse_feature_ids:
            dim += const.model_param.embedding_dim[feat_id]
        
        # Dimension for the user history sequence embedding
        dim += const.model_param.embedding_dim['item_id']
        return dim
    
    def setup_embedding_layer(self):
        """Initializes embedding layers for all sparse context features."""
        emb_dict = nn.ModuleDict()
        for feat_id in self.sparse_feature_ids:
            emb_dict[feat_id] = nn.Embedding(const.model_param.embedding_table_size[feat_id] + 1, 
                                                    const.model_param.embedding_dim[feat_id], 
                                                    padding_idx=0)
        return emb_dict

    def forward(self, feature_dict):
        """Forward pass for ContextTower."""
        feat_emb_list = []
        
        for feat_id in self.sparse_feature_ids:
            if feat_id in feature_dict:
                feat_emb_list.append(self.sparse_emb[feat_id](feature_dict[feat_id]))
                feature_dict.pop(feat_id)

        # Process the user's historical clicked item sequence
        if self.user_history_feature_id in feature_dict:
            history_seq = feature_dict[self.user_history_feature_id]
            mask = (history_seq != 0).long()
            # Note: Simple averaging might lose sequential info. Consider Attention Pooling or GRU for improvement.
            user_seq_emb = torch.sum(self.item_embedding(history_seq), dim=-2)
            valid_mask = (mask.sum(-1) != 0)
            user_seq_emb = torch.where(valid_mask.unsqueeze(-1), 
                                       user_seq_emb / mask.sum(-1).unsqueeze(-1).clamp(min=1), 
                                       torch.zeros_like(user_seq_emb))
            feat_emb_list.append(user_seq_emb)
            feature_dict.pop(self.user_history_feature_id)
        
        context_features = torch.cat(feat_emb_list, dim=-1)
        return self.dnn(context_features)

class BaselineModel(nn.Module):
    """The main model orchestrating the towers and attention mechanism."""
    def __init__(self):
        super().__init__()
        self.item_tower = ItemTower()
        self.user_tower = UserTower()
        self.context_tower = ContextTower(self.item_tower.sparse_emb['item_id'])
        
        # Improved fusion layer for all features
        self.fusion_dnn = nn.Sequential(
            nn.Linear(const.model_param.hidden_units * 3, const.model_param.hidden_units),
            nn.ReLU(),
            nn.Linear(const.model_param.hidden_units, const.model_param.hidden_units)
        )
        
        self.pos_embedding = nn.Embedding(const.max_seq_len + 1, const.model_param.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(const.model_param.dropout)
        self.casual_attention_layers = AttentionDecoder(const.model_param.num_blocks, 
                                                        const.model_param.hidden_units,
                                                        const.model_param.num_heads,
                                                        const.model_param.dropout,
                                                        const.model_param.norm_first)
        
    def add_pos_embedding(self, seqs_id, emb):
        """Adds positional embedding to the sequence embedding."""
        valid_mask = (seqs_id != 0).long()
        poss = torch.cumsum(valid_mask, dim=1)
        poss = poss * valid_mask
        emb += self.pos_embedding(poss)
        emb = self.emb_dropout(emb)
        return emb
    
    def forward_item(self, seq_id, feature_dict, token_type=None):
        """Helper to forward item features."""
        if token_type is None:
            token_type = torch.ones_like(seq_id, dtype=torch.bool, device=seq_id.device)
        else:
            token_type = token_type == 1
        return self.item_tower(seq_id, token_type, feature_dict)
    
    def forward_all_feat(self, seq_id, token_type, feature_dict):
        """Forwards all features through their respective towers and fuses them."""
        item_feat = self.forward_item(seq_id, feature_dict, token_type)
        user_feat = self.user_tower(seq_id, token_type == 2, feature_dict)
        context_feat = self.context_tower(feature_dict)
        
        # Fuse all features by concatenation followed by a DNN
        all_feat = self.fusion_dnn(torch.cat([user_feat, item_feat, context_feat], dim=-1))
        return all_feat
            
    def forward(self, seqs_id, token_type, feat_dict):
        """The main forward pass of the model."""
        emb = self.forward_all_feat(seqs_id, token_type, feat_dict)
        feat = self.add_pos_embedding(seqs_id, emb)
        
        maxlen = seqs_id.shape[1]
        
        # Create attention mask for causal self-attention with padding
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=token_type.device)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (token_type != 0)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)
        
        log_feats = self.casual_attention_layers(feat, attention_mask)
        return log_feats