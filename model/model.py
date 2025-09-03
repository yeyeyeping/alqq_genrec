import numpy as np
import torch
import torch.nn.functional as F
import const
from torch import nn
from .atten import AttentionDecoder

class UserTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.sparse_emb = self.setup_embedding_layer()

        self.dnn = nn.Sequential(
            nn.Linear(self.get_user_feature_dim(), const.model_param.user_dnn_units),
            nn.ReLU(),
            # nn.LayerNorm(const.model_param.user_dnn_units),
            # nn.Dropout(const.model_param.dropout),
            nn.Linear(const.model_param.user_dnn_units, const.model_param.hidden_units),
        )
        
    def get_user_feature_dim(self):
        print("user feature dim: ")
        num = const.model_param.embedding_dim['user_id']
        print(f"user_id : {num}",end=", ")
        for feat_id in const.user_feature.sparse_feature_ids + const.user_feature.array_feature_ids:
            num += const.model_param.embedding_dim[feat_id]
            print(f"{feat_id} : {num}",end=", ")
        num += len(const.user_feature.dense_feature_ids)
        print(f"dense : {num}")
        return num
    
    def setup_embedding_layer(self):
        emb_dict = nn.ModuleDict()
        emb_dict['user_id'] = nn.Embedding(const.model_param.embedding_table_size['user_id'] + 1, 
                                                    const.model_param.embedding_dim['user_id'], padding_idx=0)
        
        for feat_id in const.user_feature.sparse_feature_ids + const.user_feature.array_feature_ids:
            emb_dict[feat_id] = nn.Embedding(const.model_param.embedding_table_size[feat_id] + 1, 
                                                    const.model_param.embedding_dim[feat_id], padding_idx=0)
        return emb_dict

        
    def forward(self, seq_id, user_mask, feature_dict):
        id_embedding = self.sparse_emb['user_id'](seq_id * user_mask)
        
        feat_emb_list = [id_embedding, ]
        
        
        for feat_id in const.user_feature.sparse_feature_ids:
            if feat_id in feature_dict:
                feat_emb_list.append(self.sparse_emb[feat_id](feature_dict[feat_id]))
                feature_dict.pop(feat_id)
        
        for feat_id in const.user_feature.array_feature_ids:
            if feat_id in feature_dict:
                emb = self.sparse_emb[feat_id](feature_dict[feat_id])
                # 对数组特征进行求和池化，注意不能除以mask.sum(-1, keepdim=True)，因为mask可能为0
                feat_emb_list.append(emb.sum(-2))
                feature_dict.pop(feat_id)
        
        for feat_id in const.user_feature.dense_feature_ids:
            if feat_id in feature_dict:
                feat_emb_list.append(feature_dict[feat_id].unsqueeze(-1))
                feature_dict.pop(feat_id)
        
        
        user_features = torch.cat(feat_emb_list, dim=-1)
        return self.dnn(user_features)
            

class ItemTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.sparse_emb = self.setup_embedding_layer()
        self.dnn = nn.Sequential(
            nn.Linear(self.get_item_feature_dim(), const.model_param.item_dnn_units),
            nn.ReLU(),
            # nn.LayerNorm(const.model_param.item_dnn_units),
            # nn.Dropout(const.model_param.dropout),
            nn.Linear(const.model_param.item_dnn_units, const.model_param.hidden_units),
        )
        self.mm_liner = self.build_mm_liner()
        
    def build_mm_liner(self,):
        mm_liner = nn.ModuleDict()
        for feat_id in const.item_feature.mm_emb_feature_ids:
            mm_liner[feat_id] = nn.Linear(const.mm_emb_dim[feat_id], const.model_param.embedding_dim[feat_id])
        return mm_liner
            
    def get_item_feature_dim(self):
        print("item feature dim: ")
        # Start with the main item_id embedding dimension
        num = const.model_param.embedding_dim['item_id']
        print(f"item_id : {num}",end=", ")

        # Add dimensions for all auxiliary sparse features
        for feat_id in const.item_feature.sparse_feature_ids:
            num += const.model_param.embedding_dim[feat_id]
            print(f"{feat_id} : {num}",end=", ")
        
        # Add dimensions for dense and mm features
        num += len(const.item_feature.dense_feature_ids)
        for feat_id in const.item_feature.mm_emb_feature_ids:
            num += const.model_param.embedding_dim[feat_id]
            print(f"{feat_id} : {num}",end=", ")
        print(f"dense : {num}")
        return num
    
    def setup_embedding_layer(self):
        emb_dict = nn.ModuleDict()
        # Create the main item_id embedding layer
        emb_dict['item_id'] = nn.Embedding(const.model_param.embedding_table_size['item_id'] + 1, 
                                                    const.model_param.embedding_dim['item_id'], 
                                                    padding_idx=0)
        
        # Create embedding layers for all auxiliary sparse features
        for feat_id in const.item_feature.sparse_feature_ids:
            emb_dict[feat_id] = nn.Embedding(const.model_param.embedding_table_size[feat_id] + 1, 
                                                    const.model_param.embedding_dim[feat_id], 
                                                    padding_idx=0 )
        return emb_dict
        
    def forward(self, seq_id, item_mask, feature_dict):
        # 1. Get embedding for the main item ID from seq_id
        id_embedding = self.sparse_emb['item_id'](seq_id * item_mask)
        
        feat_emb_list = [id_embedding, ]
        
        # 2. Get embeddings for all auxiliary sparse features from feature_dict
        for feat_id in const.item_feature.sparse_feature_ids:
            if feat_id in feature_dict:
                feat_emb_list.append(self.sparse_emb[feat_id](feature_dict[feat_id]))
                feature_dict.pop(feat_id)
        
        # 3. Process dense features
        for feat_id in const.item_feature.dense_feature_ids:
            if feat_id in feature_dict:
                feat_emb_list.append(feature_dict[feat_id].unsqueeze(-1))
                feature_dict.pop(feat_id)
            
        # 4. Process multi-modal features
        for feat_id in const.item_feature.mm_emb_feature_ids:
            if feat_id in feature_dict:
                feat_emb_list.append(F.dropout(self.mm_liner[feat_id](feature_dict[feat_id]), p=0.4))
                feature_dict.pop(feat_id)
        
        item_features = torch.cat(feat_emb_list, dim=-1)
        return self.dnn(item_features)

class ContextTower(nn.Module):
    def __init__(self, item_embedding):
        super().__init__()
        self.sparse_emb = self.setup_embedding_layer()
        self.dnn = nn.Sequential(
            nn.Linear(self.get_context_feature_dim(), const.model_param.context_dnn_units),
            nn.ReLU(),
            nn.Linear(const.model_param.context_dnn_units, const.model_param.hidden_units),
        )
        self.item_embedding = item_embedding
        
    def get_context_feature_dim(self):
        dim = 0
        # Add dimensions for all sparse context features defined in const/feature.py
        for feat_id in const.context_feature.sparse_feature_ids:
            dim += const.model_param.embedding_dim[feat_id]
        
        # Add dimension for the sequence embedding from feature '210'
        dim += const.model_param.embedding_dim['item_id']
        return dim
    
    def setup_embedding_layer(self):
        emb_dict = nn.ModuleDict()
        # Create embedding layers for all sparse context features
        for feat_id in const.context_feature.sparse_feature_ids:
            emb_dict[feat_id] = nn.Embedding(const.model_param.embedding_table_size[feat_id] + 1, 
                                                    const.model_param.embedding_dim[feat_id], 
                                                    padding_idx=0)
        return emb_dict

    def forward(self, feature_dict):
        feat_emb_list = []
        
        # Process all sparse context features
        for feat_id in const.context_feature.sparse_feature_ids:
            if feat_id in feature_dict:
                feat_emb_list.append(self.sparse_emb[feat_id](feature_dict[feat_id]))
                feature_dict.pop(feat_id)

        # Process the user's historical clicked item sequence feature '210'
        if '210' in feature_dict:
            mask = (feature_dict['210'] != 0).long()
            user_seq_emb = torch.sum(self.item_embedding(feature_dict['210']), dim=-2)
            valid_mask = (mask.sum(-1) != 0)
            user_seq_emb = torch.where(valid_mask.unsqueeze(-1), 
                                       user_seq_emb / mask.sum(-1).unsqueeze(-1).clamp(min=1), 
                                       torch.zeros_like(user_seq_emb))
            feat_emb_list.append(user_seq_emb)
            feature_dict.pop('210')
        
        context_features = torch.cat(feat_emb_list, dim=-1)
        return self.dnn(context_features)

class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.item_tower = ItemTower()
        self.user_tower = UserTower()
        self.context_tower = ContextTower(self.item_tower.sparse_emb['item_id'])
        self.merge_dnn = nn.Sequential(
            nn.Linear(const.model_param.hidden_units, const.model_param.hidden_units),
            # nn.LayerNorm(const.model_param.hidden_units),
            # nn.Dropout(const.model_param.dropout),
        )
        self.context_dnn = nn.Linear(const.model_param.hidden_units * 2, const.model_param.hidden_units)
        
        self.pos_embedding = nn.Embedding(const.max_seq_len + 1, const.model_param.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(const.model_param.dropout)
        self.casual_attention_layers = AttentionDecoder(const.model_param.num_blocks, 
                                                        const.model_param.hidden_units,
                                                        const.model_param.num_heads,
                                                        const.model_param.dropout,
                                                        const.model_param.norm_first)
        
    def add_pos_embedding(self, seqs_id, emb, ):
        # emb *= const.model_param.hidden_units ** 0.5
        # valid_mask = (seqs_id != 0).long()
        # poss = torch.cumsum(valid_mask, dim=1)
        # poss = poss * valid_mask
        # emb += self.pos_embedding(poss)
        emb = self.emb_dropout(emb)
        return emb
    
        
    def forward_item(self, seq_id, feature_dict, token_type=None):
        if token_type is None:
            token_type = torch.ones_like(seq_id, dtype=torch.bool, device=seq_id.device)
        else:
            token_type = token_type == 1
        return self.item_tower(seq_id, token_type, feature_dict)
    
    def forward_all_feat(self, seq_id, token_type, feature_dict):
        item_feat = self.forward_item(seq_id, feature_dict, token_type)
        user_feat = self.user_tower(seq_id, token_type == 2, feature_dict)
        context_feat = self.context_tower(feature_dict)
        seq_feat = torch.cat([item_feat, context_feat], dim=-1)
        seq_feat = self.context_dnn(seq_feat)
        
        all_feat = user_feat + seq_feat
        return self.merge_dnn(all_feat)
    
        
    def forward(self, seqs_id, token_type, feat_dict):
        emb = self.forward_all_feat(seqs_id, token_type, feat_dict)
        feat = self.add_pos_embedding(seqs_id, emb)
        
        maxlen = seqs_id.shape[1]
        
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=token_type.device)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (token_type != 0)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)
        
        log_feats = self.casual_attention_layers(feat, attention_mask)
        return log_feats
