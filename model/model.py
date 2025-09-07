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

        
    def forward(self, seq_id, feature_dict):
        id_embedding = self.sparse_emb['user_id'](seq_id)
        
        feat_emb_list = [id_embedding, ]
        
        
        for feat_id in const.user_feature.sparse_feature_ids:
            feat_emb_list.append(self.sparse_emb[feat_id](feature_dict[feat_id]))
            # feature_dict.pop(feat_id)
        
        for feat_id in const.user_feature.array_feature_ids:
            emb = self.sparse_emb[feat_id](feature_dict[feat_id])
            # 对数组特征进行求和池化，注意不能除以mask.sum(-1, keepdim=True)，因为mask可能为0
            feat_emb_list.append(emb.sum(-2))
            # feature_dict.pop(feat_id)
        
        for feat_id in const.user_feature.dense_feature_ids:
            feat_emb_list.append(feature_dict[feat_id].unsqueeze(-1))
            # feature_dict.pop(feat_id)
        
        
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
        num = const.model_param.embedding_dim['item_id']
        print(f"item_id : {num}",end=", ")
        for feat_id in const.item_feature.sparse_feature_ids:
            num += const.model_param.embedding_dim[feat_id]
            print(f"{feat_id} : {num}",end=", ")
        num += len(const.item_feature.dense_feature_ids)
        for feat_id in const.item_feature.mm_emb_feature_ids:
            num += const.model_param.embedding_dim[feat_id]
            print(f"{feat_id} : {num}",end=", ")
        print(f"dense : {num}")
        return num
    
    def setup_embedding_layer(self):
        emb_dict = nn.ModuleDict()
        emb_dict['item_id'] = nn.Embedding(const.model_param.embedding_table_size['item_id'] + 1, 
                                                    const.model_param.embedding_dim['item_id'], 
                                                    padding_idx=0)
        
        for feat_id in const.item_feature.sparse_feature_ids:
            emb_dict[feat_id] = nn.Embedding(const.model_param.embedding_table_size[feat_id] + 1, 
                                                    const.model_param.embedding_dim[feat_id], 
                                                    padding_idx=0 )
        return emb_dict
        
    def forward(self, seq_id, feature_dict):
        id_embedding = self.sparse_emb['item_id'](seq_id)
        
        feat_emb_list = [id_embedding, ]
        
        for feat_id in const.item_feature.sparse_feature_ids:
            feat_emb_list.append(self.sparse_emb[feat_id](feature_dict[feat_id]))
            # feature_dict.pop(feat_id)
        
        for feat_id in const.item_feature.dense_feature_ids:
            feat_emb_list.append(feature_dict[feat_id].unsqueeze(-1))
            # feature_dict.pop(feat_id)
            
        for feat_id in const.item_feature.mm_emb_feature_ids:
            feat_emb_list.append(F.dropout(self.mm_liner[feat_id](feature_dict[feat_id]), p=0.4))
            # feature_dict.pop(feat_id)
        item_features = torch.cat(feat_emb_list, dim=-1)
        return self.dnn(item_features)

class ContextTower(nn.Module):
    def __init__(self, item_feat_embedding):
        super().__init__()
        self.sparse_emb = self.setup_embedding_layer()
        self.dnn = nn.Sequential(
            nn.Linear(self.get_context_feature_dim(), const.model_param.context_dnn_units),
            nn.ReLU(),
            nn.Linear(const.model_param.context_dnn_units, const.model_param.hidden_units),
        )
        self.item_feat_embedding = item_feat_embedding
        
    def get_context_feature_dim(self):
        dim = 0
        for feat_id in const.context_feature.sparse_feature_ids:
            dim += const.model_param.embedding_dim[feat_id]
        return dim + const.model_param.embedding_dim['item_id'] + const.model_param.embedding_dim['101']
    

    def setup_embedding_layer(self):
        emb_dict = nn.ModuleDict()
        for feat_id in const.context_feature.sparse_feature_ids:
            emb_dict[feat_id] = nn.Embedding(const.model_param.embedding_table_size[feat_id] + 1, 
                                                    const.model_param.embedding_dim[feat_id], 
                                                    padding_idx=0)
        return emb_dict

    def forward(self, feature_dict):
        feat_emb_list = []
        for feat_id in const.context_feature.sparse_feature_ids:
            feat_emb_list.append(self.sparse_emb[feat_id](feature_dict[feat_id]))
            # feature_dict.pop(feat_id)

        last_click_item_id = feature_dict['210']
        user_seq_emb = self.item_feat_embedding['item_id'](last_click_item_id)
        feat_emb_list.append(user_seq_emb)
        
        last_click_item_101 = feature_dict['401']
        last_click_item_101_emb = self.item_feat_embedding['101'](last_click_item_101)
        feat_emb_list.append(last_click_item_101_emb)
        # feature_dict.pop('210')
        context_features = torch.cat(feat_emb_list, dim=-1)
        return self.dnn(context_features)

class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.item_tower = ItemTower()
        self.user_tower = UserTower()
        self.context_tower = ContextTower(self.item_tower.sparse_emb)
        self.merge_dnn = nn.Sequential(
            nn.Linear(const.model_param.hidden_units, const.model_param.hidden_units),
            # nn.LayerNorm(const.model_param.hidden_units),
            # nn.Dropout(const.model_param.dropout),
        )
        self.context_dnn = nn.Linear(const.model_param.hidden_units * 3, const.model_param.hidden_units)
        
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
    
    
    
    def forward_all_feat(self, user_id, user_feat,input_ids, input_feat, context_feat):
        item_feat = self.item_tower(input_ids, input_feat)
        user_feat = self.user_tower(user_id, user_feat)
        context_feat = self.context_tower(context_feat)
        seq_feat = torch.cat([item_feat, user_feat[:,None].repeat(1,item_feat.shape[1],1),context_feat], dim=-1)
        seq_feat = self.context_dnn(seq_feat)
        return self.merge_dnn(seq_feat)
    
        
    def forward(self, user_id, user_feat, input_ids, input_feat, context_feat):
        emb = self.forward_all_feat(user_id, user_feat,input_ids, input_feat, context_feat)
        feat = self.emb_dropout(emb)
        
        maxlen = input_ids.shape[1]
        
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=emb.device)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (input_ids != 0)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)
        
        log_feats = self.casual_attention_layers(feat, attention_mask)
        return log_feats
