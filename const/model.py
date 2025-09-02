from utils import read_pickle
class ModelParam:
    def __init__(self,indexer_file):
        self.embedding_table_size = self.read_feature_size(indexer_file)
        self.max_decay = 20
        self.max_diff = 20
        self.embedding_table_size["201"] = self.max_diff
        self.embedding_table_size["202"] = 8
        self.embedding_table_size["203"] = 25
        self.embedding_table_size["204"] = 13
        self.embedding_table_size["205"] = 32
        self.embedding_table_size["206"] = self.max_decay
        self.embedding_table_size["301"] = 3 # PAD=0, FALSE=1, TRUE=2
        self.embedding_table_size["302"] = 3 # PAD=0, FALSE=1, TRUE=2
        self.embedding_table_size["303"] = self.embedding_table_size["101"] # Vocab size is the same as feature 101

        # Semantic features from RQ-VAE codebook sizes
        self.embedding_table_size["130"] = 256
        self.embedding_table_size["131"] = 256
        self.embedding_table_size["132"] = 256
        self.embedding_table_size["133"] = 128
        self.embedding_table_size["134"] = 128
        self.embedding_table_size["135"] = 64
        self.embedding_table_size["136"] = 64
        self.embedding_table_size["137"] = 32
        
        self.embedding_dim = {
            "user_id":64,
            "item_id":96,
            # 物品特征
            "100": 16,      # 6
            "101": 16,     # 51
            "102": 32,     # 90709
            # "111": 128,    # 4783154
            "112": 16,     # 30
            "114": 16,     # 20
            "115": 16,     # 691
            "116": 16,     # 18
            "117": 16,     # 497
            "118": 16,     # 1426
            "119": 24,     # 4191
            "120": 24,     # 3392
            # "121": 64,    # 2135891
            "122": 32,     # 90919

            # Semantic features from RQ-VAE with hierarchical dimensions
            "130": 16,     # Codebook size 256
            "131": 16,     # Codebook size 256
            "132": 16,     # Codebook size 256
            "133": 8,      # Codebook size 128
            "134": 8,      # Codebook size 128
            "135": 8,      # Codebook size 64
            "136": 8,      # Codebook size 64
            "137": 4,      # Codebook size 32

            # 多模态特征
            "81": 64,
            "82": 128,
            "83": 128,
            "84": 128,
            "85": 128,
            "86": 128,
            # 用户特征
            "103": 16,     # 86
            "104": 16,      # 2
            "105": 16,      # 7
            "106": 16,     # 14
            "107": 16,     # 19
            "108": 16,      # 4
            "109": 16,      # 3
            "110": 16,       # 2
            
            "201": 32,# 时间特征
            "202": 16,# weekday
            "203": 16, # hour
            "204": 16, # month
            "205": 16, # day   
            "206": 16, # decay
            
            "301": 16, # is_repeated_101
            "302": 16, # is_repeated_102
            "303": 16, # prev_feature_101
        }
        self.user_dnn_units = 128
        self.item_dnn_units = 128
        self.context_dnn_units = 128
        self.dropout = 0.2
        self.hidden_units = 256
        self.num_blocks = 8
        self.num_heads = 8
        self.norm_first = True
    def read_feature_size(self, indexer_file):
        indexer = read_pickle(indexer_file)
        emb_table_size = {}
        emb_table_size['item_id'] = len(indexer['i'])
        emb_table_size['user_id'] = len(indexer['u'])

        for feat_id, mapping_dict in indexer['f'].items():
            emb_table_size[feat_id] = len(mapping_dict)
            
        return emb_table_size