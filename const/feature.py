def pad_array(feat_value, max_len):
    pad_len = max_len - len(feat_value)
    return feat_value + [0] * pad_len
class UserFeature:
    def __init__(self):
        self.sparse_feature_ids = ('103', '104', '105', '109')
        self.dense_feature_ids = []
        self.array_feature_ids = ('106', '107', '108', '110')
        self.all_feature_ids = sorted(list(self.sparse_feature_ids) + list(self.dense_feature_ids) + list(self.array_feature_ids))

    
    
    def pad_array_feature(self, feat_id, feat_value):
        if feat_id == '106':
            return pad_array(feat_value, 8)
        elif feat_id == '107':
            return pad_array(feat_value, 1)
        elif feat_id == '108':
            return pad_array(feat_value, 2)
        elif feat_id == '110':
            return pad_array(feat_value, 2)
        
    def fill(self, feat_id):
        if feat_id in self.sparse_feature_ids:
            return 0
        
        elif feat_id in self.dense_feature_ids:
            return 0.0
        
        elif feat_id in self.array_feature_ids:
            return self.pad_array_feature(feat_id, [0, ])
        else:
            raise ValueError(f"Invalid feature id: {feat_id}")
        
class  ItemFeature:
    def __init__(self):
        self.sparse_feature_ids = (
            '100',
            '101',
            '102',
            '111',
            '112',
            '114',
            '115',
            '116',
            '117',
            '118',
            '119',
            '120',
            '121',
            '122',
        )
        self.dense_feature_ids = ()
        self.mm_emb_feature_ids = ("81", )
        self.all_feature_ids = sorted(list(self.sparse_feature_ids) + list(self.dense_feature_ids))
    
    def fill(self, feat_id):
        if feat_id in self.sparse_feature_ids:
            return 0
        elif feat_id in self.dense_feature_ids:
            return 0.0
        else:
            raise ValueError(f"Invalid feature id: {feat_id}")

        
class ContextFeature:
    def __init__(self):
        self.sparse_feature_ids = (
            "201",
            "202",
            "203",
            "204",
            "205",
            "206",
        )
        self.seq_len = 10
        self.array_feature_ids = ("210", )
        
        self.all_feature_ids = sorted(list(self.sparse_feature_ids) + list(self.array_feature_ids))
    
    def fill(self, feat_id):
        if feat_id in self.sparse_feature_ids:
            return 0
        elif feat_id in self.array_feature_ids:
            return pad_array([0, ] ,self.seq_len)
        else:
            raise ValueError(f"Invalid feature id: {feat_id}")
        
user_feature = UserFeature()
item_feature = ItemFeature()
context_feature = ContextFeature()
print(f"total user feature: {len(user_feature.all_feature_ids)}, ids: {user_feature.all_feature_ids}")
print(f"total item feature: {len(item_feature.all_feature_ids)}, ids: {item_feature.all_feature_ids}")

print(f"total feature: {len(user_feature.all_feature_ids) + len(item_feature.all_feature_ids)}, ids: {user_feature.all_feature_ids + item_feature.all_feature_ids}")
