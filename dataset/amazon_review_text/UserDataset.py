import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from module.Tokenizer import Tokenizer
from module.process_history_feature import process_history_feature
from module.process_text_feature import process_text_feature
from module.process_text_history_feature import process_text_history_feature


class UserDataset(Dataset):
    def __init__(self, amazon_category, tokenizer: Tokenizer,
                 item_label_encoder: LabelEncoder, user_label_encoder: LabelEncoder,
                 max_history_length=10):
        reviews = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{amazon_category}", split="full", trust_remote_code=True)
        review_df = reviews.to_pandas()
        review_df = review_df[['timestamp', 'user_id', 'verified_purchase', 'rating', 'parent_asin', 'text']]
        review_df = review_df.sort_values(by=['user_id', 'timestamp'], ascending=[True, False])
        user_df = review_df.groupby('user_id').agg({
            'verified_purchase': lambda x: list(x), 
            'parent_asin': lambda x: list(x),
            'rating': lambda x: list(x),
            'text': lambda x: list(x),
        })
        user_df.rename(columns={
            'verified_purchase': 'is_purchased_history',
            'parent_asin': 'reviewed_item_history',
            'rating': 'rating_history',
            'text': 'review_text_history',
            },inplace=True)

        self.dataframe = user_df
        self.max_history_length = max_history_length
        self.tokenizer = tokenizer
        self.user_label_encoder = user_label_encoder
        self.item_label_encoder = item_label_encoder
        self.numerical_features = ['average_rating']
        self.categorical_features = []
        self.text_features = []
        self.history_features = ['purchased_item_ids']
        self.text_history_features = ['review_text_history']
        self.input_dim = 200 + \
                         len(self.numerical_features) + \
                         len(self.categorical_features) * 20 + \
                         len(self.text_features)* 768 + \
                         len(self.history_features)* 50 + \
                         len(self.text_history_features)* 768

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, encoded_id):
        self.dataframe.loc['AED2GFGIAJ22PHMZGSKH2CPUF75Q']
        row = self.dataframe.loc[self.user_label_encoder.inverse_transform([encoded_id])[0]]
        user_id = self.user_label_encoder.transform([row.name])[0]
        purchased_item_ids = [self.item_label_encoder.transform([item_id])[0] for item_id, is_purchased in zip(row['reviewed_item_history'], row['is_purchased_history']) if is_purchased]
        average_rating = np.mean(row['rating_history'])
        review_text_history = row['review_text_history']
        return {
            'id': user_id, # user_id: integer
            'purchased_item_ids': purchased_item_ids, # history of purchased_item_ids: list[str]
            'review_text_history': review_text_history, # history of review_text: | separated string
            'average_rating': average_rating, # average_rating: float
        }
    def collate_fn(self, batch):
        # Process numerical features
        numerical_features_tensor = torch.stack([torch.tensor([item[k] for item in batch], dtype=torch.float32) for k in self.numerical_features])
        
        # Process categorical features
        categorical_features = {k: torch.tensor([item[k] for item in batch], dtype=torch.long) for k in self.categorical_features}
        
        # Process text features
        text_features = {
            k: process_text_feature([item[k] for item in batch], self.tokenizer, padding=True, truncation=True, max_length=512, return_tensors='pt')
            for k in self.text_features
        }
        
        # Process history features
        history_features = {
            k: torch.stack([process_history_feature(torch.tensor(item[k]), self.max_history_length) for item in batch]).squeeze(1)
            for k in self.history_features
        }
        
        # Process text history features
        text_history_features = {
            k: process_text_history_feature(
                [item[k] for item in batch], 
                self.max_history_length, 
                self.tokenizer, padding='max_length', 
                truncation=True, max_length=512, 
                return_tensors='pt')
            for k in self.text_history_features
        }
        
        # Return the batch in a format compatible with the model
        return {
            'numerical_features': numerical_features_tensor,
            'categorical_features': categorical_features,
            'text_features': text_features,
            'history_features': history_features,
            'text_history_features': text_history_features
        }

        
if __name__ == '__main__':
    from dataset.amazon_review.ItemDataset import ItemDataset
    item_label_encoder = LabelEncoder()
    user_label_encoder = LabelEncoder()
    tokenizer = Tokenizer()
    user_dataset = UserDataset(
        'All_Beauty', tokenizer,
        item_label_encoder=item_label_encoder, user_label_encoder=user_label_encoder,
        max_history_length=10)
    user_dataset.user_label_encoder.fit(user_dataset.dataframe.index)
    item_dataset = ItemDataset('All_Beauty', tokenizer, item_label_encoder=item_label_encoder)
    item_dataset.item_label_encoder.fit(item_dataset.dataframe.index)
    batch = [user_dataset[id] for id in [44636, 489025, 584726, 60237, 513537, 313297, 453297, 230375, 519357, 558609, 321068, 267866, 413788, 315984, 432122, 248620, 349842, 337030, 204676, 626178, 117422, 191508, 622734, 394569, 323315, 50361, 94118, 378209, 205582, 623248, 86303, 67980]]
    # batch = next(iter(user_dataset))
    # print(batch)
    # print(user_dataset.collate_fn([batch, batch]))
    # print(user_dataset.collate_fn([batch, batch])['numerical_features'].shape)
    # print(user_dataset.collate_fn([batch, batch])['numerical_features'])