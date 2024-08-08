import os
import zipfile
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from dataloader.collate_fn import collate_fn
from module.Tokenizer import Tokenizer


class UserDataset(Dataset):
    def __init__(self, amazon_category):
        if os.path.exists(f'dataset/amazon_review_base/raw_review_{amazon_category}.csv.zip'):
            review_df = pd.read_csv(f'dataset/amazon_review_base/raw_review_{amazon_category}.csv.zip')
        else:
            review_df = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{amazon_category}", split="full", trust_remote_code=True).to_pandas()
            review_df.to_csv(f'dataset/amazon_review_base/raw_review_{amazon_category}.csv', index=False)
            with zipfile.ZipFile(f'dataset/amazon_review_base/raw_review_{amazon_category}.csv.zip', 'w', zipfile.ZIP_DEFLATED) as z:
                z.write(f'dataset/amazon_review_base/raw_review_{amazon_category}.csv')
            os.remove(f'dataset/amazon_review_base/raw_review_{amazon_category}.csv')
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
        self.max_history_length = 10
        self.tokenizer = Tokenizer()
        self.user_label_encoder = LabelEncoder()
        self.user_label_encoder.fit(self.dataframe.index)
        self.user_id_to_index = {user_id: i for i, user_id in enumerate(self.user_label_encoder.classes_)}
        self.index_to_user_id = {i: user_id for i, user_id in enumerate(self.user_label_encoder.classes_)}
        self.item_label_encoder = LabelEncoder()
        self.item_label_encoder.fit([item_id for item_ids in self.dataframe['reviewed_item_history'] for item_id in item_ids])
        self.item_id_to_index = {item_id: i for i, item_id in enumerate(self.item_label_encoder.classes_)}
        self.numerical_features = ['average_rating']
        self.categorical_features = []
        self.text_features = []
        self.history_features = ['purchased_item_ids']
        self.text_history_features = ['review_text_history']
        self.num_classes = {
            "id": len(self.dataframe),
            "item_id": len(self.item_label_encoder.classes_),
        }
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, id):
        row = self.dataframe.loc[id]
        encoded_id = self.user_id_to_index[id]
        purchased_item_ids_raws = [item_id for item_id, is_purchased in zip(row['reviewed_item_history'], row['is_purchased_history']) if is_purchased]
        purchased_item_ids = np.array([self.item_id_to_index[id] for id in purchased_item_ids_raws])
        average_rating = np.mean(row['rating_history'])
        return {
            'id': encoded_id, # user_id: integer
            'purchased_item_ids': purchased_item_ids, # history of purchased_item_ids: list[str]
            'review_text_history': row['review_text_history'], # history of review_text: | separated string
            'average_rating': average_rating, # average_rating: float
        }
    def collate_fn(self, batch):
        return collate_fn(
            batch, self.numerical_features, self.categorical_features, 
            self.text_features, self.history_features, self.text_history_features, 
            self.tokenizer, self.max_history_length)

        
if __name__ == '__main__':
    user_dataset = UserDataset('All_Beauty')
    print(len(user_dataset))
    batch = [user_dataset[i] for i in user_dataset.dataframe.index[0:32]]
    print(batch)
    print(user_dataset.collate_fn(batch))