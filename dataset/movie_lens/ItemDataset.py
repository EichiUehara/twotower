from datasets import load_dataset
import torch
from torch.utils.data import Dataset

from module.Tokenizer import Tokenizer
from module.process_history_feature import process_history_feature
from module.process_text_feature import process_text_feature
from sklearn.preprocessing import LabelEncoder

from module.process_text_history_feature import process_text_history_feature

class ItemDataset(Dataset):
    def __init__(self, amazon_category, tokenizer: Tokenizer,
                 item_label_encoder: LabelEncoder,
                 max_history_length=10):
        item_metadata = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{amazon_category}", split="full", trust_remote_code=True)
        item_df = item_metadata.to_pandas()
        item_df = item_df[['parent_asin', 'main_category', 'average_rating', 'rating_number', 'store', 'details']]
        item_df = item_df.drop_duplicates(subset='parent_asin')
        item_df = item_df.reset_index(drop=True)
        item_df = item_df.set_index('parent_asin')        
        self.dataframe = item_df

        self.max_history_length = max_history_length
        self.item_id_label_encoder = item_label_encoder
        self.main_category_label_encoder = LabelEncoder()
        self.main_category_label_encoder.fit(self.dataframe['main_category'])
        self.store_label_encoder = LabelEncoder()
        self.store_label_encoder.fit(self.dataframe['store'])
        self.tokenizer = tokenizer
        self.numerical_features = ['average_rating', 'rating_number']
        self.categorical_features = ['main_category', 'store']
        self.text_features = ['details']
        self.history_features = []
        self.text_history_features = []
        self.input_dim = len(self.numerical_features) + \
                         len(self.categorical_features) + \
                         len(self.text_features)* 768 + \
                         len(self.history_features)* 10 + \
                         len(self.text_history_features)* 768

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        parent_asin = self.item_id_label_encoder.transform([row.name])[0]
        main_category = self.main_category_label_encoder.transform([row['main_category']])[0]
        store = self.store_label_encoder.transform([row['store']])[0]
        details = row['details']
        return {
            'id': parent_asin, # item_id: str
            'main_category': main_category, # category: str
            'average_rating': row['average_rating'], # rating: float
            'rating_number': row['rating_number'], # rating_number: int
            'store': store, # store: str
            'details': details # details: text
        }
    def get_item_by_encoded_id(self, encoded_id):
        return self.dataframe.loc[self.item_id_label_encoder.inverse_transform([encoded_id])[0]]
    def get_item_by_encoded_ids(self, encoded_ids):
        return self.dataframe.loc[self.user_label_encoder.inverse_transform(encoded_ids)]

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
            k: torch.stack([process_history_feature(item[k], self.max_history_length) for item in batch])
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
    item_label_encoder = LabelEncoder()
    tokenizer = Tokenizer()
    item_dataset = ItemDataset('All_Beauty', tokenizer, item_label_encoder=item_label_encoder)
    item_dataset.item_id_label_encoder.fit(item_dataset.dataframe.index)
    print(item_dataset[0])
    print(item_dataset.get_item_by_encoded_id(item_dataset[0]['id']))
    batch = [item_dataset[0], item_dataset[1]]
    print(item_dataset.collate_fn(batch))