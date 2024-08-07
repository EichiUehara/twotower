from datasets import load_dataset
from torch.utils.data import Dataset

from dataloader.collate_fn import collate_fn
from module.Tokenizer import Tokenizer
from sklearn.preprocessing import LabelEncoder

class ItemDataset(Dataset):
    def __init__(self, amazon_category):

        item_metadata = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{amazon_category}", split="full", trust_remote_code=True)
        item_df = item_metadata.to_pandas()
        item_df = item_df[['parent_asin', 'main_category', 'average_rating', 'rating_number', 'store', 'details']]
        item_df = item_df.drop_duplicates(subset='parent_asin')
        item_df = item_df.reset_index(drop=True)
        item_df = item_df.set_index('parent_asin')        
        self.dataframe = item_df

        self.max_history_length = 10
        self.item_label_encoder = item_label_encoder
        if hasattr(self.item_label_encoder, 'classes_') == False:
            self.item_label_encoder.fit(self.dataframe.index)
        self.item_id_to_index = {item_id: i for i, item_id in enumerate(self.item_label_encoder.classes_)}
        self.main_category_label_encoder = LabelEncoder()
        self.main_category_label_encoder.fit(self.dataframe['main_category'])
        self.main_category_id_to_index = {main_category_id: i for i, main_category_id in enumerate(self.main_category_label_encoder.classes_)}
        self.store_label_encoder = LabelEncoder()
        self.store_label_encoder.fit(self.dataframe['store'])
        self.store_id_to_index = {store_id: i for i, store_id in enumerate(self.store_label_encoder.classes_)}
        self.tokenizer = Tokenizer()
        self.numerical_features = ['average_rating', 'rating_number']
        self.categorical_features = ['main_category', 'store']
        self.text_features = ['details']
        # self.text_features = []
        self.history_features = []
        self.text_history_features = []
        self.input_dim = 200 + \
                         len(self.numerical_features) + \
                         len(self.categorical_features) * 50 + \
                         len(self.text_features)* 768 + \
                         len(self.history_features)* 50 + \
                         len(self.text_history_features)* 768

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, id):
        row = self.dataframe.loc[id]
        encoded_id = self.item_id_to_index[id]
        main_category = self.main_category_id_to_index[row['main_category']]
        store = self.store_id_to_index[row['store']]
        return {
            'id': encoded_id, # item_id: str
            'main_category': main_category, # category: str
            'average_rating': row['average_rating'], # rating: float
            'rating_number': row['rating_number'], # rating_number: int
            'store': store, # store: str
            'details': row['details'] # details: text
        }

    def collate_fn(self, batch):
        collate_fn(batch,
                     self.numerical_features,
                     self.categorical_features,
                     self.text_features,
                     self.history_features,
                     self.text_history_features,
                     self.tokenizer,
                     self.input_dim)

if __name__ == '__main__':
    item_label_encoder = LabelEncoder()
    tokenizer = Tokenizer()
    item_dataset = ItemDataset('All_Beauty', tokenizer, item_label_encoder=item_label_encoder)
    item_dataset.item_label_encoder.fit(item_dataset.dataframe.index)
    print(len(item_dataset))