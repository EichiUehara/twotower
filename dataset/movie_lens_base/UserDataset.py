import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from dataloader.collate_fn import collate_fn
from module.Tokenizer import Tokenizer

class UserDataset(Dataset):
    def __init__(self):
        review_df = pd.read_csv("dataset/movie_lens_base/movielens.zip")
        def most_frequent(series):
            return series.mode()[0] if not series.mode().empty else None
        user_df = review_df.groupby('user_id').agg({ 
                    'gender': most_frequent,
                    'age': most_frequent,
                    'occupation': most_frequent,
                    'zipcode': most_frequent,
                    'rating': 'mean',
                })
        self.gender_encoder = LabelEncoder()
        gender_encoded = self.gender_encoder.fit_transform(user_df[['gender']].values.ravel())
        self.zipcode_encoder = LabelEncoder()
        zipcode_encoded = self.zipcode_encoder.fit_transform(user_df[['zipcode']].values.ravel())
        processed_user_df = user_df.copy()
        processed_user_df['gender'] = gender_encoded
        processed_user_df['zipcode'] = zipcode_encoded
        self.numerical_features = ['age', 'average_rating']
        self.categorical_features = ['occupation', 'zipcode', 'gender']
        self.text_features = []
        self.history_features = []
        self.text_history_features = []
        self.tokenizer = Tokenizer()
        self.max_history_length = 10
        self.dataframe = processed_user_df
        self.index_to_id = {i: id for i, id in enumerate(self.dataframe.index)}
        self.id_to_index = {id: i for i, id in enumerate(self.dataframe.index)}
        self.num_classes = {
            "id": len(self.dataframe),
            "gender": len(gender_encoded),
            "zipcode": len(zipcode_encoded),
            "occupation": len(user_df['occupation'].unique())
        }

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, id):
        if hasattr(id, 'item'):
            id = id.item()
        row = self.dataframe.loc[id]
        return {
            'id': self.id_to_index[id], # user_id: integer
            'gender': row['gender'].astype(int),  # int
            'age': row['age'].astype(int), # age: int
            'average_rating': row['rating'], # rating: float
            'occupation': row['occupation'].astype(int), # occupation: str
            'zipcode': row['zipcode'].astype(int), # zipcode: str
        }
    def collate_fn(self, batch):
        return collate_fn(batch, 
            self.numerical_features, 
            self.categorical_features, 
            self.text_features, 
            self.history_features, 
            self.text_history_features,
            self.tokenizer,
            self.max_history_length)

        
if __name__ == '__main__':
    user_dataset = UserDataset()
    print(len(user_dataset))
    print(user_dataset[user_dataset.index_to_user_id[0]])
    for i in range(10):
        print(user_dataset[user_dataset.index_to_user_id[i]])