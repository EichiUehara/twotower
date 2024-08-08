import re
from sklearn.calibration import LabelEncoder
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from dataloader.collate_fn import collate_fn
from module.Tokenizer import Tokenizer

class ItemDataset(Dataset):
    def __init__(self):
        review_df = pd.read_csv("dataset/movie_lens/movielens.zip")
        def most_frequent(series):
            return series.mode()[0] if not series.mode().empty else None
            
        review_df['year'] = review_df['title'].str.extract(r'\((\d{4})\)')
        review_df['title'] = review_df['title'].apply(lambda x: re.sub(r'\s*\(\d{4}\)\s*$', '', x))
        movie_df = review_df.groupby('movie_id').agg({ 
                    'title': most_frequent,
                    'genres': most_frequent,
                    'year': most_frequent,
                    'rating': 'mean',
                })
        self.genres_encoder = LabelEncoder()
        self.year_encoder = LabelEncoder()
        genres_encoded = self.genres_encoder.fit_transform(movie_df[['genres']])
        year_encoded = self.year_encoder.fit_transform(movie_df[['year']])
        movie_df['genres'] = genres_encoded
        movie_df['year'] = year_encoded

        self.numerical_features = ['average_rating']
        self.categorical_features = ['genres', 'year']
        self.text_features = ['title']
        self.history_features = []
        self.text_history_features = []
        self.input_dim = 200 + \
                         len(self.numerical_features) + \
                         len(self.categorical_features) * 50 + \
                         len(self.text_features)* 768 + \
                         len(self.history_features)* 50 + \
                         len(self.text_history_features)* 768
        self.tokenizer = Tokenizer()
        self.max_history_length = 10
        self.dataframe = movie_df
        self.index_to_id = {i: id for i, id in enumerate(self.dataframe.index)}
        self.id_to_index = {id: i for i, id in enumerate(self.dataframe.index)}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, id):
        # handle if item_id is tensor
        if hasattr(id, 'item'):
            id = id.item()
        row = self.dataframe.loc[id]
        return {
            'id': self.id_to_index[id], # item_id: str
            'genres': row['genres'], # category: str
            'year': row['year'], # year: str
            'average_rating': row['rating'], # rating: float
            'title': row['title'] # store: str
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
    item_dataset = ItemDataset()
    print(len(item_dataset))
    for i in range(3):
        print(item_dataset[item_dataset.dataframe.index[i]])