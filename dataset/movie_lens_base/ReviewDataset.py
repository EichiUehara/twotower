import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class ReviewDataset(Dataset):
    def __init__(self):
        review_df = pd.read_csv("dataset/movie_lens_base/movielens.zip")
        review_df = review_df[['user_id', 'movie_id', 'rating']]
        # if rating >= 4, rating = 1, else rating = 0
        review_df['rating'] = review_df['rating'].apply(lambda x: 1 if x >= 4 else 0)
        self.dataframe = review_df

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]        
        return {
            'user_id': row['user_id'], # user_id: str
            'item_id': row['movie_id'], # item_id: str
            'rating': row['rating'] # rating: bool
        }        
if __name__ == '__main__':
    review_dataset = ReviewDataset()
    for i in range(5):
        print(review_dataset[i])