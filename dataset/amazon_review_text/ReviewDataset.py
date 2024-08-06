from torch.utils.data import Dataset
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder

class ReviewDataset(Dataset):
    def __init__(self, amazon_category, 
                 item_label_encoder: LabelEncoder, user_label_encoder: LabelEncoder):
        reviews = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{amazon_category}", split="full", trust_remote_code=True)
        review_df = reviews.to_pandas()
        review_df = review_df[['timestamp', 'user_id', 'parent_asin', 'rating']]
        review_df = review_df.sort_values(by=['timestamp'], ascending=[True])
        review_df = review_df.drop_duplicates(subset=['user_id', 'parent_asin'], keep='first')
        review_df['rating'] = review_df['rating'].apply(lambda x: 1 if x >= 4 else 0)
        self.user_label_encoder = user_label_encoder
        self.item_label_encoder = item_label_encoder
        self.dataframe = review_df

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        return {
            'user_id': self.user_label_encoder.transform([row['user_id']])[0], # user_id: str
            'item_id': self.item_label_encoder.transform([row['parent_asin']])[0], # item_id: str
            'rating': row['rating'] # rating: bool
        }
        
if __name__ == '__main__':
    item_label_encoder = LabelEncoder()
    user_label_encoder = LabelEncoder()
    review_dataset = ReviewDataset(
        'All_Beauty', 
        item_label_encoder=item_label_encoder, user_label_encoder=user_label_encoder)
    review_dataset.item_label_encoder.fit(review_dataset.dataframe['parent_asin'].unique())
    review_dataset.user_label_encoder.fit(review_dataset.dataframe['user_id'].unique())
    user_ids = [review_dataset[i]['user_id'] for i in range(32)]
    item_ids = [review_dataset[i]['item_id'] for i in range(32)]
    
    