from torch.utils.data import Dataset
from datasets import load_dataset

class ReviewDataset(Dataset):
    def __init__(self, amazon_category):
        review_df = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{amazon_category}", split="full").to_pandas()
        review_df = review_df[['user_id', 'parent_asin', 'rating']]
        review_df = review_df.drop_duplicates(subset=['user_id', 'parent_asin'], keep='first')
        review_df['rating'] = review_df['rating'].apply(lambda x: 1 if x >= 4 else 0)
        self.dataframe = review_df

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        return {
            'user_id': row['user_id'], # user_id: str
            'item_id': row['parent_asin'], # item_id: str
            'rating': row['rating'] # rating: bool
        }
        
if __name__ == '__main__':
    review_dataset = ReviewDataset('All_Beauty')
    user_ids = [review_dataset[i]['user_id'] for i in range(32)]
    item_ids = [review_dataset[i]['item_id'] for i in range(32)]