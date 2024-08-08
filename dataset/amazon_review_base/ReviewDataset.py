import os
import zipfile
import pandas as pd

from torch.utils.data import Dataset
from datasets import load_dataset

class ReviewDataset(Dataset):
    def __init__(self, amazon_category):
        if os.path.exists(f'dataset/amazon_review_base/raw_review_{amazon_category}.csv.zip'):
            review_df = pd.read_csv(f'dataset/amazon_review_base/raw_review_{amazon_category}.csv.zip')
        else:
            review_df = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{amazon_category}", split="full", trust_remote_code=True).to_pandas()
            review_df.to_csv(f'dataset/amazon_review_base/raw_review_{amazon_category}.csv', index=False, escapechar='\\')
            with zipfile.ZipFile(f'dataset/amazon_review_base/raw_review_{amazon_category}.csv.zip', 'w', zipfile.ZIP_DEFLATED) as z:
                z.write(f'dataset/amazon_review_base/raw_review_{amazon_category}.csv')
            os.remove(f'dataset/amazon_review_base/raw_review_{amazon_category}.csv')
        if os.path.exists(f'dataset/amazon_review_base/raw_meta_{amazon_category}.csv.zip'):
            item_df = pd.read_csv(f'dataset/amazon_review_base/raw_meta_{amazon_category}.csv.zip')
        else:
            item_df = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{amazon_category}", split="full", trust_remote_code=True).to_pandas()
            item_df.to_csv(f'dataset/amazon_review_base/raw_meta_{amazon_category}.csv', index=False, escapechar='\\')
            with zipfile.ZipFile(f'dataset/amazon_review_base/raw_meta_{amazon_category}.csv.zip', 'w', zipfile.ZIP_DEFLATED) as z:
                z.write(f'dataset/amazon_review_base/raw_meta_{amazon_category}.csv')
            os.remove(f'dataset/amazon_review_base/raw_meta_{amazon_category}.csv')
        review_df = review_df[['user_id', 'parent_asin', 'rating']]
        review_df = review_df.drop_duplicates(subset=['user_id', 'parent_asin'], keep='first')
        review_df['rating'] = review_df['rating'].apply(lambda x: 1 if x >= 4 else 0)
        review_df = review_df[review_df['parent_asin'].isin(item_df['parent_asin'].unique())]
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
    print(len(review_dataset))
    batch = next(iter(review_dataset))
    print(batch['user_id'], batch['item_id'], batch['rating'])