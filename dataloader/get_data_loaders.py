import torch
from torch.utils.data import DataLoader, Dataset

def get_data_loaders(dataset: Dataset, batch_size, train_size, collate_fn=None) -> tuple[DataLoader, DataLoader]:
    train_size = int(train_size * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, collate_fn=collate_fn), \
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, collate_fn=collate_fn)

if __name__ == '__main__':
    from sklearn.preprocessing import LabelEncoder
    from dataset.amazon_review.UserDataset import UserDataset
    from dataset.amazon_review.ItemDataset import ItemDataset
    from dataset.amazon_review.ReviewDataset import ReviewDataset
    from module.Tokenizer import Tokenizer
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    item_label_encoder = LabelEncoder()
    user_label_encoder = LabelEncoder()
    tokenizer = Tokenizer()
    item_dataset = ItemDataset('All_Beauty',tokenizer, item_label_encoder=item_label_encoder)
    item_label_encoder.fit(item_dataset.dataframe.index)
    user_dataset = UserDataset(
        'All_Beauty', tokenizer,
        item_label_encoder=item_label_encoder, user_label_encoder=user_label_encoder,
        max_history_length=10)
    user_label_encoder.fit(user_dataset.dataframe.index)
    review_dataset = ReviewDataset(
        'All_Beauty', 
        item_label_encoder=item_label_encoder, user_label_encoder=user_label_encoder)
    print(user_dataset[0])
    print(item_dataset[0])
    print(review_dataset[0])
    train_loader, val_loader = get_data_loaders(user_dataset, 32, 0.8, collate_fn=user_dataset.collate_fn)
    print(next(iter(train_loader)))
    print(next(iter(val_loader)))
    train_loader, val_loader = get_data_loaders(item_dataset, 32, 0.8, collate_fn=item_dataset.collate_fn)
    print(next(iter(train_loader)))
    print(next(iter(val_loader)))
    train_loader, val_loader = get_data_loaders(review_dataset, 32, 0.8)
    print(next(iter(train_loader)))
    print(next(iter(val_loader)))