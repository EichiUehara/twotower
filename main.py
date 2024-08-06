from sklearn.preprocessing import LabelEncoder

from model.TwoTowerModel import TwoTowerBinaryModel


if __name__ == '__main__':
    from module.Tokenizer import Tokenizer
    from dataset.amazon_review.UserDataset import UserDataset
    from dataset.amazon_review.ItemDataset import ItemDataset
    from dataset.amazon_review.ReviewDataset import ReviewDataset
    import time
    item_label_encoder = LabelEncoder()
    user_label_encoder = LabelEncoder()
    tokenizer = Tokenizer('BAAI/bge-base-en-v1.5')
    review_dataset = ReviewDataset('All_Beauty', item_label_encoder, user_label_encoder)
    item_label_encoder.fit(review_dataset.dataframe['parent_asin'].unique())
    user_label_encoder.fit(review_dataset.dataframe['user_id'].unique())
    user_dataset = UserDataset(
        'All_Beauty', tokenizer,
        item_label_encoder=item_label_encoder, user_label_encoder=user_label_encoder,
        max_history_length=10)
    item_dataset = ItemDataset('All_Beauty', tokenizer, item_label_encoder=item_label_encoder)
    model = TwoTowerBinaryModel(64, 10, 
            item_label_encoder, 
            user_label_encoder,
            review_dataset, user_dataset, item_dataset)
    user_ids = [review_dataset[i]['user_id'] for i in range(32)]
    item_ids = [review_dataset[i]['item_id'] for i in range(32)]
    start = time.time()
    print(model(user_ids, item_ids))
    print(time.time() - start)
    # model.index_add(item_ids, item
    # model.index_train()
    # print(model.index_search(ids, 10))
    # model.fit(torch.optim.Adam(model.parameters()), get_data_loaders(user_dataset, item_dataset, review_dataset, 32))
    # print(model.inference(ids, 10))