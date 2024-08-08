amazon_category = 'All_Beauty'
import warnings
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available. Disabling")
if __name__ == '__main__':
    import torch
    from model.TwoTowerModel import TwoTowerBinaryModel
    from dataset.amazon_review_text.UserDataset import UserDataset
    from dataset.amazon_review_text.ItemDataset import ItemDataset
    from dataset.amazon_review_text.ReviewDataset import ReviewDataset
    review_dataset = ReviewDataset(amazon_category)
    user_dataset = UserDataset(amazon_category)
    item_dataset = ItemDataset(amazon_category)
    model = TwoTowerBinaryModel(256, 10, user_dataset, item_dataset)
    train_dataloader, val_dataloader = model.get_data_loaders(review_dataset, 1024, 0.8, num_workers=8)
    model.fit(torch.optim.Adam(model.parameters(), lr=0.001), train_dataloader)
    # save the model
    torch.save(model.state_dict(), 'model.pth')
    # # load the model
    # model.load_state_dict(torch.load('model.pth'))
    # model.eval()
    