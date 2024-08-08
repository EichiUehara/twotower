from sklearn.preprocessing import LabelEncoder
import torch

from model.TwoTowerModel import TwoTowerBinaryModel


if __name__ == '__main__':
    from module.Tokenizer import Tokenizer
    from dataset.amazon_review_base.UserDataset import UserDataset
    from dataset.amazon_review_base.ItemDataset import ItemDataset
    from dataset.amazon_review_base.ReviewDataset import ReviewDataset
    # # load the model
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    