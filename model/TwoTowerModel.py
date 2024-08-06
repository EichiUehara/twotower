from sklearn.preprocessing import LabelEncoder
from torch.nn import functional as F
import torch
from torch import nn

from dataloader.amazon_review.get_data_loaders import get_data_loaders
from layer.FeatureEmbeddingLayer import FeatureEmbeddingLayer
from module.FaissIndex import FaissIndex

class TwoTowerBinaryModel(nn.Module):
    def __init__(self, 
                 embedding_dim, num_faiss_clusters, 
                 item_label_encoder: LabelEncoder, user_label_encoder: LabelEncoder,
                 ReviewDataset, UserDataset, ItemDataset):
        super(TwoTowerBinaryModel, self).__init__()
        # self.user_embedding = nn.Embedding(len(UserDataset), embedding_dim)
        # self.item_embedding = nn.Embedding(len(ItemDataset), embedding_dim)
        self.user_features_embedding = FeatureEmbeddingLayer(embedding_dim, UserDataset)
        self.item_features_embedding = FeatureEmbeddingLayer(embedding_dim, ItemDataset)
        self.item_label_encoder = item_label_encoder
        self.item_label_encoder.fit(ItemDataset.dataframe.index)
        self.user_label_encoder = user_label_encoder
        self.user_label_encoder.fit(UserDataset.dataframe.index)
        self.review_dataset = ReviewDataset
        self.user_dataset = UserDataset
        self.item_dataset = ItemDataset
        self.FaissIndex = FaissIndex(embedding_dim, num_faiss_clusters)
        self.get_data_loaders = get_data_loaders

    def forward(self, user_ids, item_ids):
        # user_embedding = self.user_embedding(torch.tensor(user_ids))
        # item_embedding = self.item_embedding(torch.tensor(item_ids))
        # user_feature_embedding = self.user_features_embedding(user_ids)
        # item_feature_embedding = self.item_features_embedding(item_ids)
        # user_emb = user_embedding + user_feature_embedding
        # item_emb = item_embedding + item_feature_embedding        
        user_emb = self.user_features_embedding(user_ids)
        item_emb = self.item_features_embedding(item_ids)
        interaction_score = torch.sum(user_emb * item_emb, dim=1)
        interaction_prob = torch.sigmoid(interaction_score)
        return interaction_prob

    def index_train(self, item_ids):
        with torch.no_grad():
            item_ids = self.item_label_encoder.transform(item_ids)
            item_embedding = self.item_embedding(item_ids)
            item_feature_embedding = self.item_features_embedding(item_ids)
            item_embedding = item_embedding + item_feature_embedding
            self.FaissIndex.train(item_embedding)
    
    def index_add(self, embedding, item_ids):
        self.FaissIndex.add(embedding, item_ids)
    
    def index_search(self, user_ids, topk):
        with torch.no_grad():
            user_ids = self.user_label_encoder.transform(user_ids)
            user_embedding = self.user_embedding(user_ids)
            user_feature_embedding = self.user_features_embedding(user_ids)
            user_embedding = user_embedding + user_feature_embedding
            _ , indices = self.FaissIndex.search(user_embedding, topk)
            return indices
    
    def fit(self, optimizer, data_loader):
        self.train()
        for batch in data_loader:
            user_features = batch['user_id']
            item_features = batch['item_id']
            labels = batch['rating']
            loss = self.train_step(optimizer, user_features, item_features, labels)
            print(f"Loss: {loss}")

    def train_step(self, optimizer, user_features, item_features, labels):
        interaction_prob = self.forward(user_features, item_features)
        loss = F.binary_cross_entropy(interaction_prob, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def inference(self, user_features, topk):
        return self.index_search(user_features, topk)


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