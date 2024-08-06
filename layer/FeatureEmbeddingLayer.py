import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torch import nn
import torch

from module.EmbedCategory import EmbedCategory
from module.EmbedHistory import EmbedHistory
from module.NormNumeric import NormNumeric
from module.EmbedText import EmbedText
from module.EmbedTextHistory import EmbedTextHistory

class FeatureEmbeddingLayer(nn.Module):
    def __init__(self, embedding_dim, dataset):
        super(FeatureEmbeddingLayer, self).__init__()
        self.id_embedding = nn.Embedding(len(dataset), 200)
        self.embed_categorical = EmbedCategory(1000000, 20)
        self.embed_history = EmbedHistory(
            nn.Embedding(1000000, 50),
            nn.TransformerEncoderLayer(d_model=50, nhead=2))
        self.embed_numerical = NormNumeric(len(dataset.numerical_features))
        self.embed_text = EmbedText("BAAI/bge-base-en-v1.5")
        self.embed_text_history = EmbedTextHistory(
            "BAAI/bge-base-en-v1.5"
        )
        self.dataset = dataset
        self.output = nn.Linear(dataset.input_dim, embedding_dim)

    def forward(self, ids)->torch.Tensor:
        device = next(self.parameters()).device
        batch = [self.dataset[id] for id in ids]
        batch = self.dataset.collate_fn(batch)
        embedded_features = []
        embedded_features.append(self.id_embedding(torch.tensor(ids).to(device)))
        if len(batch['numerical_features']) > 0:
            batch['numerical_features'].to(device)
            embedded_features.append(
                self.embed_numerical(batch['numerical_features']))
        if len(batch['categorical_features']) > 0:
            for feature in batch['categorical_features']:
                batch['categorical_features'][feature].to(device)
                embedded_features.append(
                    self.embed_categorical(batch['categorical_features'][feature]))
        if len(batch['history_features']) > 0:
            for feature in batch['history_features']:
                batch['history_features'][feature].to(device)
                embedded_features.append(
                    self.embed_history(batch['history_features'][feature]))
        if len(batch['text_features']) > 0:
            for feature in batch['text_features']:
                batch['text_features'][feature].to(device)
                embedded_features.append(
                    self.embed_text(batch['text_features'][feature]))
        if len(batch['text_history_features']) > 0:
            for feature in batch['text_history_features']:
                batch['text_history_features'][feature].to(device)
                embedded_features.append(
                    self.embed_text_history(batch['text_history_features'][feature]))
        concatenated = torch.cat(embedded_features, dim=1)
        return self.output(concatenated)
    
if __name__ == '__main__':
    from module.Tokenizer import Tokenizer
    from dataset.amazon_review.UserDataset import UserDataset
    from sklearn.preprocessing import LabelEncoder
    from dataset.amazon_review.ItemDataset import ItemDataset
    import time
    item_label_encoder = LabelEncoder()
    user_label_encoder = LabelEncoder()
    tokenizer = Tokenizer('BAAI/bge-base-en-v1.5')
    user_dataset = UserDataset(
        'All_Beauty', tokenizer,
        item_label_encoder=item_label_encoder, user_label_encoder=user_label_encoder,
        max_history_length=10)
    user_dataset.user_label_encoder.fit(user_dataset.dataframe.index)
    item_dataset = ItemDataset('All_Beauty', tokenizer, item_label_encoder=item_label_encoder)
    item_dataset.item_label_encoder.fit(item_dataset.dataframe.index)
    # User_embedding = FeatureEmbeddingLayer(user_dataset.input_dim, 64, user_dataset)
    # ids = [user_dataset[i]['id'] for i in range(32)]
    # start = time.time()
    # print(User_embedding(ids).shape)
    # print(time.time() - start)
    item_embedding = FeatureEmbeddingLayer(item_dataset.input_dim, 64, item_dataset)
    ids = [item_dataset[i]['id'] for i in range(32)]
    start = time.time()
    print(item_embedding(ids).shape)
    print(time.time() - start)