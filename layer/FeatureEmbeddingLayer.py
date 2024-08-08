from torch import nn
import torch

from layer.FeedForwardNetwork import FeedForwardNetwork
from module.EmbedCategory import EmbedCategory
from module.EmbedHistory import EmbedHistory
from module.NormNumeric import NormNumeric
from module.EmbedText import EmbedText
from module.EmbedTextHistory import EmbedTextHistory

class FeatureEmbeddingLayer(nn.Module):
    def __init__(self, embedding_dim, dataset, model_name="BAAI/bge-base-en-v1.5"):
        super(FeatureEmbeddingLayer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.id_embedding = nn.Embedding(
            self.dataset.hyperparameters['id']['num_classes'],
            self.dataset.hyperparameters['id']['embedding_dim']
        )
        self.output = FeedForwardNetwork(
            dataset.hyperparameters['feedforward_network']['input_dim'],
            embedding_dim * 2,
            embedding_dim
        )
        self.embed_categorical = {}
        self.embed_history = {}
        self.embed_text = {}
        self.embed_text_history = {}
        self.embed_numerical = NormNumeric(len(dataset.numerical_features))
        for feature in dataset.categorical_features:
            self.embed_categorical[feature] = EmbedCategory(
                dataset.hyperparameters['categorical_features'][feature]['num_classes'],
                dataset.hyperparameters['categorical_features'][feature]['embedding_dim']
            )
        for feature in dataset.history_features:
            self.embed_history[feature] = EmbedHistory(
                nn.Embedding(
                    dataset.hyperparameters['history_features'][feature]['num_classes'],
                    dataset.hyperparameters['history_features'][feature]['embedding_dim']
                ),
                nn.TransformerEncoderLayer(
                    dataset.hyperparameters['history_features'][feature]['embedding_dim'], 
                    dataset.hyperparameters['history_features'][feature]['transformer_head']
                )
            )
        for feature in dataset.text_features:
            self.embed_text[feature] = EmbedText(
                model_name
            )
        for feature in dataset.text_history_features:
            self.embed_text_history[feature] = EmbedTextHistory(
                model_name
            )

    def forward(self, ids)->torch.Tensor:
        print()
        ids = ids.to(self.device)
        batch = [self.dataset[id] for id in ids]
        batch = self.dataset.collate_fn(batch)
        batch = {
            'id': batch['id'].to(self.device),
            'numerical_features': batch['numerical_features'].to(self.device),
            'categorical_features': {k: v.to(self.device) for k, v in batch['categorical_features'].items()},
            'text_features': {k: {k2: v2.to(self.device) for k2, v2 in v.items()} for k, v in batch['text_features'].items()},
            'history_features': {k: v.to(self.device) for k, v in batch['history_features'].items()},
            'text_history_features': {k: {k2: v2.to(self.device) for k2, v2 in v.items()} for k, v in batch['text_history_features'].items()}
        }

        print(f"ids device: {ids.device}")
        print(f"Batch 'id' device: {batch['id'].device}")
        print(f"Batch 'numerical_features' device: {batch['numerical_features'].device}")
        embedded_features = []
        embedded_features.append(self.id_embedding(batch['id']))
        embedded_features.append(self.embed_numerical(batch['numerical_features']))
        
        for feature in self.dataset.categorical_features:
            print(f"Batch 'categorical_features[{feature}]' device: {batch['categorical_features'][feature].device}")
            embedded_features.append(self.embed_categorical[feature](batch['categorical_features'][feature]))
            
        for feature in self.dataset.history_features:
            print(f"Batch 'history_features[{feature}]' device: {batch['history_features'][feature].device}")
            embedded_features.append(self.embed_history[feature](batch['history_features'][feature]))
            
        for feature in self.dataset.text_features:
            print(f"Batch 'text_features[{feature}]' device: {batch['text_features'][feature].device}")
            embedded_features.append(self.embed_text[feature](batch['text_features'][feature]))
            
        for feature in self.dataset.text_history_features:
            print(f"Batch 'text_history_features[{feature}]' device: {batch['text_history_features'][feature].device}")
            embedded_features.append(self.embed_text_history[feature](batch['text_history_features'][feature]))
        concatenated = torch.cat(embedded_features, dim=1)
        return self.output(concatenated)