from torch import nn
from torch.nn import functional as F
import torch

class EmbedHistory(nn.Module):
    def __init__(self, embedding_layer, transformer_encoder):
        super(EmbedHistory, self).__init__()
        self.embedding_layer = embedding_layer
        self.transformer_encoder = transformer_encoder
    def forward(self, sequences):
        embedded = self.embedding_layer(sequences.long())
        # Pass through transformer encoder
        encoded = self.transformer_encoder(embedded)
        # Pooling operation
        pooled = F.avg_pool1d(encoded.transpose(1, 2), encoded.size(1)).squeeze(2)
        return pooled

if __name__ == '__main__':
    from module.process_history_feature import process_history_feature
    transformer_encoder = nn.TransformerEncoderLayer(d_model=10, nhead=2)
    embed_history = EmbedHistory(nn.Embedding(1000, 10), transformer_encoder)

    # Create a batch of sequences (already padded)
    
    # create a tensor of size (32, 10)
    sequences = torch.rand(32, 10) * 1000

    history_features = embed_history(sequences)
    print(f"History features size: {history_features.size()}")  # Should be (3, 10)

    # Example other features
    other_features = torch.randn(3, 5)  # (batch_size, feature_dim)
    print(f"Other features size: {other_features.size()}")  # Should be (3, 5)

    # Concatenate features along the second dimension
    combined_features = torch.cat([other_features, history_features], dim=1)
    print(f"Combined features size: {combined_features.size()}")  # Should be (3, 15)

    # Define a feedforward network
    class FeedForwardNetwork(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(FeedForwardNetwork, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    feedforward_network = FeedForwardNetwork(input_dim=15, hidden_dim=20, output_dim=1)

    # Pass the combined features through the feedforward network
    output = feedforward_network(combined_features)
    print(f"Output size: {output.size()}")  # Should be (3, 1)