from torch import nn
from torch.nn import functional as F
import torch

class NormNumeric(nn.Module):
    def __init__(self, num_features):
        super(NormNumeric, self).__init__()
        self.norm = nn.LayerNorm(num_features)
    def forward(self, feature):
        if feature.dim() == 1:
            feature = feature.view(-1, 1)
        elif feature.dim() == 2:
            if feature.size(1) != self.norm.normalized_shape[0]:
                feature = feature.view(-1, self.norm.normalized_shape[0])
        else:
            raise ValueError("Feature tensor must be 1D or 2D.")
        
        return self.norm(feature)

if __name__ == '__main__':
    class FeedForwardNetwork(nn.Module):
        def __init__(self, numeric_features_dim, other_features_dim, hidden_dim, output_dim):
            super(FeedForwardNetwork, self).__init__()
            self.fc1 = nn.Linear(numeric_features_dim + other_features_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, numeric_features, other_features):
            # Concatenate numeric features and other features along the feature dimension
            combined_features = torch.cat([numeric_features, other_features], dim=1)
            x = F.relu(self.fc1(combined_features))
            x = self.fc2(x)
            return x
    batch_size = 5
    numeric_features_dim = 1
    other_features_dim = 3
    hidden_dim = 10
    output_dim = 1

    # Initialize the normalization layer and feed-forward network
    norm_numeric = NormNumeric(numeric_features_dim)
    feedforward_network = FeedForwardNetwork(numeric_features_dim=numeric_features_dim, 
                                             other_features_dim=other_features_dim, 
                                             hidden_dim=hidden_dim, 
                                             output_dim=output_dim)

    # Example numeric features
    # numeric_features_batch = torch.tensor([[5., 5.], [5., 4.], [5., 1.], [4., 5.], [5., 5.]])  # (batch_size, numeric_features_dim)
    numeric_features_batch = torch.tensor([5., 5., 5., 4., 5.])  # (batch_size, numeric_features_dim)
    normalized_numeric_features = norm_numeric(numeric_features_batch)
    print(f"Normalized numeric features shape: {normalized_numeric_features.shape}")

    # Example other features
    other_features_batch = torch.randn(batch_size, other_features_dim)  # (batch_size, other_features_dim)
    print(f"Other features shape: {other_features_batch.shape}")

    # Pass normalized numeric features and other features through the feed-forward network
    output = feedforward_network(normalized_numeric_features, other_features_batch)
    print(f"Output shape: {output.shape}")