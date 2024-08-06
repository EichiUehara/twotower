import torch

def process_history_feature(history_feature, max_history_length):
    # Ensure the input is a tensor
    history_feature = torch.tensor(history_feature).squeeze()  # Ensure it's a 1D tensor
    
    if history_feature.dim() == 0:  # Handle the case where history_feature is an empty tensor
        history_feature = history_feature.unsqueeze(0)

    if history_feature.size(0) > max_history_length:
        # Truncate the history_feature to max_history_length
        truncated_history_feature = history_feature[:max_history_length]
    else:
        # Pad the history_feature with zeros to max_history_length
        truncated_history_feature = torch.cat(
            (history_feature, torch.zeros(max_history_length - history_feature.size(0), dtype=history_feature.dtype)), 
            dim=0
        )
    
    return truncated_history_feature
if __name__ == '__main__':
    history_feature = torch.tensor([1, 2, 3])
    max_history_length = 10
    print(process_history_feature(history_feature, max_history_length).shape) # (1, max_history_length)