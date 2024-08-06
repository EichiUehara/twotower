import torch

def process_history_feature(history_feature, max_history_length):
    return torch.cat(
        (history_feature, torch.zeros(max_history_length - len(history_feature), dtype=history_feature.dtype)), 0).unsqueeze(0)
if __name__ == '__main__':
    history_feature = torch.tensor([1, 2, 3])
    max_history_length = 10
    print(process_history_feature(history_feature, max_history_length).shape) # (1, max_history_length)