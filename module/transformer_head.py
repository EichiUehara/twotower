import math
from module import embedding_dim

def transformer_head(num_classes):
    return math.ceil(math.log2(math.sqrt(embedding_dim(num_classes))))