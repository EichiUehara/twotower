import math
from module.embedding_dim import embedding_dim

def transformer_head(num_classes):
    return math.ceil(math.sqrt(math.log2(math.sqrt(embedding_dim(num_classes)))))