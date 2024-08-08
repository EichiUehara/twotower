import math
def embedding_dim(target):
    sum_log = 1
    quad_root = int(math.sqrt(math.sqrt(target) + 25) +25 ) +25
    for i in range(2, quad_root + 25):
        sum_log += math.log(quad_root, i)
    dim = int(min(math.sqrt(target), sum_log))
    if dim > 50:
        dim = int(min(math.sqrt(math.sqrt(target) + 100) + 100, dim))
    if dim > 250:
        dim = int(min(math.sqrt(math.sqrt(target) + 500) + 500, dim))
    return dim



if __name__ == "__main__":
    print(embedding_dim(2))
    print(embedding_dim(9))
    print(embedding_dim(10))
    print(embedding_dim(100))
    print(embedding_dim(1000))
    print(embedding_dim(10000))
    print(embedding_dim(100000))
    print(embedding_dim(1000000))
    print(embedding_dim(10000000))
    print(embedding_dim(100000000))
    print(embedding_dim(1000000000))
    print(embedding_dim(10000000000))
    print(embedding_dim(100000000000))
    print(embedding_dim(1000000000000))
