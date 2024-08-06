from typing import Tuple
import faiss
import numpy as np

class FaissIndex:
    def __init__(self, embedding_dim, num_clusters, retrain_threshold=1000):
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.retrain_threshold = retrain_threshold
        self.embeddings = None
        self.ids = None
        self.index_ivf_sq = None
        self.aux_index = None
        self.new_embeddings = []
        self.new_ids = []
        self.create_main_index()

    def create_main_index(self):
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        self.index_ivf_sq = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.num_clusters, faiss.METRIC_INNER_PRODUCT)
        self.index_ivf_sq = faiss.IndexIDMap(self.index_ivf_sq)

    def create_aux_index(self):
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        self.aux_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 1, faiss.METRIC_INNER_PRODUCT)
        self.aux_index = faiss.IndexIDMap(self.aux_index)

    def train(self, embeddings, ids):
        if self.embeddings is None:
            self.embeddings = embeddings
            self.ids = ids
        else:
            self.embeddings = np.vstack((self.embeddings, embeddings))
            self.ids = np.hstack((self.ids, ids))

        if not self.index_ivf_sq.is_trained:
            self.index_ivf_sq.train(self.embeddings)
        
        self.index_ivf_sq.add_with_ids(embeddings, ids)

    def add(self, embeddings: np.ndarray, ids: np.ndarray):
        self.new_embeddings.append(embeddings)
        self.new_ids.append(ids)

        if len(self.new_embeddings) >= self.retrain_threshold:
            self.retrain()
        else:
            if self.aux_index is None:
                self.create_aux_index()
            if not self.aux_index.is_trained:
                self.aux_index.train(embeddings)
            self.aux_index.add_with_ids(embeddings, ids)

    def retrain(self):
        if len(self.new_embeddings) == 0:
            return

        # Combine new embeddings and IDs
        new_embeddings = np.vstack(self.new_embeddings)
        new_ids = np.hstack(self.new_ids)

        # Retrain index with new embeddings
        self.embeddings = np.vstack((self.embeddings, new_embeddings))
        self.ids = np.hstack((self.ids, new_ids))

        self.create_main_index()
        self.index_ivf_sq.train(self.embeddings)
        self.index_ivf_sq.add_with_ids(self.embeddings, self.ids)

        # Clear new embeddings and IDs
        self.new_embeddings = []
        self.new_ids = []

        # Reset auxiliary index
        self.aux_index = None

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        D_main, I_main = self.index_ivf_sq.search(query, k)
        if self.aux_index is not None and self.aux_index.ntotal > 0:
            D_aux, I_aux = self.aux_index.search(query, k)
            D = np.hstack((D_main, D_aux))
            I = np.hstack((I_main, I_aux))
            sorted_indices = np.argsort(D, axis=1)
            D_sorted = np.take_along_axis(D, sorted_indices, axis=1)
            I_sorted = np.take_along_axis(I, sorted_indices, axis=1)
            return D_sorted[:, :k], I_sorted[:, :k]
        else:
            return D_main, I_main

if __name__ == '__main__':
    embedding_dim = 64
    num_clusters = 25
    retrain_threshold = 1000
    index = FaissIndex(embedding_dim, num_clusters, retrain_threshold)

    # Initial training
    embeddings = np.random.rand(1000000, embedding_dim).astype(np.float32)
    index.train(embeddings)

    # Adding new embeddings
    new_embeddings = np.random.rand(100, embedding_dim).astype(np.float32)
    new_ids = np.arange(1000000, 1000100)
    for _ in range(10):  # Adding new embeddings in batches to trigger retrain
        index.add(new_embeddings, new_ids)

    # Perform search
    query = np.random.rand(1, embedding_dim).astype(np.float32)
    k = 5
    distance, indices = index.search(query, k)
    print(distance)
    print(indices)
