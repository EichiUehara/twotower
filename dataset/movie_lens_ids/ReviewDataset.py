from dataset.movie_lens_base.ReviewDataset import ReviewDataset as ReviewDatasetBase
class ReviewDataset(ReviewDatasetBase):
    def __init__(self):
        super().__init__()
if __name__ == '__main__':
    review_dataset = ReviewDataset()
    for i in range(5):
        print(review_dataset[i])