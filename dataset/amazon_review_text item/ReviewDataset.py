from dataset.amazon_review.ReviewDataset import ReviewDataset
class ReviewDataset(ReviewDataset):
    def __init__(self, category):
        super().__init__(category)

if __name__ == '__main__':
    review_dataset = ReviewDataset('All_Beauty')
    print(len(review_dataset))