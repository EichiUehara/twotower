from dataset.amazon_review_base.ReviewDataset import ReviewDataset
class ReviewDataset(ReviewDataset):
    def __init__(self, category):
        super().__init__(category)

if __name__ == '__main__':
    review_dataset = ReviewDataset('All_Beauty')
    print(len(review_dataset))
    batch = next(iter(review_dataset))
    print(batch['user_id'], batch['item_id'], batch['rating'])