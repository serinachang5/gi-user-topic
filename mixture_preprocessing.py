import numpy as np
from data_loader import DataLoader

'''Tweet-level topic modeling'''
def make_train(dl, num_tweets):
    all_tids = list(dl.get_tweet_ids())
    assert(num_tweets <= len(all_tids))
    np.random.shuffle(all_tids)
    return all_tids[:num_tweets]

def make_test(dl, num_tweets, train_tids):
    all_tids = set(dl.get_tweet_ids())
    valid_test = list(all_tids - set(train_tids))
    assert(num_tweets <= len(valid_test))
    np.random.shuffle(valid_test)
    return valid_test[:num_tweets]