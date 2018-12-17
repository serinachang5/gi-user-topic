import pickle
import numpy as np

class DataLoader:
    def __init__(self, vocab_size = 20000, max_len = 50, verbose = True,
                 load_tweets = True):
        self.vocab_size, self.max_len, self.verbose = vocab_size, max_len, verbose
        if self.verbose:
            print('Initializing DataLoader. Loading vocabulary...')
        self.idx2word = pickle.load(open('./data/idx2word.pkl', 'rb'))[:self.vocab_size]
        self.word2idx = pickle.load(open('./data/word2idx.pkl', 'rb'))
        if self.vocab_size is not None and self.vocab_size < len(self.word2idx):
            if self.verbose:
                print('Orig vocab size:', len(self.word2idx))
            to_remove = []
            for word in self.word2idx:
                if self.word2idx[word] >= self.vocab_size:
                    to_remove.append(word)
            for rem in to_remove:
                del self.word2idx[rem]
        else:
            self.vocab_size = len(self.word2idx)
        assert(len(self.word2idx) == self.vocab_size)
        if self.verbose:
            print('Done. Vocab size:', self.vocab_size)

        if self.verbose:
            print('Loading users...')
        self.user_dict = pickle.load(open('./data/users.pkl', 'rb'))
        self.idx2user = pickle.load(open('./data/idx2user.pkl', 'rb'))
        self.user2idx = pickle.load(open('./data/user2idx.pkl', 'rb'))
        if self.verbose:
            print('Done. User size:', len(self.user_dict))

        if load_tweets:
            if self.verbose:
                print('Loading tweets...')
            self.tweet_dict = pickle.load(open('./data/tweets.pkl', 'rb'))
            if self.verbose:
                print('Done. Num tweets:', len(self.tweet_dict))

            if self.verbose:
                print('Processing tweets...')
            for tid in self.tweet_dict:
                self.tweet_dict[tid]['indices'] = self._process_int_arr(self.tweet_dict[tid]['indices'],
                                                                    self.vocab_size, self.max_len)
        else:
            self.tweet_dict = {}

        if self.verbose:
            print('DataLoader initialization finished.')

    def get_user_ids(self):
        return self.user_dict.keys()

    def has_user_id(self, uid):
        return uid in self.user_dict

    def get_tweets_by_user(self, uid):
        if not self.has_user_id(uid):
            return []
        return self.user_dict[uid]['tweet_ids']

    def uid_to_uname(self, uid):
        if not self.has_user_id(uid):
            return None
        return self.idx2user[uid]

    def get_tweet_ids(self):
        return self.tweet_dict.keys()

    def has_tweet_id(self, tid):
        return tid in self.tweet_dict

    def get_tweet(self, tid):
        if not self.has_tweet_id(tid):
            return None
        return self.tweet_dict[tid]

    def idx_to_str(self, arr):
        return [self.idx2word[i] for i in arr]

    def bow_to_str(self, bow):
        nonzero = np.nonzero(bow)[0]
        indices = []
        for i in nonzero:
            for j in range(int(bow[i])):
                indices.append(i)
        return self.idx_to_str(indices)

    '''Helper functions'''
    def _process_int_arr(self, arr, vocab_size, max_len):
        for i in range(len(arr)):  # anything ranked over vocab_size goes to 1
            if arr[i] >= vocab_size:
                arr[i] = 1
        if len(arr) < max_len:
            arr += [0] * (max_len - len(arr))  # pad with 0's
        else:
            arr = arr[:max_len]  # truncate to max_len
        assert(len(arr) == max_len)
        return arr

if __name__ == '__main__':
    dl = DataLoader(load_tweets=False)
    bow = np.zeros(dl.vocab_size, dtype=int)
    bow[5] = 2
    bow[8] = 1
    bow[2] = 4
    print(dl.bow_to_str(bow))