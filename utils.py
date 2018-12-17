from data_loader import DataLoader
import numpy as np
import pickle
import random
from sklearn.feature_extraction.text import TfidfTransformer
from gensim.models import KeyedVectors
from scipy.spatial.distance import euclidean, cosine

'''General'''
def dense_arr_to_tuples(arr):
    tuples = []
    for idx, score in enumerate(arr):
        if score > 0:
            tuples.append((idx, score))
    return tuples

def dense_mat_to_tuples(mat):
    all_tuples = []
    for arr in mat:
        tuples = dense_arr_to_tuples(arr)
        all_tuples.append(tuples)
    return all_tuples

def indices_to_tuples(indices):
    tuple_dict = {}
    j = 0
    while j < len(indices) and indices[j] > 0:
        idx = indices[j]
        if idx in tuple_dict:
            tuple_dict[idx] += 1
        else:
            tuple_dict[idx] = 1
        j += 1
    return list(tuple_dict.items())

'''Tweet-level'''
def make_bow_for_tweet(vocab_size, indices):
    bow = np.zeros(vocab_size)
    j = 0
    while j < len(indices) and indices[j] > 0:  # ignore padding
        bow[indices[j]] += 1
        j += 1
    return bow

def make_emb_for_tweet(wv, toks):
    embs = []
    for tok in toks:
        if tok in wv:
            embs.append(wv[tok])
    if len(embs) > 0:
        return np.mean(embs, axis=0)
    return None

def make_tweet_mat_bow(dl, tweet_ids):
    num_tweets = len(tweet_ids)
    X = np.zeros((num_tweets, dl.vocab_size), dtype=int)
    for j, tid in enumerate(tweet_ids):
        indices = dl.get_tweet(tid)['indices']
        X[j] = make_bow_for_tweet(dl.vocab_size, indices)
    return X

def make_tweet_mat_emb(dl, wv, tweet_ids):
    X = []
    for tid in tweet_ids:
        toks = dl.get_tweet(tid)['tokens']
        emb = make_emb_for_tweet(wv, toks)
        if emb is not None:
            X.append(emb)
    return np.array(X)

def get_topics_for_tweets(dl, wv, mix, tweet_ids):
    X = make_tweet_mat_emb(dl, wv, tweet_ids)
    return mix.predict(X)

def bow_to_tfidf(X):
    tfidf = TfidfTransformer().fit_transform(X)
    return tfidf.toarray()

'''User-level'''
def make_bow_for_user(dl, uid, agg=True):
    tids = dl.get_tweets_by_user(uid)
    bow = make_tweet_mat_bow(dl, tids)
    if agg:
        return np.sum(bow, axis=0)
    return bow

def make_emb_for_user(dl, wv, uid, agg=True):
    tids = dl.get_tweets_by_user(uid)
    embs = make_tweet_mat_emb(dl, wv, tids)
    if embs.shape[0] > 0:
        if agg:
            return np.mean(embs, axis=0)
        return embs
    return None

def get_topics_for_user(dl, wv, mix, uid, agg=True):
    tids = dl.get_tweets_by_user(uid)
    topics = get_topics_for_tweets(dl, wv, mix, tids)
    if topics.shape[0] > 0:
        if agg:
            vec = np.zeros((len(mix.weights_)))
            for t in topics:
                vec[t] += 1
            return vec
        return topics
    return None

def filter_users(dl, cutoff=1000):
    filtered = set()
    for uid in dl.get_user_ids():
        if len(dl.get_tweets_by_user(uid)) >= cutoff:
            filtered.add(uid)
    return filtered

def make_random_init_vecs(uids, dim=300):
    user_vecs = {}
    for uid in uids:
        user_vecs[uid] = np.random.rand(dim)
    return user_vecs

def make_bow_vecs(uids, dl):
    user_vecs = {}
    for uid in uids:
        bow = make_bow_for_user(dl, uid, agg=True)
        bow = np.add(bow, 1)  # smoothing
        user_vecs[uid] = np.divide(bow, np.sum(bow))
    return user_vecs

def save_bow_distances():
    user_vecs = pickle.load(open('./saved/user_vecs_bow.pkl', 'rb'))
    uids = list(user_vecs.keys())
    num_users = len(uids)
    dists = []
    for i in range(num_users):
        ui = uids[i]
        for j in range(i+1, num_users):
            uj = uids[j]
            print('{}-{}'.format(ui, uj))
            dist = get_dist(user_vecs, ui, uj, dist_func='btc')
            dists.append(dist)
    pickle.dump(dists, open('./saved/user_vecs_bow_dists.pkl', 'wb'))
    print('mean: {}, std: {}, max: {}, min: {}'.format(
        round(np.mean(dists), 4), round(np.std(dists), 4), round(np.max(dists), 4), round(np.min(dists), 4)))

def make_avg_emb_vecs(uids, dl, wv):
    user_vecs = {}
    for uid in uids:
        emb = make_emb_for_user(dl, wv, uid, agg=True)
        if emb is not None:
            print('Adding vector for', dl.uid_to_uname(uid))
            user_vecs[uid] = emb
        else:
            print('No embeddings found for', dl.uid_to_uname(uid))
    return user_vecs

def make_topic_vecs(uids, dl, wv, mix):
    user_vecs = {}
    for uid in uids:
        topic_counts = get_topics_for_user(dl, wv, mix, uid, agg=True)
        if topic_counts is not None:
            print('Adding vector for', dl.uid_to_uname(uid))
            user_vecs[uid] = np.divide(topic_counts, sum(topic_counts))
        else:
            print('No topics found for', dl.uid_to_uname(uid))
    return user_vecs

def bhattacharyya(u, v):
    sigma = np.sum(np.sqrt(np.multiply(u, v)))
    return np.multiply(np.log(sigma), -1)

def get_dist(user_vecs, ui, uj, dist_func):
    if ui in user_vecs and uj in user_vecs:
        if dist_func == 'euc':
            return euclidean(user_vecs[ui], user_vecs[uj])
        elif dist_func == 'cos':
            return cosine(user_vecs[ui], user_vecs[uj])
        else:
            return bhattacharyya(user_vecs[ui], user_vecs[uj])
    return None

def get_stats(user_vecs, dist_func='euc'):
    assert(dist_func == 'euc' or dist_func == 'cos' or dist_func == 'btc')
    uids = list(user_vecs.keys())
    num_users = len(uids)
    dists = []
    for i in range(num_users):
        ui = uids[i]
        for j in range(i+1, num_users):
            uj = uids[j]
            dist = get_dist(user_vecs, ui, uj, dist_func)
            dists.append(dist)
    return np.mean(dists), np.std(dists), np.max(dists), np.min(dists)

def print_stats(user_vecs, dist_func='euc'):
    mean, std, max, min = get_stats(user_vecs, dist_func=dist_func)
    print('Maximum:', round(max, 4))
    print('Minimum:', round(min, 4))
    print('Average:', round(mean, 4))
    print('Standard dev:', round(std, 4))

if __name__ == '__main__':
    dl = DataLoader()
    users = filter_users(dl)
    print('Num filtered users:', len(users))
    user_vecs = make_bow_vecs(users, dl)
    pickle.dump(user_vecs, open('./saved/user_vecs_bow.pkl', 'wb'))

    # wv = KeyedVectors.load_word2vec_format('./models/word_embs_d300.bin', binary=True)
    # mix = pickle.load(open('./models/tweet_gmm_k10.pkl', 'rb'))
    #



    # uv_rand = pickle.load(open('./saved/user_vecs_rand.pkl', 'rb'))
    # print('Random')
    # print_stats(uv_rand, dist_func='euc')
    # print()
    #
    # uv_emb = pickle.load(open('./saved/user_vecs_emb.pkl', 'rb'))
    # print('Avg Emb')
    # print_stats(uv_emb, dist_func='cos')
    # print()
    #
    # uv_topics = pickle.load(open('./saved/user_vecs_topics.pkl', 'rb'))
    # print('Topic Freq')
    # print_stats(uv_topics, dist_func='btc')
    #
    # uv_hmm = pickle.load(open('./saved/user_vecs_hmm.pkl', 'rb'))
    # print('HMM')
    # print_stats(uv_hmm, dist_func='cos')
    # print()
