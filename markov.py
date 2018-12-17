from collections import Counter
from data_loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from scipy.sparse import csc_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import KeyedVectors
from utils import make_tweet_mat_emb, dense_mat_to_tuples
from hmmlearn.hmm import GaussianHMM
from markov_preprocessing import make_emb_seq_for_user, make_topic_seq_for_user
from sklearn.decomposition import PCA
from mixture import train_mixture

class MarkovChain:
    def __init__(self, n_states):
        self.k = n_states
        self.trans_mat = np.zeros((self.k, self.k))
        self.fitted = False

    def set_params(self, priors, trans_mat):
        self.priors = priors
        self.trans_mat = trans_mat
        self.fitted = True

    def fit(self, X, lengths):
        assert(len(X) == np.sum(lengths))
        freq_mat = np.zeros((self.k, self.k), dtype=int)
        i = 0
        for l in lengths:
            samples = X[i:i+l]
            for j in range(len(samples)-1):
                curr_state = samples[j]
                next_state = samples[j+1]
                freq_mat[curr_state][next_state] += 1
            i += l
        freq_mat = np.add(freq_mat, 1)  # smoothing
        for state in range(self.k):
            norm = np.sum(freq_mat[state])  # sum the number of transitions starting with this state
            self.trans_mat[state] = np.divide(freq_mat[state], norm)
        self.fitted = True

    def log_joint(self, X, lengths):
        if not self.fitted:
            print('MarkovChain not fitted yet')
            return
        lj = 0
        i = 0
        for l in lengths:
            samples = X[i:i+l]
            for j in range(len(samples)-1):
                curr_state = samples[j]
                next_state = samples[j+1]
                lj += np.log(self.trans_mat[curr_state][next_state])
            i += l
        return lj

    def print_trans_mat(self):
        for si in range(self.k):
            for sj in range(self.k):
                print('p({}|{}) = {}'.format(si, sj, self.trans_mat[sj][si]))
            print()

def fit_mc(X, L, k):
    mc = MarkovChain(n_states=k)
    mc.fit(X, L)
    print('Finished fitting MarkovChain:', mc.log_joint(X, L))
    return mc

def fit_mc_for_all_users(dl, wv, mix, cutoff=1000):
    k = len(mix.weights_)
    mc_dict = {}
    for uid in dl.get_user_ids():
        tids = dl.get_tweets_by_user(uid)
        if len(tids) >= cutoff:
            print('Fitting MC for {}'.format(dl.uid_to_uname(uid)))
            X, L = make_topic_seq_for_user(uid, dl, wv, mix)
            print('X-shape:', X.shape)
            mc = fit_mc(X, L, k)
            mc_dict[uid] = mc
    pickle.dump(mc_dict, open('./models/mc_dict.pkl'.format(k), 'wb'))

def make_vecs_from_mc(dl, wv, mix, mc_dict):
    uids = sorted(mc_dict.keys())
    num_uids = len(uids)
    print('Num users:', num_uids)
    vecs = {}
    for ui in uids:
        vec = np.zeros(num_uids)
        print('Making vec for {}'.format(dl.uid_to_uname(ui)))
        X, L = make_topic_seq_for_user(ui, dl, wv, mix)
        for j, uj in enumerate(uids):
            mcj = mc_dict[uj]
            vec[j] = mcj.log_joint(X, lengths=L)
        vec = np.divide(vec, X.shape[0])
        vecs[ui] = vec
    return vecs

def train_hmm(X, L, k):
    hmm = GaussianHMM(n_components=k)
    hmm.fit(X, lengths=L)
    print('Finished training Gaussian HMM:', hmm.score(X, lengths=L))
    return hmm

def train_hmm_for_all_users(dl, wv, k=10, cutoff=1000):
    hmm_dict = {}
    for uid in dl.get_user_ids():
        tids = dl.get_tweets_by_user(uid)
        if len(tids) >= cutoff:
            print('Training HMM for {}'.format(dl.uid_to_uname(uid)))
            X, L = make_emb_seq_for_user(uid, dl, wv)
            print('X-shape:', X.shape)
            hmm = train_hmm(X, L, k)
            hmm_dict[uid] = hmm
    pickle.dump(hmm_dict, open('./models/hmm_dict.pkl'.format(k), 'wb'))

def make_vecs_from_hmm(dl, wv, hmm_dict):
    uids = sorted(hmm_dict.keys())
    num_uids = len(uids)
    print('Num users:', num_uids)
    vecs = {}
    for ui in uids:
        vec = np.zeros(num_uids)
        print('Making vec for {}'.format(dl.uid_to_uname(ui)))
        X, L = make_emb_seq_for_user(ui, dl, wv)
        for j, uj in enumerate(uids):
            hmmj = hmm_dict[uj]
            vec[j] = hmmj.score(X, lengths=L)
        vec = np.divide(vec, X.shape[0])
        vecs[ui] = vec
    return vecs

def vecs_viz(dl, user_vecs):
    labels, vecs = [], []
    for uid in user_vecs:
        labels.append(dl.uid_to_uname(uid))
        vecs.append(user_vecs[uid])
    vecs_compressed = PCA(n_components=2).fit_transform(vecs)
    x = [v[0] for v in vecs_compressed]
    y = [v[1] for v in vecs_compressed]
    plt.scatter(x, y)
    for i in range(len(x)):
        plt.annotate(labels[i], (x[i], y[i]))
    plt.show()

def eval_vecs(dl, user_vecs):
    uids = sorted(user_vecs.keys())
    for i, uid in enumerate(uids):
        vec = user_vecs[uid]
        most_sim_idxs = np.argsort(np.multiply(vec, -1))[:5]
        most_sim_unames = ', '.join([dl.uid_to_uname(uids[idx]) for idx in most_sim_idxs])
        print('{}: most similar to {}'.format(dl.uid_to_uname(uid), most_sim_unames))

def cluster_user_vecs(user_vecs, train_uids, k0, kn, viz=True):
    X_train, X_test = [],[]
    for uid in user_vecs:
        vec = user_vecs[uid]
        if uid in train_uids:
            X_train.append(vec)
        else:
            X_test.append(vec)

    X_train, X_test = np.array(X_train), np.array(X_test)
    train_bics, test_bics = [], []
    for k in range(k0, kn+1):
        mix = train_mixture(X_train, k=k)
        train_bics.append(mix.bic(X_train))
        test_bics.append(mix.bic(X_test))
        print('Train BIC: {}. Test BIC: {}.'.format(round(train_bics[-1], 4), round(test_bics[-1], 4)))

    if viz:
        X = range(k0, kn+1)
        plt.plot(X, train_bics, 'b', label='train')
        plt.plot(X, test_bics, 'r', label='test')
        plt.title('BIC Over Number of Components')
        plt.ylabel('BIC (Bayesian Information Criterion)')
        plt.xlabel('Number of Components')
        plt.legend()
        plt.show()

    return train_bics, test_bics

def eval_hmm_transmat(hmm):
    mat = hmm.transmat_
    k = mat.shape[0]
    for si in range(k):
        for sj in range(k):
            print('p({}|{}) = {}'.format(si, sj, mat[sj][si]))
        print()

def eval_hmm_samples(hmm, wv, num_samples=10000, num_show=25):
    state2dict = {}
    k = hmm.transmat_.shape[0]
    print('k =', k)
    for state in range(k):
        state2dict[state] = {}
    X_sample, state_sample = hmm.sample(n_samples=num_samples)
    print('Generated {} samples'.format(num_samples))

    for i in range(num_samples):
        vec = X_sample[i]
        state = state_sample[i]
        state_dict = state2dict[state]
        closest = wv.similar_by_vector(vec)[0][0]  # first tuple, first element
        if closest in state_dict:
            state_dict[closest] += 1
        else:
            state_dict[closest] = 1

    for state in range(k):
        print('STATE {}'.format(state))
        state_dict = state2dict[state]
        word_freq = sorted(state_dict.items(), key=lambda x:x[1], reverse=True)
        if len(word_freq) > num_show:
            word_freq = word_freq[:num_show]
        for word, freq in word_freq:
            print(word, freq)
        print()

def adjust_user_vecs(user_vecs, test_uids):
    uids = sorted(list(user_vecs.keys()))
    num_users = len(uids)
    to_keep = []
    for i, uid in enumerate(uids):
        if uid not in test_uids:
            to_keep.append(i)
    assert(len(to_keep) == num_users-len(test_uids))
    adj_user_vecs = {}
    for uid in user_vecs:
        if uid in test_uids:
            continue
        else:
            adj_user_vecs[uid] = user_vecs[uid][to_keep]
    return adj_user_vecs

if __name__ == '__main__':
    dl = DataLoader()
    wv = KeyedVectors.load_word2vec_format('./models/word_embs_d300.bin', binary=True)
    # mix = pickle.load(open('./models/tweet_gmm_k10.pkl', 'rb'))
    # fit_mc_for_all_users(dl, wv, mix)

    hmm_dict = pickle.load(open('./models/hmm_dict.pkl', 'rb'))
    hmm_vecs = make_vecs_from_hmm(dl, wv, hmm_dict)
    pickle.dump(hmm_vecs, open('./saved/user_vecs_hmm.pkl', 'wb'))

    # dl = DataLoader(load_tweets=False)
    # hmm_vecs = pickle.load(open('./saved/hmm_vecs.pkl', 'rb'))
    # # hmm_vecs_viz(dl, hmm_vecs)
    # # eval_hmm_vecs(dl, hmm_vecs)
    # train_uids = pickle.load(open('./saved/train_uids.pkl', 'rb'))
    # train_bics, test_bics = cluster_hmm_vecs(hmm_vecs, train_uids, k0=1, kn=30)

    # user_vecs = pickle.load(open('./saved/user_vecs_mc.pkl', 'rb'))
    # test_uids = pickle.load(open('./saved/test_uids.pkl', 'rb'))
    # adjust_user_vecs(user_vecs, test_uids)

