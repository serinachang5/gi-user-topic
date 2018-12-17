from data_loader import DataLoader
from gensim.models import KeyedVectors
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from utils import make_tweet_mat_emb

def train_mixture(X, k):
    mixture = GaussianMixture(n_components=k)
    print('Training {}-component Gaussian Mixture...'.format(k))
    mixture.fit(X)
    return mixture

def train_wv_mixture(tweet_ids, dl, wv, k):
    X = []
    for tid in tweet_ids:
        toks = dl.get_tweet(tid)['tokens']
        for tok in toks:
            if tok in wv:
                X.append(wv[tok])
    X = np.array(X)
    print(X.shape)
    mix = train_mixture(X, k=k)
    pickle.dump(mix, open('./models/wv_gmm_k{}.pkl'.format(str(k)), 'wb'))

def train_tweet_mixture(tweet_ids, dl, wv, k):
    X = make_tweet_mat_emb(dl, wv, tweet_ids)
    print(X.shape)
    mix = train_mixture(X, k=k)
    pickle.dump(mix, open('./models/tweet_gmm_k{}.pkl'.format(str(k)), 'wb'))

def train_user_mixture(uv_type, color, viz=True):
    k_range = range(1,11)
    bic = []
    user_vecs = pickle.load(open('./saved/user_vecs_{}.pkl'.format(uv_type), 'rb'))
    X = np.array([user_vecs[uid] for uid in user_vecs])
    for k in k_range:
        mix = train_mixture(X, k=k)
        bic.append(mix.bic(X))
    if viz:
        plt.plot(k_range, bic, color=color)
        plt.xlabel('Number of Components')
        plt.ylabel('BIC')
        plt.title('BIC Curve for {} User Representations'.format(uv_type.upper()))
        plt.xticks(k_range)
        plt.show()
    else:
        for i, k in enumerate(k_range):
            print(k, bic[i])

def eval_mixture_means(mix, wv):
    weights = np.array(mix.weights_)
    topics_in_order = list(np.argsort(weights*-1))
    for topic in topics_in_order:
        print('TOPIC {}. Weight={}'.format(topic, weights[topic]))
        mean = mix.means_[topic]
        tuples = wv.similar_by_vector(mean, topn=10)
        for word, prob in tuples:
            print(word, prob)
        print()

def eval_mixture_samples(mix, wv, num_samples=50000, num_show=25):
    topic_to_dict = {}
    num_topics = len(mix.weights_)
    for topic in range(num_topics):
        topic_to_dict[topic] = {}
    X_sample, y_sample = mix.sample(n_samples=num_samples)
    print('Generated {} samples'.format(num_samples))

    for i in range(num_samples):
        vec = X_sample[i]
        topic = y_sample[i]
        topic_dict = topic_to_dict[topic]
        closest = wv.similar_by_vector(vec)[0][0]  # first tuple, first element
        if closest in topic_dict:
            topic_dict[closest] += 1
        else:
            topic_dict[closest] = 1

    weights = np.array(mix.weights_)
    topics_in_order = list(np.argsort(weights*-1))
    for topic in topics_in_order:
        print('TOPIC {}. Weight={}'.format(topic, weights[topic]))
        topic_dict = topic_to_dict[topic]
        word_freq = sorted(topic_dict.items(), key=lambda x:x[1], reverse=True)
        if len(word_freq) > num_show:
            word_freq = word_freq[:num_show]
        for word, freq in word_freq:
            print(word, freq)
        print()

def eval_mixture_bic(mix, tweet_ids, dl, wv):
    X = make_tweet_mat_emb(dl, wv, tweet_ids)
    print('Sample shape:', X.shape)
    k = len(mix.means_)
    print('Mixture parameters: k =', k)
    bic = mix.bic(X)
    print('BIC:', round(bic,4))
    return bic

def eval_mixture_on_data(mix, tweet_ids, dl, wv, top_n=10):
    X = make_tweet_mat_emb(dl, wv, tweet_ids)
    print('Sample shape:', X.shape)
    k = len(mix.means_)
    weights = mix.weights_
    print('Mixture parameters: k =', k)
    print('BIC:', round(mix.bic(X),4))

    top_n_dict = dict([(c, []) for c in range(k)])
    topic_by_prob = mix.predict_proba(X).T  # transpose, so one row per topic
    print(topic_by_prob.shape)  # should be k x n
    for c, topic_probs in enumerate(topic_by_prob):
        top_idx = list(np.argsort(np.multiply(topic_probs, -1)))[:top_n]  # sort in descending order
        for idx in top_idx:
            tid = tweet_ids[idx]
            top_n_dict[c].append((topic_probs[idx], dl.get_tweet(tid)['tokens']))

    topics_in_order = list(np.argsort(np.multiply(weights, -1)))
    for c in topics_in_order:
        print('TOPIC {}. Weight={}'.format(c, weights[c]))
        top_tweets = top_n_dict[c]
        for i, (prob, toks) in enumerate(top_tweets):
            print('{}. p={}, toks: {}'.format(i+1, round(prob, 4), toks))
        print()

def train_test_viz(dl, wv, k0 = 5, kn = 10):
    train_bics = []
    test_bics = []
    train_tids = pickle.load(open('./saved/train_tids.pkl', 'rb'))
    test_tids = pickle.load(open('./saved/test_tids.pkl', 'rb'))
    X = range(k0, kn+1)
    for k in X:
        file_name = './models/tweet_gmm_k{}.pkl'.format(k)
        mix = pickle.load(open(file_name, 'rb'))
        train_bics.append(eval_mixture_bic(mix, train_tids, dl,  wv))
        test_bics.append(eval_mixture_bic(mix, test_tids, dl, wv))
        print()

    plt.plot(X, train_bics, 'b', label='train')
    plt.plot(X, test_bics, 'r', label='test')
    plt.title('BIC Over Number of Components')
    plt.ylabel('BIC (Bayesian Information Criterion)')
    plt.xlabel('Number of Components')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # dl = DataLoader()
    # wv = KeyedVectors.load_word2vec_format('./models/word_embs_d300.bin', binary=True)
    # train_tids = pickle.load(open('./saved/train_tids.pkl', 'rb'))
    # K = range(11,13)
    # for k in K:
    #     train_tweet_mixture(train_tids, dl, wv, k=k)

    # train_test_viz(dl, wv, k0=5, kn=12)
    # mix = pickle.load(open('./models/tweet_gmm_k10.pkl', 'rb'))
    # eval_mixture_samples(mix, wv, num_samples=50000, num_show=25)

    # train_user_mixture('emb', 'blue')
    uv_type = 'emb'
    user_vecs = pickle.load(open('./saved/user_vecs_{}.pkl'.format(uv_type), 'rb'))
    sorted_uids = sorted(list(user_vecs.keys()))
    X = np.array([user_vecs[uid] for uid in sorted_uids])
    mix = train_mixture(X, k=2)
    e_clusters = mix.predict(X)

    uv_type = 'topics'
    user_vecs = pickle.load(open('./saved/user_vecs_{}.pkl'.format(uv_type), 'rb'))
    sorted_uids = sorted(list(user_vecs.keys()))
    X = np.array([user_vecs[uid] for uid in sorted_uids])
    mix = train_mixture(X, k=2)
    t_clusters = mix.predict(X)

    uv_type = 'hmm'
    user_vecs = pickle.load(open('./saved/user_vecs_{}.pkl'.format(uv_type), 'rb'))
    sorted_uids = sorted(list(user_vecs.keys()))
    X = np.array([user_vecs[uid] for uid in sorted_uids])
    mix = train_mixture(X, k=2)
    h_clusters = mix.predict(X)

    print('EMB-EMB', adjusted_rand_score(e_clusters, e_clusters))
    print('EMB-TOPIC', adjusted_rand_score(e_clusters, t_clusters))
    print('EMB-HMM', adjusted_rand_score(e_clusters, h_clusters))
    print('TOPIC-HMM', adjusted_rand_score(t_clusters, h_clusters))







