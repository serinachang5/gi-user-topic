from data_loader import DataLoader
from gensim.models import Word2Vec, KeyedVectors
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def make_sentences(tweet_dict):
    sentences = []
    for tid in tweet_dict:
        sentences.append(tweet_dict[tid]['tokens'])
    print('Num sentences:', len(sentences))
    print('Check sentence0:', sentences[0])
    return sentences

def train_w2v_embs(sentences, dim=300, win=5, min_count=5, epochs=20):
    print('Training Word2Vec...')
    model = Word2Vec(sentences, size=dim, window=win,
                     min_count=min_count, iter=epochs)
    wv = model.wv
    print('Finished. Vocab size:', len(wv.vocab))
    save_file = './models/word_embs_d{}.bin'.format(str(dim))
    wv.save_word2vec_format(save_file, binary=True)
    print('Word2Vec vectors saved to', save_file)

def eval_embs(wv):
    print('rip vs miss', wv.similarity('rip', 'miss'))
    print('rip vs bitch', wv.similarity('rip', 'bitch'))
    print('rip vs the', wv.similarity('rip', 'the'))

def viz_groups(wv, word2idx, loss=None, agg=None, sub=None, neut=None, word2emb=None):
    if word2emb is None:
        vocab = sorted(list(wv.vocab.keys()))
        all_wv = []
        for w in vocab:
            all_wv.append(wv[w])
        all_wv = np.array(all_wv)
        print('Transforming with PCA...')
        compressed = PCA(n_components=2).fit_transform(all_wv)
        print('Done. Matching tok to 2D embedding...')
        word2emb = {}
        for i, tok in enumerate(vocab):
            word2emb[tok] = compressed[i]

    x, y, colors, labels = [], [], [], []

    if loss is not None:
        print('Matching Loss...')
        loss_x, loss_y, loss_l = find_embs(loss, word2idx, word2emb)
        x += loss_x
        y += loss_y
        labels += loss_l
        colors += ['blue'] * len(loss_x)

    if agg is not None:
        print('Matching Agg...')
        agg_x, agg_y, agg_l = find_embs(agg, word2idx, word2emb)
        x += agg_x
        y += agg_y
        labels += agg_l
        colors += ['red'] * len(agg_x)

    if sub is not None:
        print('Matching Sub...')
        sub_x, sub_y, sub_l = find_embs(sub, word2idx, word2emb)
        x += sub_x
        y += sub_y
        labels += sub_l
        colors += ['green'] * len(sub_x)

    if neut is not None:
        print('Matching Neut...')
        neut_x, neut_y, neut_l = find_embs(neut, word2idx, word2emb)
        x += neut_x
        y += neut_y
        labels += neut_l
        colors += ['grey'] * len(neut_x)

    plt.scatter(x, y, c=colors)
    for i, lbl in enumerate(labels):
        plt.annotate(lbl, (x[i], y[i]), )
    plt.show()

def find_embs(words, word2idx, word2emb, lim=10):
    x = []
    y = []
    labels = []
    tuples = []
    for word in words:
        word = word.lower()
        if word in word2emb:
            tuples.append((word2idx[word], word, word2emb[word]))
    tuples = sorted(tuples, key=lambda x:x[0])
    tuples = tuples[:lim]
    for idx, word, emb in tuples:
        x.append(emb[0])
        y.append(emb[1])
        labels.append(word)
    return x, y, labels

if __name__ == '__main__':
    # dl = DataLoader(vocab_size=None)
    # sentences = make_sentences(dl.tweet_dict)
    # train_w2v_embs(sentences, dim=300, min_count=5)

    wv = KeyedVectors.load_word2vec_format('./models/word_embs_d300.bin', binary=True)
    print('Num vectors:', len(wv.vocab))
    eval_embs(wv)
    loss, agg, sub = pickle.load(open('./saved/seeds_hc.pkl', 'rb'))
    neut = ['rt', 'URL', '@USER', 'to', 'a', 'the']
    dl = DataLoader(load_tweets=False, verbose=False)
    viz_groups(wv, dl.word2idx, loss=loss, agg=agg, neut=neut)
