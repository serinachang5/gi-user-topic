from csv import DictReader
from collections import Counter
from data_loader import DataLoader
from gensim.models import KeyedVectors
import numpy as np
import pickle
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_fscore_support
import utils as ut
import matplotlib.pyplot as plt
from markov_preprocessing import sort_tids_by_timestamp, find_continuous_intervals

def parse_labeled_tids(filename, dl):
    valid_labels = {'Aggression', 'Loss', 'Other'}
    tid2label = {}
    with open(filename) as f:
        reader = DictReader(f)
        dupe_count = 0
        conflict_count = 0
        blacklist = set()
        for row in reader:
            tid = row['tweet_id']
            if dl.has_tweet_id(tid) and tid not in blacklist and row['label'] in valid_labels:
                if tid in tid2label:  # already seen this before
                    stored = tid2label[tid]
                    if stored == row['label']:  # agreement, do nothing
                        dupe_count += 1
                    else:  # conflict --> remove this
                        conflict_count += 1
                        del tid2label[tid]
                        blacklist.add(tid)
                else:
                    tid2label[tid] = row['label']
        print('Clean unique labels: {}. Dupes: {}. Conflicts: {}.'.format(len(tid2label), dupe_count, conflict_count))
    return tid2label

def test_correlations(mdl, tid2label, dl, wv):
    label2idx = {'Aggression':0, 'Loss':1, 'Other':2}
    idx2label = ['Aggression', 'Loss', 'Other']
    num_labels = len(label2idx)
    num_topics = len(mdl.weights_)
    label_mat = []
    topic_mat = []
    for tid in tid2label:
        emb = ut.make_emb_for_tweet(wv, dl.get_tweet(tid)['tokens'])
        if emb is not None:
            label_bow = np.zeros(num_labels, dtype=int)
            label = tid2label[tid]
            label_bow[label2idx[label]] = 1
            label_mat.append(label_bow)

            topic_bow = np.zeros(num_topics, dtype=int)
            pred = mdl.predict([emb])[0]
            topic_bow[pred] = 1
            topic_mat.append(topic_bow)
    label_mat = np.array(label_mat).T  # label by sample
    topic_mat = np.array(topic_mat).T  # topic by sample

    R = np.zeros((num_labels, num_topics))
    P = np.zeros((num_labels, num_topics))
    for li in range(label_mat.shape[0]):
        for tj in range(topic_mat.shape[0]):
            label_row = label_mat[li]
            topic_row  = topic_mat[tj]
            r, p = pearsonr(label_row, topic_row)
            R[li][tj] = r
            P[li][tj] = p
            print('Label: {} - Topic: {} --> correlation = {}, p = {}'.format(idx2label[li], tj, round(r, 4), round(p, 4)))
    return R, P

def predictive_performance(topic_as_feat, label_to_pred, mdl, tid2label, dl, wv):
    true = []
    pred = []
    for tid in tid2label:
        emb = ut.make_emb_for_tweet(wv, dl.get_tweet(tid)['tokens'])
        if emb is not None:
            label = tid2label[tid]
            if label == label_to_pred:
                true.append(1)
            else:
                true.append(0)
            if mdl.predict([emb])[0] == topic_as_feat:
                pred.append(1)
            else:
                pred.append(0)
    print(precision_recall_fscore_support(true, pred))

def compare_to_model(mdl, tid2label, dl, wv):
    adict, acount = {}, 0
    ldict, lcount = {}, 0
    odict, ocount = {}, 0
    for tid in tid2label:
        emb = ut.make_emb_for_tweet(wv, dl.get_tweet(tid)['tokens'])
        if emb is not None:
            label = tid2label[tid]
            if label == 'Aggression':
                label_dict = adict
                acount += 1
            elif label == 'Loss':
                label_dict = ldict
                lcount += 1
            else:
                label_dict = odict
                ocount += 1
            pred = mdl.predict([emb])[0]
            if pred in label_dict:
                label_dict[pred] += 1
            else:
                label_dict[pred] = 1
    atuples = [(c, adict[c], round(adict[c]/acount, 4)) for c in adict]
    print('AGGRESSION')
    print(sorted(atuples, key=lambda x:x[1], reverse=True))

    ltuples = [(c, ldict[c], round(ldict[c]/lcount, 4)) for c in ldict]
    print('LOSS')
    print(sorted(ltuples, key=lambda x:x[1], reverse=True))

    otuples = [(c, odict[c], round(odict[c]/ocount, 4)) for c in odict]
    print('OTHER')
    print(sorted(otuples, key=lambda x:x[1], reverse=True))

def make_relation_mat(dl):
    uids = sorted(dl.get_user_ids())
    num_users = max(uids)+1
    mat = np.zeros((num_users, num_users), dtype=int)
    for ui in uids:
        print(ui, dl.uid_to_uname(ui))
        out_edge_vec = np.zeros(num_users)  # from ui to uj
        for tid in dl.get_tweets_by_user(ui):
            tweet = dl.get_tweet(tid)
            rt = tweet['retweet_id']
            if rt > 0:
                out_edge_vec[rt] += 1
            mentions = tweet['mention_ids']
            if len(mentions) > 0:
                for m in mentions:
                    out_edge_vec[m] += 1
        print(np.sum(out_edge_vec))
        mat[ui] = out_edge_vec
    print(np.sum(mat))
    return mat

def find_pairs_above_threshold(rel_mat, t, bidir=True):
    num_users = rel_mat.shape[0]
    pairs = set()
    if bidir:
        for i in range(num_users):
            for j in range(i+1, num_users):
                weight = rel_mat[i][j] + rel_mat[j][i]
                if weight >= t:
                    pairs.add((i,j))
    else:
        for i in range(num_users):
            for j in range(num_users):
                weight = rel_mat[i][j]
                if weight >= t:
                    pairs.add((i,j))
    return pairs

def pairs_over_threshold_viz(rel_mat):
    print('Num users:', rel_mat.shape[0])
    T = range(1,21)
    pair_counts = []
    for t in T:
        pairs = find_pairs_above_threshold(rel_mat, t=t, bidir=True)
        pair_counts.append(len(pairs))
        print('t={}, bidirectional --> {} pairs'.format(t, len(pairs)))
    plt.plot(T, pair_counts)
    plt.title('Number of user pairs surpassing threshold number of interactions')
    plt.xlabel('Threshold number of interactions')
    plt.ylabel('Number of user pairs')
    plt.xticks(T)
    plt.yticks([1000*i for i in range(9)])
    plt.show()

def eval_pairs(user_vecs, rel_mat, t, dist_func):
    mean, std, max, min = ut.get_stats(user_vecs, dist_func)
    pairs = find_pairs_above_threshold(rel_mat, t)
    num_pairs_found = 0
    dist_sum = 0
    for ui, uj in pairs:
        dist = ut.get_dist(user_vecs, ui, uj, dist_func)
        if dist is not None:
            num_pairs_found += 1
            norm_dist = np.divide(np.subtract(dist, mean), std)
            dist_sum += norm_dist
    return dist_sum/num_pairs_found

def compare_vecs_across_t(uv_dict, t_range, rel_mat, viz=True):
    score_dict = dict([(uv_type, []) for uv_type in uv_dict])
    for t in t_range:
        for uv_type in uv_dict:
            if uv_type == 'bow':
                dist_func = 'btc'
            elif uv_type == 'emb':
                dist_func = 'cos'
            elif uv_type == 'topic':
                dist_func = 'btc'
            elif uv_type == 'mc':
                dist_func = 'cos'
            else:  # hmm
                assert(uv_type == 'hmm')
                dist_func = 'cos'
            uv = uv_dict[uv_type]
            score = eval_pairs(uv, rel_mat, t, dist_func)
            score_dict[uv_type].append(score)
        report = []
        for uv_type in score_dict:
            scores = score_dict[uv_type]
            report.append('{}: {}'.format(uv_type, round(scores[-1], 4)))
        print('t={}. {}.'.format(t, '. '.join(report)))

    if viz:
        for uv_type in score_dict:
            scores = score_dict[uv_type]
            if uv_type == 'bow':
                color = 'black'
            elif uv_type == 'emb':
                color = 'blue'
            elif uv_type == 'topic':
                color = 'green'
            elif uv_type == 'mc':
                color = 'grey'
            else:  # hmm
                assert(uv_type == 'hmm')
                color = 'red'
            plt.plot(t_range, scores, color, label=uv_type.upper())
        plt.xlabel('Threshold')
        plt.ylabel('Normalized distance score')
        plt.legend()
        plt.show()

def find_interesting_seqs_in_labeled(dl, wv, tid2label, cutoff=5):
    users2labeled = {}
    for tid in tid2label:
        label = tid2label[tid]
        tweet = dl.get_tweet(tid)
        uid = tweet['user_id']
        users2labeled.setdefault(uid, []).append((tid, label))
    user2seqs = {}
    for uid in users2labeled:
        tids = [x[0] for x in users2labeled[uid]]
        sorted_time_tid_tups = sort_tids_by_timestamp(tids, dl)
        valid_tids = []
        dates = []
        for date, time, tid in sorted_time_tid_tups:
            emb = ut.make_emb_for_tweet(wv, dl.get_tweet(tid)['tokens'])
            if emb is not None:
                valid_tids.append(tid)
                dates.append(date)
        ints = find_continuous_intervals(dates)
        for i,j in ints:
            tids_in_interval = valid_tids[i:j+1]
            labels = [tid2label[tid] for tid in tids_in_interval]
            freq_dict = Counter(labels)
            if 'Aggression' in freq_dict and 'Loss' in freq_dict:
                if freq_dict['Aggression'] + freq_dict['Loss'] >= cutoff:
                    print('Found seq by {} with {} Aggression and {} Loss'.format(dl.uid_to_uname(uid), freq_dict['Aggression'], freq_dict['Loss']))
                    user2seqs.setdefault(uid, []).append(tids_in_interval)
    return user2seqs

if __name__ == '__main__':
    # dl = DataLoader()
    # wv = KeyedVectors.load_word2vec_format('./models/word_embs_d300.bin', binary=True)
    # tid2label = parse_labeled_tids('./raw_data/labeled_2018_03_21.csv', dl)
    # user2seqs = find_interesting_seqs_in_labeled(dl, wv, tid2label, cutoff=10)
    #
    # for uid in user2seqs:
    #     print('USERNAME:', dl.uid_to_uname(uid))
    #     for i, seq in enumerate(user2seqs[uid]):
    #         print('SEQUENCE,', i)
    #         for tid in seq:
    #             print('LABEL: {}. TEXT: {}.'.format(tid2label[tid], ' '.join(dl.get_tweet(tid)['tokens'])))

    #
    # train_uids = pickle.load(open('./saved/train_uids.pkl', 'rb'))
    # test_uids = pickle.load(open('./saved/test_uids.pkl', 'rb'))
    # valid_uids = train_uids + test_uids
    # print(len(valid_uids))

    rel_mat = pickle.load(open('./saved/user_relation_mat.pkl', 'rb'))
    # uv_bow = pickle.load(open('./saved/user_vecs_bow.pkl', 'rb'))
    uv_emb = pickle.load(open('./saved/user_vecs_emb.pkl', 'rb'))
    uv_topic = pickle.load(open('./saved/user_vecs_topics.pkl', 'rb'))
    uv_hmm = pickle.load(open('./saved/user_vecs_hmm.pkl', 'rb'))
    uv_mc = pickle.load(open('./saved/user_vecs_mc.pkl', 'rb'))

    t_range = [i * 10 for i in range(0,6)]
    uv_dict = {'emb':uv_emb, 'topic':uv_topic, 'hmm':uv_hmm}
    compare_vecs_across_t(uv_dict, t_range, rel_mat, viz=True)

