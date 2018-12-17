from data_loader import DataLoader
from gensim.models import KeyedVectors
import numpy as np
import pickle
import random
from utils import make_emb_for_tweet, make_tweet_mat_emb, get_topics_for_tweets

def extract_timestamp(created_at):
    if created_at.startswith('20'):
        date, time = created_at.split()
        return date, time
    pieces = created_at.split()
    month_name, day, year = pieces[1].lower(), pieces[2], pieces[-1]
    if month_name == 'jan':
        month = '01'
    elif month_name == 'feb':
        month = '02'
    elif month_name == 'mar':
        month = '03'
    elif month_name == 'apr':
        month = '04'
    elif month_name == 'may':
        month = '05'
    elif month_name == 'jun':
        month = '06'
    elif month_name == 'jul':
        month = '07'
    elif month_name == 'aug':
        month = '08'
    elif month_name == 'sep':
        month = '09'
    elif month_name == 'oct':
        month = '10'
    elif month_name == 'nov':
        month = '11'
    else:
        assert(month_name == 'dec')
        month = '12'
    date = '{}-{}-{}'.format(year, month, day)
    time = pieces[3]
    return date, time

def sort_tids_by_timestamp(tids, dl):
    time_tid_tuples = []
    for tid in tids:
        date, time = extract_timestamp(dl.get_tweet(tid)['created_at'])
        time_tid_tuples.append((date, time, tid))
    sorted_tids = sorted(time_tid_tuples, key=lambda x: (x[0], x[1]))
    return sorted_tids

def next_day(date):
    y0, m0, d0 = [int(tok) for tok in date.split('-')]
    y1, m1, d1 = y0, m0, d0
    if d0 == 28 and m0 == 4 and y0 % 4 != 0:  # 4/28/yy to 5/1/yy in non-leap year
        d1 = 1
        m1 += 1
    elif d0 == 29 and m0 == 4:  # 4/29/yy to 5/1/yy in leap year
        d1 = 1
        m1 += 1
    elif d0 == 30 and m0 in {2, 4, 6, 9, 11}:  # -/30/yy to -/1/yy
        d1 = 1
        m1 += 1
    elif d0 == 31 and m0 in {1, 3, 5, 7, 8, 10}:  # -/31/yy to -/1/yy
        d1 = 1
        m1 += 1
    elif d0 == 31 and m0 == 12:  # new year's
        d1 = 1
        m1 = 1
        y1 += 1
    else:  # just a normal increment
        d1 += 1
    y1, m1, d1 = str(y1), str(m1), str(d1)
    if len(m1) == 1:
        m1 = '0' + m1
    if len(d1) == 1:
        d1 = '0' + d1
    if len(y1) == 2:
        y1 = '20' + y1
    return '{}-{}-{}'.format(y1, m1, d1)

def find_continuous_intervals(sorted_dates):
    intervals = []
    i = 0
    while i < len(sorted_dates):
        j = i
        while j < len(sorted_dates)-1 and \
                        sorted_dates[j+1] in {sorted_dates[j], next_day(sorted_dates[j])}:  # either same day or next day
            j += 1
        intervals.append((i, j))
        i = j+1
    return intervals

def make_emb_seq_for_user(uid, dl, wv):
    tids = dl.get_tweets_by_user(uid)
    sorted_time_tid_tups = sort_tids_by_timestamp(tids, dl)
    embs = []
    dates = []
    for date, time, tid in sorted_time_tid_tups:
        emb = make_emb_for_tweet(wv, dl.get_tweet(tid)['tokens'])
        if emb is not None:  # there is at least one word vec for this tweet's tokens
            embs.append(emb)
            dates.append(date)
    intervals = find_continuous_intervals(dates)
    lengths = [i[1]-i[0]+1 for i in intervals]
    return np.array(embs), lengths

def get_all_emb_seqs(uids, dl, wv):
    all_embs = []
    all_lengths = []
    for uid in uids:
        embs, lengths = make_emb_seq_for_user(uid, dl, wv)
        all_embs.append(embs)
        all_lengths.append(lengths)
    X = np.concatenate(all_embs, axis=0)
    L = np.concatenate(all_lengths, axis=0)
    return X, L

def make_topic_seq_for_user(uid, dl, wv, mix):
    tids = dl.get_tweets_by_user(uid)
    sorted_time_tid_tups = sort_tids_by_timestamp(tids, dl)
    dates = []
    for date, time, tid in sorted_time_tid_tups:
        emb = make_emb_for_tweet(wv, dl.get_tweet(tid)['tokens'])
        if emb is not None:  # there is at least one word vec for this tweet's tokens
            dates.append(date)
    topics = get_topics_for_tweets(dl, wv, mix, tids)
    intervals = find_continuous_intervals(dates)
    lengths = [i[1]-i[0]+1 for i in intervals]
    return np.array(topics), lengths

def get_all_topic_seqs(uids, dl, wv, mix):
    all_topics = []
    all_lengths = []
    for uid in uids:
        topics, lengths = make_topic_seq_for_user(uid, dl, wv, mix)
        all_topics.append(topics)
        all_lengths.append(lengths)
    X = np.concatenate(all_topics, axis=0)
    L = np.concatenate(all_lengths, axis=0)
    return X, L

def make_train_test(dl, cutoff=1000, train_prop=.7):
    all_users = list(dl.get_user_ids())
    valid_users = []
    for uid in all_users:
        if len(dl.get_tweets_by_user(uid)) >= cutoff:
            valid_users.append(uid)
    np.random.shuffle(valid_users)
    train_cutoff = int(len(valid_users) * train_prop)
    return valid_users[:train_cutoff], valid_users[train_cutoff:]

if __name__ == '__main__':
    # wv = KeyedVectors.load_word2vec_format('./models/word_embs_d300.bin', binary=True)
    dl = DataLoader()
    # mix = pickle.load(open('./models/tweet_gmm_k10.pkl', 'rb'))
    # users = range(100,105)
    # X, lengths = get_all_topic_sequences(users, dl, wv, mix)
    # print('Num sequences:', len(lengths))
    # print(X.shape)
    # print(X[0])
    train_uids, test_uids = make_train_test(dl)
    print(len(train_uids), len(test_uids))
    pickle.dump(train_uids, open('./saved/train_uids.pkl', 'wb'))
    pickle.dump(test_uids, open('./saved/test_uids.pkl', 'wb'))



