from collections import Counter
from csv import DictReader
from emoji import UNICODE_EMOJI
import re
import pickle

PUNCTUATION = {'.',',','!','?',';',':','\'','\"'}
PATTERN = re.compile(r"(.)\1{2,}")  # recognize repetition of 3 or more of the same character

def process_tweets(csv_filenames, debug_lim=None):
    tweet_dict = {}  # tweet id to tweet properties
    user_dict = {}  # user id to user properties
    uname2tids = {}  # temp dictionary of username to tweet_ids
    all_toks = []  # temp list of all tokens

    # first pass - store everything besides indices (which rely on global info)
    dupe_count = 0
    total_count = 0
    for filename in csv_filenames:
        print('Processing {}...'.format(filename))
        with open(filename) as f:
            reader = DictReader(f)
            for row in reader:
                tid = row['tweet_id']
                if tid not in tweet_dict:
                    toks, rt, mentions = preprocess(row['text'])
                    all_toks += toks
                    uname = row['user_name'].lower()
                    tweet_dict[tid] = {'tweet_id':row['tweet_id'], 'created_at':row['created_at'],
                                       'tokens':toks, 'indices':[],
                                       'user_name:':uname, 'user_id':0,
                                       'retweet':rt, 'retweet_id':0,
                                       'mentions':mentions, 'mention_ids':[]}
                    if uname not in uname2tids:
                        uname2tids[uname] = []
                    uname2tids[uname].append(tid)
                else:
                    dupe_count += 1
                total_count += 1
                if total_count % 100000 == 0: print(total_count)
                if debug_lim is not None and total_count > debug_lim:
                    break
    print('Finished first pass. Parsed {} rows -> {} dupes, {} unique.'.format(total_count, dupe_count, len(tweet_dict)))

    ct = Counter(all_toks)
    word_freq = [(tok, ct[tok]) for tok in ct]
    sorted_word_freq = sorted(word_freq, key=lambda x:x[1], reverse=True)
    idx2word = ['<PAD>', '<UNK>'] + [t[0] for t in sorted_word_freq]
    word2idx = dict((idx2word[idx], idx) for idx in range(len(idx2word)))

    user_freq = [(uname, len(uname2tids[uname])) for uname in uname2tids]
    sorted_user_freq = sorted(user_freq, key=lambda x:x[1], reverse=True)
    idx2user = ['<PAD>', '<UNK>'] + [t[0] for t in sorted_user_freq]
    user2idx = dict((idx2user[idx], idx) for idx in range(len(idx2user)))

    # second pass - fill in indices
    for uname in uname2tids:
        uidx = user2idx[uname]
        tids = uname2tids[uname]
        user_dict[uidx] = {'user_id':uidx, 'user_name':uname, 'tweet_ids':tids}
        for tid in tids:
            tprops = tweet_dict[tid]
            tprops['indices'] = [word2idx[tok] for tok in tprops['tokens']]
            tprops['user_id'] = uidx
            if tprops['retweet'] is not None:
                ruser = tprops['retweet']
                if ruser in user2idx:
                    tprops['retweet_id'] = user2idx[ruser]
                else:
                    tprops['retweet_id'] = 1
            for muser in tprops['mentions']:
                if muser in user2idx:
                    tprops['mention_ids'].append(user2idx[muser])
                else:
                    tprops['mention_ids'].append(1)
    return tweet_dict, user_dict, idx2word, word2idx, idx2user, user2idx

def preprocess(s):
    s = s.lower()
    rt = None
    mentions = []
    orig_toks = s.split()
    toks = []
    for i, t in enumerate(orig_toks):
        if t.startswith('@'):
            name = t[1:]
            if name.endswith(':'):
                name = name[:-1]
            mentions.append(name)
            if i > 0 and orig_toks[i-1] == 'rt':
                rt = name
            toks.append('@USER')
        elif t.startswith('http'):
            toks.append('URL')
        else:
            t = ''.join([' ' + c + ' ' if c in UNICODE_EMOJI or c in PUNCTUATION else c for c in t])  # add spaces for emojis and punctuations
            t = PATTERN.sub(r"\1\1\1", t)
            toks.extend(t.split())
    return toks, rt, mentions

if __name__ == '__main__':
    processed = process_tweets(['./raw_data/gnip_data.csv', './raw_data/oct26.csv'])
    tweet_dict, user_dict, idx2word, word2idx, idx2user, user2idx = processed
    print('Vocab length:', len(idx2word))
    print('Num tweets:', len(tweet_dict))

    pickle.dump(tweet_dict, open('./data/tweets.pkl', 'wb'))
    print('Num users:', len(user_dict))
    pickle.dump(user_dict, open('./data/users.pkl', 'wb'))

    print('Top 10 Words')
    for i in range(10):
        word = idx2word[i]
        print(i, word, word2idx[word])
    pickle.dump(idx2word, open('./data/idx2word.pkl', 'wb'))
    pickle.dump(word2idx, open('./data/word2idx.pkl', 'wb'))

    print('Top 10 Users')
    for j in range(10):
        user = idx2user[j]
        print(j, user, user2idx[user])
    pickle.dump(idx2user, open('./data/idx2user.pkl', 'wb'))
    pickle.dump(user2idx, open('./data/user2idx.pkl', 'wb'))