#/usr/bin/python

# %%
import argparse
import torch
import pickle 
import numpy as np 
import os 
import math 
import random 
import sys
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.io

import data 

from sklearn.decomposition import PCA
from torch import nn, optim
from torch.nn import functional as F

from detm import DETM
from utils import nearest_neighbors, get_topic_coherence
import gensim

parser = argparse.ArgumentParser(description='The Embedded Topic Model')

### data and file related arguments
parser.add_argument('--dataset', type=str, default='un', help='name of corpus')
parser.add_argument('--data_path', type=str, default='un/', help='directory containing data')
parser.add_argument('--emb_path', type=str, default='skipgram/embeddings.txt', help='directory containing embeddings')
parser.add_argument('--save_path', type=str, default='./results', help='path to save results')
parser.add_argument('--batch_size', type=int, default=1000, help='number of documents in a batch for training')
parser.add_argument('--min_df', type=int, default=100, help='to get the right data..minimum document frequency')

### model-related arguments
parser.add_argument('--num_topics', type=int, default=50, help='number of topics')
parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu', help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
parser.add_argument('--train_embeddings', type=int, default=1, help='whether to fix rho or train it')
parser.add_argument('--eta_nlayers', type=int, default=3, help='number of layers for eta')
parser.add_argument('--eta_hidden_size', type=int, default=200, help='number of hidden units for rnn')
parser.add_argument('--delta', type=float, default=0.005, help='prior variance')

### optimization-related arguments
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--mode', type=str, default='train', help='train or eval model')
parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--seed', type=int, default=2019, help='random seed (default: 1)')
parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')
parser.add_argument('--eta_dropout', type=float, default=0.0, help='dropout rate on rnn for eta')
parser.add_argument('--clip', type=float, default=0.0, help='gradient clipping')
parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
parser.add_argument('--anneal_lr', type=int, default=0, help='whether to anneal the learning rate or not')
parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')

### evaluation, visualization, and logging-related arguments
parser.add_argument('--num_words', type=int, default=20, help='number of words for topic viz')
parser.add_argument('--log_interval', type=int, default=10, help='when to log training')
parser.add_argument('--visualize_every', type=int, default=1, help='when to visualize results')
parser.add_argument('--eval_batch_size', type=int, default=1000, help='input batch size for evaluation')
parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')
parser.add_argument('--tc', type=int, default=0, help='whether to compute tc or not')

parser.add_argument('--idx_sim', type=int, default=0, help='第几次模拟')
parser.add_argument('--m_set', type=int, default=100, help='number of documents')
parser.add_argument('--t_set', type=int, default=5, help='number of time slices')
parser.add_argument('--len_docs', type=int, default=50, help='length of documents')

args = parser.parse_args("--dataset acl --data_path ./data_acl_largev --emb_path ./embeddings/acl/skipgram_emb_300d.txt --min_df 10 --num_topics 30 --lr 0.0001 --epochs 1000 --mode train".split())

# args = parser.parse_args()

# %%

pca = PCA(n_components=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## set seed
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)

## get data
# 1. vocabulary
# print('Getting vocabulary ...')
# data_file = os.path.join(args.data_path, 'min_df_{}'.format(args.min_df))
# vocab, train, valid, test = data.get_data(data_file, temporal=True)
# vocab_size = len(vocab)
# args.vocab_size = vocab_size

## get data
# 1. vocabulary
# print('Getting vocabulary ...')
# data_file = os.path.join(args.data_path, 'min_df_{}'.format(args.min_df))
# vocab, train, valid, test = data.get_data(data_file, temporal=True)
# vocab_size = len(vocab)
# args.vocab_size = vocab_size

# %% load train, valid, test
dat = pickle.load(open('../ch.pkl','rb'))
dat = dat[dat['年份']!=2003]
dat.reset_index(inplace = True, drop = True)
dat = dat.drop([323, 1602])
dat.reset_index(inplace = True, drop = True)

dictionary_data = gensim.corpora.Dictionary(dat['ab_words'])
dictionary_data.filter_extremes(no_below=5)
corpus_data = [dictionary_data.doc2bow(doc) for doc in dat['ab_words']]

time_slice = dat['年份'].value_counts().sort_index().values.tolist()
size_dictionary = len(dictionary_data)
vocab_size = size_dictionary
corpus_len = len(corpus_data)
num_time_slices = len(time_slice)

# %%
num_topics = args.num_topics  # 主题数K
args.vocab_size = vocab_size

vocab = list(dictionary_data.values())

# %%
def get_dic_data(corpus_sample, time_slice):
    dic_data = {}
    times = []
    tokens = np.empty((len(corpus_sample), ), dtype=object)
    counts = np.empty((len(corpus_sample), ), dtype=object)
    for i, corpus in enumerate(corpus_sample):
        tokens_c = []
        counts_c = []
        for word, count in corpus:
            tokens_c.append(word)
            counts_c.append(count)
        tokens_c = np.expand_dims(np.array(tokens_c), axis=0)
        tokens[i] = tokens_c
        counts_c = np.expand_dims(np.array(counts_c), axis=0)
        counts[i] = counts_c
    for t, num in enumerate(time_slice):
        times.extend([t]*num)
    dic_data['tokens'] = np.array(tokens, dtype=object)
    dic_data['counts'] = np.array(counts, dtype=object)
    dic_data['times'] = np.array(times)
    return dic_data

train = get_dic_data(corpus_data, time_slice)

# %%
def get_corpus(tokens, counts):
    corpus = []
    for i in range(len(tokens)):
        corp = []
        for j in range(len(tokens[i][0])):
            corp.append((tokens[i][0][j], counts[i][0][j]))
        corpus.append(corp)
    return corpus

def get_newcorpus_theta(newcorpus, num_topics, topic_word, num_time_slices, time_slice):
    M = len(newcorpus)
    theta = np.zeros((M, num_topics))
    for t in range(num_time_slices):
        if t == 0:
            d_slice = range(np.cumsum(time_slice)[t])
        else:
            d_slice = range(np.cumsum(time_slice)[t-1],np.cumsum(time_slice)[t])
        for d in d_slice:
            theta_m = np.zeros(num_topics)
            for word, num in newcorpus[d]:
                theta_m += num * topic_word[t, :, word]
            theta_m = theta_m / np.sum(theta_m)
            theta[d] = theta_m
    return theta

# %%
# 1. training data
print('Getting training data ...')
train_tokens = train['tokens']
train_counts = train['counts']
train_times = train['times']
args.num_times = len(np.unique(train_times))
args.num_docs_train = len(train_tokens)
train_rnn_inp = data.get_rnn_input(
    train_tokens, train_counts, train_times, args.num_times, args.vocab_size, args.num_docs_train)
corpus_train = get_corpus(train_tokens, train_counts)

# %%
# 2. dev set
print('Getting validation data ...')
valid = train
valid_tokens = valid['tokens']
valid_counts = valid['counts']
valid_times = valid['times']
args.num_docs_valid = len(valid_tokens)
valid_rnn_inp = data.get_rnn_input(
    valid_tokens, valid_counts, valid_times, args.num_times, args.vocab_size, args.num_docs_valid)

# %%
## get embeddings 
print('Getting embeddings ...')
emb_path = args.emb_path
vect_path = os.path.join(args.data_path.split('/')[0], 'embeddings.pkl')   
vectors = {}
with open(emb_path, 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        if word in vocab:
            vect = np.array(line[1:]).astype("float")
            vectors[word] = vect
embeddings = np.zeros((vocab_size, args.emb_size))
words_found = 0
for i, word in enumerate(vocab):
    try: 
        embeddings[i] = vectors[word]
        words_found += 1
    except KeyError:
        embeddings[i] = np.random.normal(scale=0.6, size=(args.emb_size, ))
embeddings = torch.from_numpy(embeddings).to(device)
args.embeddings_dim = embeddings.size()

# %%
ckpt = os.path.join(args.save_path, 
        '/home/xueqiong/undergraduate/DETM-master/results/detm_ch_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.0001_Bsz_1000_RhoSize_300_L_3_minDF_10_trainEmbeddings_1')
with open(ckpt, 'rb') as f:
    model = torch.load(f)

print('\nDETM architecture: {}'.format(model))
model.to(device)

# %%
def _eta_helper(rnn_inp):
    inp = model.q_eta_map(rnn_inp).unsqueeze(1)
    hidden = model.init_hidden()
    output, _ = model.q_eta(inp, hidden)
    output = output.squeeze()
    etas = torch.zeros(model.num_times, model.num_topics).to(device)
    inp_0 = torch.cat([output[0], torch.zeros(model.num_topics,).to(device)], dim=0)
    etas[0] = model.mu_q_eta(inp_0)
    for t in range(1, model.num_times):
        inp_t = torch.cat([output[t], etas[t-1]], dim=0)
        etas[t] = model.mu_q_eta(inp_t)
    return etas

def get_eta(source):
    model.eval()
    with torch.no_grad():
        if source == 'val':
            rnn_inp = valid_rnn_inp
            return _eta_helper(rnn_inp)
        else:
            rnn_1_inp = test_1_rnn_inp
            return _eta_helper(rnn_1_inp)

def get_theta(eta, bows):
    model.eval()
    with torch.no_grad():
        inp = torch.cat([bows, eta], dim=1)
        q_theta = model.q_theta(inp)
        mu_theta = model.mu_q_theta(q_theta)
        theta = F.softmax(mu_theta, dim=-1)
        return theta    

def get_completion_ppl(source):
    """Returns document completion perplexity.
    """
    model.eval()
    with torch.no_grad():
        alpha = model.mu_q_alpha
        if source == 'val':
            indices = torch.split(torch.tensor(range(args.num_docs_valid)), args.eval_batch_size)
            tokens = valid_tokens
            counts = valid_counts
            times = valid_times

            eta = get_eta('val')

            acc_loss = 0
            cnt = 0
            for idx, ind in enumerate(indices):
                data_batch, times_batch = data.get_batch(
                    tokens, counts, ind, args.vocab_size, args.emb_size, temporal=True, times=times)
                sums = data_batch.sum(1).unsqueeze(1)
                if args.bow_norm:
                    normalized_data_batch = data_batch / sums
                else:
                    normalized_data_batch = data_batch

                eta_td = eta[times_batch.type('torch.LongTensor')]
                theta = get_theta(eta_td, normalized_data_batch)
                alpha_td = alpha[:, times_batch.type('torch.LongTensor'), :]
                beta = model.get_beta(alpha_td).permute(1, 0, 2)
                loglik = theta.unsqueeze(2) * beta
                loglik = loglik.sum(1)
                loglik = torch.log(loglik)
                nll = -loglik * data_batch
                nll = nll.sum(-1)
                loss = nll / sums.squeeze()
                loss = loss.mean().item()
                acc_loss += loss
                cnt += 1
            cur_loss = acc_loss / cnt
            ppl_all = round(math.exp(cur_loss), 1)
            print('*'*100)
            print('{} PPL: {}'.format(source.upper(), ppl_all))
            print('*'*100)
            return ppl_all
        else: 
            indices = torch.split(torch.tensor(range(args.num_docs_test)), args.eval_batch_size)
            tokens_1 = test_1_tokens
            counts_1 = test_1_counts

            tokens_2 = test_2_tokens
            counts_2 = test_2_counts

            eta_1 = get_eta('test')

            acc_loss = 0
            cnt = 0
            indices = torch.split(torch.tensor(range(args.num_docs_test)), args.eval_batch_size)
            for idx, ind in enumerate(indices):
                data_batch_1, times_batch_1 = data.get_batch(
                    tokens_1, counts_1, ind, args.vocab_size, args.emb_size, temporal=True, times=test_times)
                sums_1 = data_batch_1.sum(1).unsqueeze(1)
                if args.bow_norm:
                    normalized_data_batch_1 = data_batch_1 / sums_1
                else:
                    normalized_data_batch_1 = data_batch_1

                eta_td_1 = eta_1[times_batch_1.type('torch.LongTensor')]
                theta = get_theta(eta_td_1, normalized_data_batch_1)

                data_batch_2, times_batch_2 = data.get_batch(
                    tokens_2, counts_2, ind, args.vocab_size, args.emb_size, temporal=True, times=test_times)
                sums_2 = data_batch_2.sum(1).unsqueeze(1)

                alpha_td = alpha[:, times_batch_2.type('torch.LongTensor'), :]
                beta = model.get_beta(alpha_td).permute(1, 0, 2)
                loglik = theta.unsqueeze(2) * beta
                loglik = loglik.sum(1)
                loglik = torch.log(loglik)
                nll = -loglik * data_batch_2
                nll = nll.sum(-1)
                loss = nll / sums_2.squeeze()
                loss = loss.mean().item()
                acc_loss += loss
                cnt += 1
            cur_loss = acc_loss / cnt
            ppl_dc = round(math.exp(cur_loss), 1)
            print('*'*100)
            print('{} Doc Completion PPL: {}'.format(source.upper(), ppl_dc))
            print('*'*100)
            return ppl_dc

def get_beta(model):
    """Returns document completion perplexity.
    """
    model.eval()
    with torch.no_grad():
        alpha = model.mu_q_alpha
        betas = []
        for time in range(num_time_slices):
            alpha_td = alpha[:, time, :].unsqueeze(1)
            beta = model.get_beta(alpha_td).permute(1, 0, 2)
            betas.append(beta.cpu().numpy())
        betas = np.array(betas)
        betas = np.squeeze(betas, axis=1)
        return betas

def _diversity_helper(beta, num_tops):
    list_w = np.zeros((args.num_topics, num_tops))
    for k in range(args.num_topics):
        gamma = beta[k, :]
        top_words = gamma.cpu().numpy().argsort()[-num_tops:][::-1]
        list_w[k, :] = top_words
    list_w = np.reshape(list_w, (-1))
    list_w = list(list_w)
    n_unique = len(np.unique(list_w))
    diversity = n_unique / (args.num_topics * num_tops)
    return diversity

def calculate_coherence(phi, corpus, N_word):
    corpus_diff_words = [list(map(lambda x: x[0], corpus[i])) for i in range(len(corpus))]
    word_list = []
    for i in range(phi.shape[0]):
        word_list.append(np.argsort(phi[i,:],)[-N_word:])
    coherence = np.zeros(phi.shape[0])
    for k in range(phi.shape[0]):
        for i in range(N_word-1):
            for j in range(i+1):
                count_gongxian = 0
                count_single = 0
                for n in range(len(corpus_diff_words)):
                    count_gongxian += int(set([word_list[k][i+1],word_list[k][j]]).issubset(set(corpus_diff_words[n])))
                    count_single += int(set([word_list[k][j]]).issubset(set(corpus_diff_words[n])))
                if count_single > 0:
                    coherence[k] += np.log((count_gongxian+1)/count_single)
    return coherence

def calculate_coherence_dynamic(topic_word, corpus, num_time_slices, time_slice, N):
    css = []
    for t in range(num_time_slices):
        if t == 0:
            d_slice = range(np.cumsum(time_slice)[t])
        else:
            d_slice = range(np.cumsum(time_slice)[t - 1], np.cumsum(time_slice)[t])
        corpus_sample_t = [corpus[d] for d in d_slice]
        cs = calculate_coherence(topic_word[t], corpus_sample_t, N).mean()
        css.append(cs)
    return css

def get_topic_quality(corpus):
    """Returns topic coherence and topic diversity.
    """
    model.eval()
    with torch.no_grad():
        alpha = model.mu_q_alpha
        beta = model.get_beta(alpha) 
        print('beta: ', beta.size())

        print('\n')
        print('#'*100)
        print('Get topic diversity...')
        num_tops = 25
        TD_all = np.zeros((args.num_times,))
        for tt in range(args.num_times):
            TD_all[tt] = _diversity_helper(beta[:, tt, :], num_tops)
        TD = np.mean(TD_all)
        print('Topic Diversity is: {}'.format(TD))
        print('\n')
        print('Get topic coherence...')
        print('train_tokens: ', train_tokens[0])
        
        beta2 = torch.transpose(beta, 0, 1).cpu().numpy()
        TC_all = calculate_coherence_dynamic(beta2, corpus, num_time_slices, time_slice, 50)

        print('TC_all: ', TC_all)

        return TD, TC_all

def Perplexity_static(theta, Phi, testset, num_topics):
    """calculate the perplexity of a lda-model"""
    #print ('the info of this ldamodel: \n')
    #print ('num of testset: %s; size_dictionary: %s; num of topics: %s'%(len(testset), size_dictionary, num_topics))
    prep = 0.0
    prob_doc_sum = 0.0
    testset_word_num = 0
    for i in range(len(testset)):
        prob_doc = 0.0 # the probablity of the doc
        doc = testset[i] #得到对应词编号&数量
        doc_word_num = 0 # the num of words in the doc
        for word_id, num in doc:
            prob_word = 0.0 # the probablity of the word
            doc_word_num += num#词出现总数

            for topic_id in range(num_topics):
                # cal p(w) : p(w) = sumz(p(z)*p(w|z))
                prob_topic = theta[i][topic_id]#p(z) 主题为z的概率
                prob_topic_word = Phi[topic_id][word_id]#p(w|z)主题为z时取不同词的概率

                prob_word += prob_topic*prob_topic_word#p(w)
            prob_doc += math.log(max(1e-7,prob_word))*num # p(d) = sum(log(p(w)))，num为词数
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum/testset_word_num) # perplexity = exp(-sum(p(d)/sum(Nd))
    #print ("the perplexity of this dtmmodel is : %s"%prep)
    return prep

def perplexity_dynamic(theta, topic_word, testset, num_topics, num_time_slices, time_slice):
    perplexity = 0
    for t in range(num_time_slices):
        if t == 0:
            d_slice = range(np.cumsum(time_slice)[t])
        else:
            d_slice = range(np.cumsum(time_slice)[t-1],np.cumsum(time_slice)[t])
        corpus_t = [testset[d] for d in d_slice]
        perplexity += Perplexity_static(theta[d_slice], topic_word[t], corpus_t, num_topics)
    perplexity = perplexity / num_time_slices
    return perplexity

# %%
topic_word = get_beta(model)

# %%
theta_train = get_newcorpus_theta(corpus_train, num_topics, topic_word, num_time_slices, time_slice)
train_ppl = perplexity_dynamic(theta_train, topic_word, corpus_train, num_topics, num_time_slices, time_slice)
print(train_ppl)

# %%
train_ppl = get_completion_ppl('val')

# %%
print('computing topic coherence and topic diversity...')
td, CS = get_topic_quality(corpus_train)

# %%
res = {'train_ppl': train_ppl, 'val_ppl': val_ppl, 'test_ppl': test_ppl, 'td': td}
with open('./res/res_ch.pkl', 'wb') as file:
    pickle.dump(res, file)
