# %%
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import LdaSeqModel
import logging
from wordsparsedynamic import WSTMdynamic, get_WSTM_data, get_corpus
import numpy as np
from index import perplexity_dynamic, calculate_pmis, sKL_sum_static, calculate_coherence_dynamic, perplexity_dynamic2
from DSTM_fast import DSTM_para
import pandas as pd
import time
import pickle
import os

# %%
def get_test_data(traindata, nTerms=100, T=3, M=100, K=3, N=75, x=0.5, y=0.5, mean=0.5, sd=0.5, seed=1234):
    """
    参数说明：
    *nTerms：单词表的总词数
    *M:文档数
    *K:主题数
    *ND:每篇文档词数
    *s,v,x,y:主题以及词稀疏Beta分布的参数
    """
    pi = 1/K
    beta_bar = 1e-7

    Mt = np.full(shape=[T], fill_value=M, dtype=int)
    M = np.sum(Mt)  # 文档总数
    Nd = np.full(shape=[M], fill_value=N, dtype=int)
    Pi = np.full(K, pi)

    Beta = traindata['beta']

    # gamma--beta分布得到的所有词表达的概率，维数T*K
    p2 = traindata['p2']

    # b--词稀疏参数,维数T*K*nTerms
    b = traindata['b']

    # pi平滑--dir先验参数 M*K维
    Pi_final = [Pi for i in range(M)]
    np.random.seed(seed)
    # Theta[d]--第d篇文档的主题分布
    Theta = [np.random.dirichlet(Pi_final[i]) for i in range(M)]

    # Phi[k]--第k个主题的词分布
    Phi = traindata['phi']

    # z_[d][n]--第d篇文档第n个词的主题
    z = np.zeros((M, N, K))
    z_ = np.zeros((M, N))
    K_topic = range(K)
    for d in range(M):
        for n in range(Nd[d]):
            z[d][n] = np.random.multinomial(1, Theta[d])
            z_[d][n] = np.dot(z[d][n], K_topic)

    # w_[d][n]--第d篇文档第n个词的内容
    w = np.zeros((M, N, nTerms))
    w_ = np.zeros((M, N))
    for t in range(T):
        if t == 0:
            d_slice = range(Mt[t])
        else:
            d_slice = range(np.cumsum(Mt)[t - 1], np.cumsum(Mt)[t])

        for d in d_slice:
            for n in range(Nd[d]):
                z_dn = int(z_[d][n])
                w[d][n] = np.random.multinomial(1, Phi[t, z_dn])
                w_[d][n] = int(np.dot(w[d][n], range(nTerms)))

    d = dict()
    d['p2'] = p2
    d['b'] = b
    d['phi'] = Phi
    d['theta'] = Theta
    d['z'] = z_
    d['w'] = w_
    d['pi'] = Pi_final
    d['beta'] = Beta
    return d

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

# %%
m_set = 200
t_set = 20
num_docs = m_set  # 每个时刻的文档数M_t
num_terms = 1000  # 词表长度V
len_docs = 100  # 文档长度N_d
num_topics = 20  # 主题数K
num_time_slices = t_set  # 时刻数T
data0 = get_WSTM_data(nTerms=num_terms, T=num_time_slices, M=num_docs, K=num_topics, N=len_docs, seed=123)

# %%
sim_num = 20
for sim in range(sim_num):
    # 生成模拟数据
    data = get_test_data(data0,nTerms=num_terms, T=num_time_slices, M=num_docs, K=num_topics, N=len_docs,seed=sim)
    words_series = get_corpus(data)
    dict_sample = Dictionary(words_series.words)
    corpus_sample = [dict_sample.doc2bow(doc) for doc in words_series.words]  # 转化为词袋形式，用于后续读取操作
    size_dictionary = len(dict_sample)
    vocab_len = size_dictionary
    time_slice = [num_docs for time in range(num_time_slices)]
    corpus_len = len(corpus_sample)
    train = get_dic_data(corpus_sample, time_slice)

    # 验证集
    #生成模拟数据
    num_docs_valid = round(num_docs/4)
    validdata = get_test_data(data, nTerms = num_terms,T=num_time_slices,M = num_docs_valid,K = num_topics,N = len_docs,seed =sim*10)
    words_valid = get_corpus(validdata)
    corpus_valid = [dict_sample.doc2bow(doc) for doc in words_valid.words]#转化为词袋形式，用于后续读取操作
    time_slice_valid = np.repeat(num_docs_valid, num_time_slices)
    valid = get_dic_data(corpus_valid, time_slice_valid)

    # 测试集
    #生成模拟数据
    num_docs_test = round(num_docs/4)
    testdata = get_test_data(data, nTerms = num_terms,T=num_time_slices,M = num_docs_test,K = num_topics,N = len_docs,seed =2*(sim+1))
    words_test = get_corpus(testdata)
    corpus_test = [dict_sample.doc2bow(doc) for doc in words_test.words]#转化为词袋形式，用于后续读取操作
    time_slice_test = np.repeat(num_docs_test, num_time_slices)
    test = get_dic_data(corpus_test, time_slice_test)

    if not os.path.exists('./simulate_data'):
        os.mkdir('./simulate_data')

    with open('./simulate_data/train_M{}_N{}_K{}_T{}_{}.pkl'.format(num_docs, len_docs, num_topics, num_time_slices, sim), 'wb') as file:
        pickle.dump(train, file)
    with open('./simulate_data/valid_M{}_N{}_K{}_T{}_{}.pkl'.format(num_docs, len_docs, num_topics, num_time_slices, sim), 'wb') as file:
        pickle.dump(valid, file)
    with open('./simulate_data/test_M{}_N{}_K{}_T{}_{}.pkl'.format(num_docs, len_docs, num_topics, num_time_slices, sim), 'wb') as file:
        pickle.dump(test, file)

# %%
 