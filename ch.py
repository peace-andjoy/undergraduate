#!/usr/bin/env python
# coding: utf-8

# In[1]:
import gensim
from gensim.models.ldamodel import LdaModel
from gensim.models import LdaSeqModel, HdpModel
import pandas as pd
import pickle
from wordsparsedynamic import WSTMdynamic
from DSTM_fast import DSTM_para
import numpy as np
from index import perplexity_dynamic, calculate_pmis, sKL_sum_static, calculate_coherence_dynamic, perplexity_dynamic2

# %%
data = pickle.load(open('ch.pkl','rb'))
data = data[data['年份']!=2003]
data.reset_index(inplace = True, drop = True)

dictionary_data = gensim.corpora.Dictionary(data['ab_words'])
dictionary_data.filter_extremes(no_below=5)
corpus_data = [dictionary_data.doc2bow(doc) for doc in data['ab_words']]
time_slice = data['年份'].value_counts().sort_index().values.tolist()

# %%
num_topics = 30
size_dictionary = len(dictionary_data)
vocab_len = size_dictionary
corpus_len = len(corpus_data)
num_time_slices = len(time_slice)

# %%
# LDA
topic_word_LDA = []
LDA = LdaModel(corpus=corpus_data, num_topics=num_topics, id2word=dictionary_data, minimum_probability=0.0,random_state=100)
Phi_LDA = LDA.state.sstats
for k in range(num_topics):
    Phi_LDA[k] = Phi_LDA[k] / np.sum(Phi_LDA[k])
for t in range(num_time_slices):
    topic_word_LDA.append(Phi_LDA)
topic_word_LDA = np.asarray(topic_word_LDA)

# %%

# DTM
DTM = LdaSeqModel(corpus=corpus_data, id2word=dictionary_data, num_topics=num_topics, time_slice=time_slice,random_state=100)
topic_word_DTM = np.zeros((num_time_slices, num_topics, vocab_len))
for k in range(num_topics):
    topic_word_DTM[:, k, :] = np.exp(DTM.topic_chains[k].e_log_prob).T
for t in range(num_time_slices):
    for k in range(num_topics):
        topic_word_DTM[t, k] = topic_word_DTM[t, k] / np.sum(topic_word_DTM[t, k])

# %%
# sDTM建模
sparsedynamic = WSTMdynamic(corpus=corpus_data, time_slice=time_slice, id2word=dictionary_data, num_topics=num_topics,random_state=100)
topic_word_sDTM = sparsedynamic.topic_word

# %%
# DSTM(双稀疏)建模 合在一起
DSTM2_parameter = DSTM_para(corpus=corpus_data, id2word=dictionary_data, num_topics=num_topics, iterations=50,
                            alphas=1 / num_topics, alpha_bar=1e-7, beta_bar=1e-7,seed=100)
DSTM2 = DSTM2_parameter.fit_DSTM_model(iterations=50)
topic_word_DSTM2 = []
for t in range(num_time_slices):
    topic_word_DSTM2.append(DSTM2_parameter.estimate_Phi())
topic_word_DSTM2 = np.asarray(topic_word_DSTM2)

# %%
# HDP建模
print('HDP')
hdp = HdpModel(corpus=corpus_data, id2word=dictionary_data)
topic_word_HDP = []
for t in range(num_time_slices):
    topic_word_HDP.append(hdp.lda_beta[:num_topics])
topic_word_HDP = np.asarray(topic_word_HDP)

# 计算CS
# CS_LDA50 = calculate_coherence_dynamic(topic_word_LDA, corpus_data, num_time_slices, time_slice, 50)
# CS_DTM50 = calculate_coherence_dynamic(topic_word_DTM, corpus_data, num_time_slices, time_slice, 50)
# CS_sDTM50 = calculate_coherence_dynamic(topic_word_sDTM, corpus_data, num_time_slices, time_slice, 50)
# CS_DSTM2_50 = calculate_coherence_dynamic(topic_word_DSTM2, corpus_data, num_time_slices, time_slice, 50)

# %%
CS_HDP_50 = calculate_coherence_dynamic(topic_word_HDP, corpus_data, num_time_slices, time_slice, 50)

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

def _diversity_helper(beta, num_tops):
    list_w = np.zeros((num_topics, num_tops))
    for k in range(num_topics):
        gamma = beta[k, :]
        top_words = gamma.argsort()[-num_tops:][::-1]
        list_w[k, :] = top_words
    list_w = np.reshape(list_w, (-1))
    list_w = list(list_w)
    n_unique = len(np.unique(list_w))
    diversity = n_unique / (num_topics * num_tops)
    return diversity

def get_topic_diversity(betas, num_tops=25):
    TD_all = np.zeros((num_time_slices))
    for tt in range(num_time_slices):
        TD_all[tt] = _diversity_helper(betas[tt], num_tops)
        TD = np.mean(TD_all)
    return TD

# 估计theta
# theta_test_DTM = get_newcorpus_theta(corpus_data, num_topics, topic_word_DTM, num_time_slices, time_slice)
# theta_test_LDA = get_newcorpus_theta(corpus_data, num_topics, topic_word_LDA, num_time_slices, time_slice)
# theta_test_sDTM = get_newcorpus_theta(corpus_data, num_topics, topic_word_sDTM, num_time_slices, time_slice)
# theta_test_DSTM2 = get_newcorpus_theta(corpus_data, num_topics, topic_word_DSTM2, num_time_slices, time_slice)
theta_test_HDP = get_newcorpus_theta(corpus_data, num_topics, topic_word_HDP, num_time_slices, time_slice)
# 计算perplexity
# perplexity_LDA = perplexity_dynamic(theta_test_LDA, topic_word_LDA, corpus_data, num_topics, num_time_slices,
#                                     time_slice)

# perplexity_DTM = perplexity_dynamic(theta_test_DTM, topic_word_DTM, corpus_data, num_topics, num_time_slices,
#                                     time_slice)

# perplexity_sDTM = perplexity_dynamic(theta_test_sDTM, topic_word_sDTM, corpus_data, num_topics, num_time_slices,
#                                      time_slice)

# perplexity_DSTM2 = perplexity_dynamic(theta_test_DSTM2, topic_word_DSTM2, corpus_data, num_topics, num_time_slices,
#                                       time_slice)
perplexity_HDP = perplexity_dynamic(theta_test_HDP, topic_word_HDP, corpus_data, num_topics, num_time_slices,
                                      time_slice)

# CS_DTM50 = np.mean(CS_DTM50)
# CS_LDA50 = np.mean(CS_LDA50)
# CS_sDTM50 = np.mean(CS_sDTM50)
# CS_DSTM2_50 = np.mean(CS_DSTM2_50)
CS_HDP_50 = np.mean(CS_HDP_50)

#计算td
td_LDA = get_topic_diversity(topic_word_LDA)
td_DTM = get_topic_diversity(topic_word_DTM)
td_sDTM = get_topic_diversity(topic_word_sDTM)
td_DSTM2 = get_topic_diversity(topic_word_DSTM2)
td_HDP = get_topic_diversity(topic_word_HDP)

df = pd.DataFrame({'model':['LDA','wsDTM','DTM','DSTM','HDP'], \
                   'Perplexity':[0, 0, 0, 0, perplexity_HDP], \
                   'CS50':[0, 0, 0, 0, CS_HDP_50],\
                    'TD': [td_LDA, td_sDTM, td_DTM, td_DSTM2, td_HDP]})

df.to_csv('output_new/RUCRSS%d.csv' %num_topics, index = False)

with open ("output_new/RUCRSS%d.txt" %num_topics, 'wb') as f: #打开文件
    pickle.dump(sparsedynamic, f) #用 dump 函数将 Python 对象转成二进制对象文件
# %%
