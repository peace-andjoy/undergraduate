from gensim.corpora import Dictionary
from gensim.models import LdaSeqModel
from gensim.models.ldamodel import LdaModel
from wordsparsedynamic import WSTMdynamic, get_data_sp, get_WSTM_data, get_corpus
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math

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

def perplexity_dynamic2(theta, topic_word, testset, num_topics, num_time_slices, time_slice):
    prob_doc_sum = 0.0
    testset_word_num = 0
    for t in range(num_time_slices):
        if t == 0:
            d_slice = range(np.cumsum(time_slice)[t])
        else:
            d_slice = range(np.cumsum(time_slice)[t-1],np.cumsum(time_slice)[t])
        corpus_t = [testset[d] for d in d_slice]
        for i in range(len(corpus_t)):
            prob_doc = 0.0  # the probablity of the doc
            doc = corpus_t[i]  # 得到对应词编号&数量
            doc_word_num = 0  # the num of words in the doc
            for word_id, num in doc:
                prob_word = 0.0  # the probablity of the word
                doc_word_num += num  # 词出现总数

                for topic_id in range(num_topics):
                    # cal p(w) : p(w) = sumz(p(z)*p(w|z))
                    prob_topic = theta[i][topic_id]  # p(z) 主题为z的概率
                    prob_topic_word = topic_word[t][topic_id][word_id]  # p(w|z)主题为z时取不同词的概率

                    prob_word += prob_topic * prob_topic_word  # p(w)
                prob_doc += math.log(max(1e-7, prob_word)) * num  # p(d) = sum(log(p(w)))，num为词数
            prob_doc_sum += prob_doc
            testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum / testset_word_num)  # perplexity = exp(-sum(p(d)/sum(Nd))
    return prep

def calculate_pmi(Phi, N, corpus):
    '''
    Input: Phi: Phi的估计值
    '''
    important_word = np.apply_along_axis(lambda x: np.argsort(-x)[:N], 1, Phi)  # 每个主题的高频词，K*N
    K = Phi.shape[0]
    pmik = np.zeros(K)
    for k in range(K):
        pmikk = 0
        for i in range(N - 1):  # 选择第一个term
            for j in range(i + 1, N):  # 选择第二个term
                D_i = 0
                D_j = 0
                D_ij = 0
                for doc in corpus:  # 每篇文档
                    words = [word for word, freq in doc]
                    if important_word[k, i] in words:
                        D_i += 1
                        if important_word[k, j] in words:
                            D_j += 1
                            D_ij += 1
                    elif important_word[k, j] in words:
                        D_j += 1

                if D_ij != 0:
                    pmikk += np.log(D_ij) - np.log(D_i) - np.log(D_j) + np.log(len(corpus))

        pmik[k] = 2 * pmikk / (N * (N - 1))

    return pmik

def calculate_pmis(topic_word, corpus, num_time_slices, time_slice, N):
    pmis = []
    for t in range(num_time_slices):
        if t == 0:
            d_slice = range(np.cumsum(time_slice)[t])
        else:
            d_slice = range(np.cumsum(time_slice)[t - 1], np.cumsum(time_slice)[t])
        corpus_sample_t = [corpus[d] for d in d_slice]
        pmi = calculate_pmi(topic_word[t], N, corpus_sample_t).mean()
        pmis.append(pmi)
    return pmis


# 静态版本的sKL：输入Phi：K*V维的主题-词分布
def sKL_static(Phi, topici, topicj):
    probi = Phi[topici]
    probj = Phi[topicj]
    probi[probi == 0] = 1e-5
    probj[probj == 0] = 1e-5
    skl = 0.5 * np.sum(probi * np.log(probi / probj) + probj * np.log(probj / probi))

    return skl


def sKL_sum_static(Phi):
    num_topics = Phi.shape[0]
    skl_sum = 0.0
    for topic_i in range(num_topics):
        for topic_j in range(num_topics):
            if topic_i == topic_j:
                pass
            else:
                skl_sum += sKL_static(Phi, topic_i, topic_j)
    return (skl_sum)

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