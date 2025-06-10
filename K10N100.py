from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import LdaSeqModel
import logging
logger = logging.getLogger(__name__)
from wordsparsedynamic import WSTMdynamic, get_WSTM_data, get_corpus
import numpy as np
from index import perplexity_dynamic, calculate_pmis, sKL_sum_static, calculate_coherence_dynamic, perplexity_dynamic2
from DSTM_fast import DSTM_para
import pandas as pd
import time

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

for m_set in [100,200]:
    for t_set in [5,10,20]:
        l_perp_sDTM = []
        l_perp0_sDTM = []
        l_CS_sDTM50 = []
        l_perp_DTM = []
        l_perp0_DTM = []
        l_CS_DTM50 = []
        l_perp_LDA = []
        l_perp0_LDA = []
        l_CS_LDA50 = []
        l_perp_DSTM2 = []
        l_perp0_DSTM2 = []
        l_CS_DSTM2_50 = []

        num_docs = m_set  # 每个时刻的文档数M_t
        num_terms = 1000  # 词表长度V
        len_docs = 100  # 文档长度N_d
        num_topics = 10  # 主题数K
        num_time_slices = t_set  # 时刻数T
        data0 = get_WSTM_data(nTerms=num_terms, T=num_time_slices, M=num_docs, K=num_topics, N=len_docs, seed=123)
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
            #LDA
            topic_word_LDA = []
            LDA = LdaModel(corpus=corpus_sample, num_topics=num_topics, id2word=dict_sample, minimum_probability=0.0,
                           random_state=123)
            Phi_LDA = LDA.state.sstats
            for k in range(num_topics):
                Phi_LDA[k] = Phi_LDA[k] / np.sum(Phi_LDA[k])
            for t in range(num_time_slices):
                topic_word_LDA.append(Phi_LDA)
            topic_word_LDA = np.asarray(topic_word_LDA)

            #DTM
            DTM = LdaSeqModel(corpus=corpus_sample, id2word=dict_sample, num_topics=num_topics, time_slice=time_slice,random_state=123)
            topic_word_DTM = np.zeros((num_time_slices, num_topics, vocab_len))
            for k in range(num_topics):
                topic_word_DTM[:, k, :] = np.exp(DTM.topic_chains[k].e_log_prob).T
            for t in range(num_time_slices):
                for k in range(num_topics):
                    topic_word_DTM[t, k] = topic_word_DTM[t, k] / np.sum(topic_word_DTM[t, k])

            #sDTM建模
            sparsedynamic = WSTMdynamic(corpus = corpus_sample, time_slice = time_slice, id2word = dict_sample, num_topics = num_topics, random_state=123)
            topic_word_sDTM = sparsedynamic.topic_word

            #DSTM(双稀疏)建模 合在一起
            DSTM2_parameter = DSTM_para(corpus=corpus_sample, id2word=dict_sample, num_topics=num_topics, iterations=50,
                                       alphas=1 / num_topics, alpha_bar=1e-7, beta_bar=1e-7,seed=123)
            DSTM2 = DSTM2_parameter.fit_DSTM_model(iterations=50)
            topic_word_DSTM2 = []
            for t in range(num_time_slices):
                topic_word_DSTM2.append(DSTM2_parameter.estimate_Phi())
            topic_word_DSTM2 = np.asarray(topic_word_DSTM2)

            #计算CS
            CS_LDA50 = calculate_coherence_dynamic(topic_word_LDA, corpus_sample, num_time_slices, time_slice, 50)
            CS_DTM50 = calculate_coherence_dynamic(topic_word_DTM, corpus_sample, num_time_slices, time_slice, 50)
            CS_sDTM50 = calculate_coherence_dynamic(topic_word_sDTM, corpus_sample, num_time_slices, time_slice, 50)
            CS_DSTM2_50 = calculate_coherence_dynamic(topic_word_DSTM2, corpus_sample, num_time_slices, time_slice, 50)
            # 训练集
            #估计theta
            theta0_test_DTM = get_newcorpus_theta(corpus_sample, num_topics, topic_word_DTM, num_time_slices, time_slice)
            theta0_test_LDA = get_newcorpus_theta(corpus_sample, num_topics, topic_word_LDA, num_time_slices, time_slice)
            theta0_test_sDTM = get_newcorpus_theta(corpus_sample, num_topics, topic_word_sDTM, num_time_slices, time_slice)
            theta0_test_DSTM2 = get_newcorpus_theta(corpus_sample, num_topics, topic_word_DSTM2, num_time_slices, time_slice)
            # 计算perplexity
            perplexity0_LDA = perplexity_dynamic(theta0_test_LDA, topic_word_LDA, corpus_sample, num_topics, num_time_slices,
                                                time_slice)
            perplexity0_DTM = perplexity_dynamic(theta0_test_DTM, topic_word_DTM, corpus_sample, num_topics, num_time_slices,
                                                time_slice)
            perplexity0_sDTM = perplexity_dynamic(theta0_test_sDTM, topic_word_sDTM, corpus_sample, num_topics, num_time_slices,
                                                 time_slice)
            perplexity0_DSTM2 = perplexity_dynamic(theta0_test_DSTM2, topic_word_DSTM2, corpus_sample, num_topics, num_time_slices,
                                                 time_slice)

            # 测试集
            #生成模拟数据
            num_docs_test = round(num_docs/4)
            testdata = get_test_data(data, nTerms = num_terms,T=num_time_slices,M = num_docs_test,K = num_topics,N = len_docs,seed =sim*2)
            words_test = get_corpus(testdata)
            corpus_test = [dict_sample.doc2bow(doc) for doc in words_test.words]#转化为词袋形式，用于后续读取操作
            time_slice_test = np.repeat(num_docs_test, num_time_slices)
            #估计theta
            theta_test_DTM = get_newcorpus_theta(corpus_test, num_topics, topic_word_DTM, num_time_slices, time_slice_test)
            theta_test_LDA = get_newcorpus_theta(corpus_test, num_topics, topic_word_LDA, num_time_slices, time_slice_test)
            theta_test_sDTM = get_newcorpus_theta(corpus_test, num_topics, topic_word_sDTM, num_time_slices, time_slice_test)
            theta_test_DSTM2 = get_newcorpus_theta(corpus_test, num_topics, topic_word_DSTM2, num_time_slices, time_slice_test)
            # 计算perplexity
            perplexity_LDA = perplexity_dynamic(theta_test_LDA, topic_word_LDA, corpus_test, num_topics, num_time_slices,
                                                time_slice_test)
            perplexity_DTM = perplexity_dynamic(theta_test_DTM, topic_word_DTM, corpus_test, num_topics, num_time_slices,
                                                time_slice_test)
            perplexity_sDTM = perplexity_dynamic(theta_test_sDTM, topic_word_sDTM, corpus_test, num_topics, num_time_slices,
                                                 time_slice_test)
            perplexity_DSTM2 = perplexity_dynamic(theta_test_DSTM2, topic_word_DSTM2, corpus_test, num_topics, num_time_slices,
                                                 time_slice_test)

            CS_DTM50 = np.mean(CS_DTM50)
            CS_LDA50 = np.mean(CS_LDA50)
            CS_sDTM50 = np.mean(CS_sDTM50)
            CS_DSTM2_50 = np.mean(CS_DSTM2_50)

            print('第', sim, '次实验')
            print('Perplexity,DTM:',perplexity_DTM, 'LDA:',perplexity_LDA, 'sDTM:',perplexity_sDTM,
                   'DSTM2:', perplexity_DSTM2)
            print('Perplexity0,DTM:',perplexity0_DTM , 'LDA:',perplexity0_LDA, 'sDTM:', perplexity0_sDTM,
                   'DSTM2:', perplexity0_DSTM2)
            print('CS,DTM50:', CS_DTM50, 'LDA50', CS_LDA50, 'sDTM50', CS_sDTM50, 'DSTM2_50:', CS_DSTM2_50)

            l_perp_DTM.append(perplexity_DTM)
            l_perp0_DTM.append(perplexity0_DTM)
            l_CS_DTM50.append(CS_DTM50)

            l_perp_LDA.append(perplexity_LDA)
            l_perp0_LDA.append(perplexity0_LDA)
            l_CS_LDA50.append(CS_LDA50)

            l_perp_sDTM.append(perplexity_sDTM)
            l_perp0_sDTM.append(perplexity0_sDTM)
            l_CS_sDTM50.append(CS_sDTM50)

            l_perp_DSTM2.append(perplexity_DSTM2)
            l_perp0_DSTM2.append(perplexity0_DSTM2)
            l_CS_DSTM2_50.append(CS_DSTM2_50)

        #计算所有实验的均值
        mean_perp_sDTM = np.mean(l_perp_sDTM)
        mean_perp0_sDTM = np.mean(l_perp0_sDTM)
        mean_CS_sDTM50 = np.mean(l_CS_sDTM50)

        mean_perp_DTM = np.mean(l_perp_DTM)
        mean_perp0_DTM = np.mean(l_perp0_DTM)
        mean_CS_DTM50 = np.mean(l_CS_DTM50)

        mean_perp_LDA = np.mean(l_perp_LDA)
        mean_perp0_LDA = np.mean(l_perp0_LDA)
        mean_CS_LDA50 = np.mean(l_CS_LDA50)

        mean_perp_DSTM2 = np.mean(l_perp_DSTM2)
        mean_perp0_DSTM2 = np.mean(l_perp0_DSTM2)
        mean_CS_DSTM2_50 = np.mean(l_CS_DSTM2_50)

        #计算所有实验的标准差
        std_perp_sDTM = np.std(l_perp_sDTM,ddof=1)
        std_perp0_sDTM = np.std(l_perp0_sDTM,ddof=1)
        std_CS_sDTM50 = np.std(l_CS_sDTM50,ddof=1)

        std_perp_DTM = np.std(l_perp_DTM,ddof=1)
        std_perp0_DTM = np.std(l_perp0_DTM,ddof=1)
        std_CS_DTM50 = np.std(l_CS_DTM50,ddof=1)

        std_perp_LDA = np.std(l_perp_LDA,ddof=1)
        std_perp0_LDA = np.std(l_perp0_LDA,ddof=1)
        std_CS_LDA50 = np.std(l_CS_LDA50,ddof=1)

        std_perp_DSTM2 = np.std(l_perp_DSTM2,ddof=1)
        std_perp0_DSTM2 = np.std(l_perp0_DSTM2,ddof=1)
        std_CS_DSTM2_50 = np.std(l_CS_DSTM2_50,ddof=1)

        df = pd.DataFrame({'model':['wsDTM','LDA','DTM','DSTM2'], \
                           'Perplexity':[mean_perp_sDTM,mean_perp_LDA,mean_perp_DTM,mean_perp_DSTM2], \
                           'Perplexity_std':[std_perp_sDTM,std_perp_LDA,std_perp_DTM,std_perp_DSTM2], \
                           'Perplexity0': [mean_perp0_sDTM, mean_perp0_LDA, mean_perp0_DTM, mean_perp0_DSTM2], \
                           'Perplexity0_std': [std_perp0_sDTM, std_perp0_LDA, std_perp0_DTM, std_perp0_DSTM2], \
                           'CS50': [mean_CS_sDTM50,mean_CS_LDA50, mean_CS_DTM50, mean_CS_DSTM2_50], \
                           'CS50_std': [std_CS_sDTM50,std_CS_LDA50, std_CS_DTM50, std_CS_DSTM2_50]})

        df.to_csv('output/M%dN%dK%dT%dV%d.csv'%(num_docs,len_docs,num_topics,num_time_slices,num_terms), index = False)