#!/usr/bin/env python
# coding: utf-8

from gensim import utils
from scipy.special import gammaln, psi
import numpy as np
import pandas as pd
import time


# 生成模拟数据（不极端，正常抽样）
def get_WSTM_data(nTerms=100, M=20, K=3, N=30, x=2, y=2, mean = 0.5, sd = 1, seed=1234):
    """
    参数说明：
    *nTerms：单词表的总词数
    *M:文档数
    *K:主题数
    *ND:每篇文档词数
    *s,v,x,y:主题以及词稀疏Beta分布的参数
    """
    np.random.seed(seed)
    pi = 0.1
    beta_bar = 1e-5

    Nd = np.full(shape=[M], fill_value=N, dtype=int)
    Pi = np.full(K, pi)
    Beta = np.zeros((K, nTerms))
    for k in range(K):
        Beta[k] = np.random.normal(mean, sd, nTerms)
        Beta[k] = np.exp(Beta[k]) / np.sum(np.exp(Beta[k]))

    # gamma--beta分布得到的所有词表达的概率，维数K
    p2 = np.random.beta(x, y, K)
    p2 = np.asarray([p2[i].repeat(nTerms) for i in range(0, len(p2))])

    # b--词稀疏参数,维数K*nTerms
    b = np.random.binomial(1, p2)

    # pi平滑--dir先验参数
    Pi_final = [Pi for i in range(M)]

    # Theta[d]--第d篇文档的主题分布
    Theta = [np.random.dirichlet(Pi_final[i]) for i in range(M)]

    # Phi[k]--第k个主题的词分布
    Phi = Beta * b + beta_bar
    for k in range(K):
        Phi[k] = Phi[k] / np.sum(Phi[k])

    # z_[d][n]--第d篇文档第n个词的主题
    z = np.zeros((M, N, K))
    z_ = np.zeros((M, N))
    K_topic = range(K)
    for d in range(M):
        for n in range(Nd[d]):
            try:
                z[d][n] = np.random.multinomial(1, Theta[d])
            except:
                Theta[d] = np.repeat(1 / K, K)
                z[d][n] = np.random.multinomial(1, Theta[d])
            z_[d][n] = np.dot(z[d][n], K_topic)

    # w_[d][n]--第d篇文档第n个词的内容
    w = np.zeros((M, N, nTerms))
    w_ = np.zeros((M, N))
    for d in range(M):
        for n in range(Nd[d]):
            z_dn = int(z_[d][n])
            w[d][n] = np.random.multinomial(1, Phi[z_dn])
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

# 生成模拟数据（极端生成）
def get_data_sp(nTerms=100, M=20, K=3, N=30, x=1, y=1, mean = 0.5, sd = 1, seed=1234):
    """
    参数说明：
    *nTerms：单词表的总词数
    *M:文档数
    *K:主题数
    *ND:每篇文档词数
    *s,v,x,y:主题以及词稀疏Beta分布的参数
    """
    np.random.seed(seed)
    pi = 0.1
    beta_bar = 1e-5

    Nd = np.full(shape=[M], fill_value=N, dtype=int)
    Pi = np.full(K, pi)
    Beta = np.zeros((K, nTerms))
    for k in range(K):
        Beta[k] = np.random.normal(mean, sd, nTerms)
        Beta[k] = np.exp(Beta[k])
    #         Beta[k] = np.exp(Beta[k])/np.sum(np.exp(Beta[k]))

    # gamma--beta分布得到的所有词表达的概率，维数K
    p2 = np.random.beta(x, y, K)
    p2 = np.asarray([p2[i].repeat(nTerms) for i in range(0, len(p2))])

    # b--词稀疏参数,维数K*nTerms
    b = np.zeros((K, nTerms))
    i = int(nTerms / K)
    for k in range(K):
        b[k, range(i * k, i * (k + 1))] = np.repeat(1, i)

    # pi平滑--dir先验参数
    Pi_final = [Pi for i in range(M)]

    # Theta[d]--第d篇文档的主题分布
    Theta = [np.random.dirichlet(Pi_final[i]) for i in range(M)]

    # Phi[k]--第k个主题的词分布
    Phi = Beta * b + beta_bar
    for k in range(K):
        Phi[k] = Phi[k] / np.sum(Phi[k])

    # z_[d][n]--第d篇文档第n个词的主题
    z = np.zeros((M, N, K))
    z_ = np.zeros((M, N))
    K_topic = range(K)
    for d in range(M):
        for n in range(Nd[d]):
            try:
                z[d][n] = np.random.multinomial(1, Theta[d])
            except:
                Theta[d] = np.repeat(1 / K, K)
                z[d][n] = np.random.multinomial(1, Theta[d])
            z_[d][n] = np.dot(z[d][n], K_topic)

    # w_[d][n]--第d篇文档第n个词的内容
    w = np.zeros((M, N, nTerms))
    w_ = np.zeros((M, N))
    for d in range(M):
        for n in range(Nd[d]):
            z_dn = int(z_[d][n])
            w[d][n] = np.random.multinomial(1, Phi[z_dn])
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

# 将生成的数据格式转化为主题模型可以读入的形式
def get_corpus(data):
    words = np.array(np.array(data['w'], dtype=int), dtype=str)
    words_series = pd.DataFrame()
    word_series = pd.Series(dtype=object)
    words1 = [str(doc) for doc in words]
    words2 = pd.Series(words1)
    words3 = pd.Series(
        [words2[i].replace("\n", "").replace(" ", ",").replace("\'", "").replace('[', "").replace(']', "") for i in
         range(len(words2))])
    word_series.reset_index(inplace=True, drop=True)
    words_series['word'] = words3
    words_series['words'] = [words_series['word'][i].split(',') for i in range(len(words_series))]
    return (words_series)

def betaln(a, b):
    # 计算log Beta(a,b)，为后面公式更新准备
    betaln = gammaln(a) + gammaln(b) - gammaln(a + b)
    return (betaln)


class wordsparseTM(utils.SaveLoad):

    def __init__(self, corpus=None, id2word=None, x = 1, y = 1, pi = 1, beta_bar = 1e-5,
                 iters_E_max=50, iters_E_min = 5, iters_EM_max = 20, iters_EM_min = 5, threshold_E = 1,
                 threshold_EM = 1, num_topics=10,random_state=123):

        self.corpus = corpus
        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError(
                'at least one of corpus/id2word must be specified, to establish input space dimensionality'
            )

        if self.id2word is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.vocab_len = len(self.id2word)
        elif self.id2word:
            self.vocab_len = len(self.id2word)
        else:
            self.vocab_len = 0

        if corpus is not None:
            try:
                self.corpus_len = len(corpus)
            except TypeError:
                logger.warning("input corpus stream has no len(); counting documents")
                self.corpus_len = sum(1 for _ in corpus)
        vocab_len = self.vocab_len
        self.num_topics = num_topics
        self.pi = pi
        self.beta_bar = beta_bar
        self.iters_E_max = iters_E_max
        self.iters_E_min = iters_E_min
        self.iters_EM_max = iters_EM_max
        self.iters_EM_min = iters_EM_min
        self.threshold_E = threshold_E
        self.threshold_EM = threshold_EM
        self.x = x
        self.y = y

        self.fit_model()

    def beta_init(self):
        num_topics = self.num_topics
        vocab_len = self.vocab_len
        beta = np.zeros((num_topics, vocab_len))
        for k in range(num_topics):
            beta[k] = np.random.normal(0.5, 1, vocab_len)
            beta[k] = np.exp(beta[k])
        #         beta[k] = np.exp(beta[k])/np.sum(np.exp(beta[k]))
        return beta

    def b_init(self):
        num_topics = self.num_topics
        vocab_len = self.vocab_len
        x = self.x
        y = self.y
        bkv = np.random.beta(x, y, (num_topics, vocab_len))
        return bkv

    # topic_word: (b_kr*beta_kr+beta_bar)归一化后的期望
    def topic_word_init(self, beta, bkv):
        num_topics = self.num_topics
        beta_bar = self.beta_bar
        topic_word = beta * bkv + beta_bar
        for k in range(num_topics):
            topic_word[k] = topic_word[k] / np.sum(topic_word[k])
        return topic_word

    # phi: M*n_max*K
    def phi_init(self):
        corpus_len = self.corpus_len
        num_topics = self.num_topics
        corpus = self.corpus
        n_list = [len(x) for x in corpus]
        n_max = max(n_list)
        phi = np.zeros((corpus_len, n_max, num_topics))
        for m in range(corpus_len):
            n_per = n_list[m]
            temp = np.random.normal(0.5, 1, (n_per, num_topics))
            for n in range(n_per):
                phi[m][n] = np.exp(temp[n]) / np.sum(np.exp(temp[n]))
        return phi

    def gamma_init(self, phi):
        pi = self.pi
        gamma = pi + np.sum(phi, axis=1)
        return gamma

    def calculate_Bk(self, bkv, beta, k):
        # 计算E_q[log(\sum_r{bkr*betakr}+V*beta_bar)]
        beta_bar = self.beta_bar
        vocab_len = self.vocab_len
        b = bkv[k]
        be = beta[k]
        # 先用近似的做法，有待改进
        Bk = np.log(np.sum(b * be) + vocab_len * beta_bar)
        return Bk

    def calculate_deriv_Bk(self, bkv, beta, k, v):
        # 计算\partial{E_q[log(\sum_r{bkr*betakr}+V*beta_bar)]} {bkv}
        beta_bar = self.beta_bar
        vocab_len = self.vocab_len
        vid = list(set(list(range(vocab_len))) - set([v]))
        b = bkv[k, vid]
        be = beta[k, vid]
        # 先用近似的做法，有待改进
        dBkv = np.log(beta[k, v] + np.sum(b * be) + vocab_len * beta_bar) - np.log(np.sum(b * be) + vocab_len * beta_bar)
        return dBkv

    def update_phi(self, phi, bkv, beta, gamma):
        beta_bar = self.beta_bar
        corpus = self.corpus
        corpus_len = self.corpus_len
        num_topics = self.num_topics
        temp0 = beta * bkv + beta_bar
        for k in range(num_topics):
            temp0[k] = temp0[k] / np.sum(temp0[k])
        n_list = [len(x) for x in corpus]
        for m in range(corpus_len):
            n_per = n_list[m]
            wordidlist = [wordid for wordid, count in corpus[m]]
            temp1 = np.log(temp0[:, wordidlist])
            temp2 = psi(gamma[m])
            temp_phi = np.exp(temp1 + np.tile(temp2, (n_per, 1)).T).T  # n_per*K维
            # for n in range(n_per):
            #     phi[m, n] = temp_phi[n] / np.sum(temp_phi[n])
            phi[m, range(n_per)] = (temp_phi.T / temp_phi.sum(axis=1)).T
        return phi


    def update_gamma(self, phi):
        pi = self.pi
        gamma = pi + np.sum(phi, axis=1)
        return gamma

    def calculate_nkw(self, phi):
        corpus = self.corpus
        corpus_len = self.corpus_len
        num_topics = self.num_topics
        vocab_len = self.vocab_len
        nkw = np.zeros((num_topics, vocab_len))
        n_list = [len(x) for x in corpus]
        for m in range(corpus_len):
            n_per = n_list[m]
            wordidlist = [corpus[m][n][0] for n in range(n_per)]
            wordfreqlist = [corpus[m][n][1] for n in range(n_per)]
            nkw[:, wordidlist] += (phi[m, range(n_per), :] * np.tile(wordfreqlist, (num_topics, 1)).T).T
        return nkw

    def update_bkv(self, bkv, beta, x_hat, y_hat, nkw):
        beta_bar = self.beta_bar
        num_topics = self.num_topics
        vocab_len = self.vocab_len
        for k in range(num_topics):
            dBk = beta[k] / np.sum(bkv[k] * beta[k] + vocab_len * beta_bar)  # V维
            temp = nkw[k] * (np.log(beta[k] + beta_bar) - np.log(beta_bar) - dBk) + psi(x_hat[k]) - psi(y_hat[k])
            #         temp = nkw[k] * (np.log(beta[k] + beta_bar) - dBk) + psi(x_hat[k]) - psi(y_hat[k])
            temp[temp > 700] = 700
            bkv[k] = np.exp(temp) / (np.exp(temp) + 1)
        return bkv

    def update_xy(self, bkv):
        x = self.x
        y = self.y
        vocab_len = self.vocab_len
        x_hat = x + np.sum(bkv, axis=1)
        y_hat = y + vocab_len - np.sum(bkv, axis=1)
        return (x_hat, y_hat)

    def update_topic_word(self, topic_word, nkw):
        num_topics = self.num_topics
        for k in range(num_topics):
            topic_word[k] = nkw[k] / np.sum(nkw[k])
        return topic_word

    def update_beta(self, topic_word, bkv):
        beta = topic_word / bkv
        #     for k in range(num_topics):
        #         beta[k] = beta[k] / np.sum(beta[k])
        return beta

    def cal_ELBO(self, phi, nkw, gamma, bkv, x_hat, y_hat, beta):
        beta_bar = self.beta_bar
        x = self.x
        y = self.y
        pi = self.pi
        vocab_len = self.vocab_len
        num_topics = self.num_topics
        B = np.asarray([self.calculate_Bk(bkv, beta, k) for k in range(num_topics)])
        term1 = np.sum(nkw * (bkv * np.log(beta + beta_bar) + (1 - bkv) * np.log(beta_bar) - np.tile(B, (vocab_len, 1)).T))
        term2 = np.sum((np.sum(bkv, axis=1) + x - 1) * (psi(x_hat) - psi(x_hat + y_hat)) + (
                    vocab_len - np.sum(bkv, axis=1) + y - 1) * (psi(y_hat) - psi(x_hat + y_hat)))
        term3 = np.sum(
            (pi + np.sum(phi, axis=1) - 1) * (psi(gamma) - psi(np.tile(np.sum(gamma, axis=1), (num_topics, 1)).T)))
        term5 = - np.sum(phi[phi > 0] * np.log(phi[phi > 0]))
        term6 = - np.sum((gamma - 1) * psi(gamma)) + np.sum((np.sum(gamma, axis=1) - 1) * psi(np.sum(gamma, axis=1)))
        term7 = - np.sum(gammaln(np.sum(gamma, axis=1))) + np.sum(gammaln(gamma))
        term9 = - np.sum(bkv[bkv > 0] * np.log(bkv[bkv > 0])) - np.sum((1 - bkv[bkv < 1]) * np.log(1 - bkv[bkv < 1]))
        term11 = np.sum(betaln(x_hat, y_hat) - (x_hat - 1) * (psi(x_hat) - psi(x_hat + y_hat)) - (y_hat - 1) * (
                    psi(y_hat) - psi(x_hat + y_hat)))
        ELBO = term1 + term2 + term3 + term5 + term6 + term7 + term9 + term11
        return ELBO

    def cal_ELBO_beta(self, nkw, bkv, beta):
        beta_bar = self.beta_bar
        vocab_len = self.vocab_len
        num_topics = self.num_topics
        B = np.asarray([self.calculate_Bk(bkv, beta, k) for k in range(num_topics)])
        term1 = np.sum(nkw * (bkv * np.log(beta + beta_bar) + (1 - bkv) * np.log(beta_bar) - np.tile(B, (vocab_len, 1)).T))
        return term1

    def fit_model(self):
        num_topics = self.num_topics
        iters_EM_min = self.iters_EM_min
        iters_EM_max = self.iters_EM_max
        iters_E_min = self.iters_E_min
        iters_E_max = self.iters_E_max
        threshold_E = self.threshold_E
        threshold_EM = self.threshold_EM
        # update
        ELBO_EM = 0
        beta = self.beta_init()
        bkv = self.b_init()
        topic_word = self.topic_word_init(beta, bkv)
        for it in range(iters_EM_max):
            # initialize
            bkv = self.b_init()
            phi = self.phi_init()
            gamma = self.gamma_init(phi)
            x_hat = np.full(num_topics, 1)
            y_hat = np.full(num_topics, 10)  # 把y_hat改得很大，才能保证b能被估计出比较小的值。否则b永远>=0.5
            ELBO_E = 0
            ELBO_EM_old = ELBO_EM
            ## E-step
            for it_E in range(iters_E_max):
                T0 = time.time()
                ELBO_E_old = ELBO_E
                phi = self.update_phi(phi, bkv, beta, gamma)
                T1 = time.time()
                gamma = self.update_gamma(phi)
                T2 = time.time()
                nkw = self.calculate_nkw(phi)
                T3 = time.time()
                bkv = self.update_bkv(bkv, beta, x_hat, y_hat, nkw)
                T4 = time.time()
                #         x_hat, y_hat = self.update_xy(bkv)
                ELBO_E = self.cal_ELBO(phi, nkw, gamma, bkv, x_hat, y_hat, beta)
                T5 = time.time()
                if (ELBO_E - ELBO_E_old <= threshold_E and it_E >= iters_E_min - 1) or (it_E == iters_E_max - 1):
                    break

            ELBO_EM = ELBO_E
            term1_old = self.cal_ELBO_beta(nkw, bkv, beta)
            ## M-step
            topic_word = self.update_topic_word(topic_word, nkw)
            beta = self.update_beta(topic_word, bkv)

            term1 = self.cal_ELBO_beta(nkw, bkv, beta)
            ELBO_EM += term1 - term1_old
            print('iter', it, 'completed!', 'ELBO:', round(ELBO_EM, 4))
            # 判断收敛
            if (ELBO_EM - ELBO_EM_old <= threshold_EM and it >= iters_EM_min - 1) or (it == iters_EM_max - 1):
                break

        self.beta = beta
        self.bkv = bkv
        self.topic_word = topic_word
        self.nkw = nkw
        self.phi = phi
        self.gamma = gamma

    def get_theta(self):
        gamma = self.gamma
        corpus_len = self.corpus_len
        num_topics = self.num_topics
        theta = np.zeros((corpus_len, num_topics))
        for m in range(corpus_len):
            theta[m] = gamma[m] / np.sum(gamma[m])
        return theta