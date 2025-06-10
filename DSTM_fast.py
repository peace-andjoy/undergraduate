# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 11:18:10 2021

@author: zhourui
"""

from gensim import utils,matutils
from scipy.special import gammaln
import numpy as np
import pandas as pd
import mpmath
import time

#生成模拟数据
def get_DSTM_data(nTerms = 100,M = 20,K = 3,N = 30,s = 2,v=2,x=2,y=2,seed =1234):
    """
    参数说明：
    *nTerms：单词表的总词数
    *M:文档数
    *K:主题数
    *ND:每篇文档词数
    *s,v,x,y:主题以及词稀疏Beta分布的参数
    """
    alpha = 0.1
    alpha_bar = 1e-12
    beta = 0.1
    beta_bar = 1e-12

    Nd=np.full(shape=[M],fill_value=N,dtype=np.int)
    Alpha = np.full(K,alpha)
    Alp = np.full(K,alpha_bar)
    Beta = np.full(nTerms,beta)
    Be = np.full(nTerms,beta_bar)
    np.random.seed(seed)
    
    #pai--beta分布得到的各主题表达的概率，维数M
    pai = np.random.beta(s,v,M)
    pai = [pai[i].repeat(K) for i in range(0, len(pai))]
    
    #gamma--beta分布得到的所有词表达的概率，维数K
    gamma = np.random.beta(x,y,K)
    gamma = [gamma[i].repeat(nTerms) for i in range(0, len(gamma))]
    
    #a--主题稀疏参数,维数M*K
    a = np.random.binomial(1,pai)
    
    #b--词稀疏参数,维数K*nTerms
    b = np.random.binomial(1,gamma)

    
    #alpha平滑--dir先验参数
    Alpha_final = [(Alpha*a[i]+Alp) for i in range(M)]
    
    #Theta[d]--第d篇文档的主题分布
    Theta = [np.random.dirichlet(Alpha_final[i]) for i in range(M)]
    
    #beta平滑--dir先验参数
    Beta_final = [(Beta*b[i]+Be)for i in range(K)]
    
    #Phi[k]--第k个主题的词分布
    Phi = [np.random.dirichlet(Beta_final[i]) for i in range(K)]
    
    
    #z_[d][n]--第d篇文档第n个词的主题
    z = np.zeros((M,N,K))
    z_ = np.zeros((M,N))
    K_topic=range(K)
    for d in range(M):
        for n in range(Nd[d]):
            try:
                z[d][n] = np.random.multinomial(1, Theta[d])
            except:
                Theta[d] = np.repeat(1/K, K)
                z[d][n] = np.random.multinomial(1, Theta[d])
            z_[d][n] =np.dot(z[d][n],K_topic)
    
    #w_[d][n]--第d篇文档第n个词的内容
    w = np.zeros((M,N,nTerms))
    w_ = np.zeros((M,N))
    for d in range(M):
        for n in range(Nd[d]):
            z_dn = int(z_[d][n])
            w[d][n] = np.random.multinomial(1,Phi[z_dn])
            w_[d][n] = int(np.dot(w[d][n],range(nTerms)))
    d = dict()
    d['pi']=pai
    d['gamma'] = gamma
    d['a']=a
    d['b']=b
    d['phi']=Phi
    d['theta']=Theta
    d['z']=z_
    d['w']=w_
    d['alpha']=Alpha_final
    d['beta']=Beta_final
    return(d)

#将生成的数据格式转化为主题模型可以读入的形式
def get_corpus(data):
    words=np.array(np.array(data['w'],dtype=np.int),dtype=np.str)
    words_series = pd.DataFrame()
    word_series = pd.Series()
    words1=[str(doc) for doc in words]
    words2 = pd.Series(words1)
    words3 = pd.Series([words2[i].replace("\n","").replace(" ",",").replace("\'","").replace('[',"").replace(']',"") for i in range(len(words2))])
    word_series.reset_index(inplace=True,drop=True)
    words_series['word']=words3
    words_series['words']=[words_series['word'][i].split(',') for i in range(len(words_series))]
    return(words_series)

def betaln(a,b):
#计算log Beta(a,b)，为后面公式更新准备
    betaln = gammaln(a)+gammaln(b)-gammaln(a+b)
    return(betaln)

class DSTM_para(utils.SaveLoad):
    def __init__(self,corpus=None,id2word=None,
                 alphas=0.1,num_topics=10,initialize='gensim',
                 alpha_bar=1e-5,beta=0.1,beta_bar =1e-5,s=1,v=1,x=1,y=1,iterations=100,seed=123):
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
        vocab_len =self.vocab_len
        self.num_topics = num_topics
        self.alphas = np.full(num_topics, alphas)
        self.alpha_smooth = np.full(num_topics,alpha_bar)
        self.beta = np.full(vocab_len, beta)
        self.beta_smooth = np.full(vocab_len,beta_bar)
        self.iterations = iterations
        self.seed=seed
  
        self.s = s
        self.v = v
        self.x = x
        self.y = y
        
    def a_init(self):
    #初始化主题稀疏参数amk,维数M*K；amk_sum是第m篇文档所有主题表达概率加和，用于amk的更新
        M = self.corpus_len
        num_topics = self.num_topics
        s = self.s
        v = self.v
        amk_sum = np.zeros((M))
        amk = np.random.beta(s,v,(M,num_topics))
        amk_sum = np.sum(amk, axis = 1)
        return(amk,amk_sum)
    def b_init(self):
    #初始化词稀疏参数bkv,维数K*vocab_len；bkv_sum是第k个主题所有词表达概率加和，用于bkv的更新
        vocab_len = self.vocab_len
        num_topics= self.num_topics
        x = self.x
        y = self.y
    
        bkv_sum = np.zeros((num_topics))
        bkv = np.random.beta(x,y,(num_topics,vocab_len))
        for k in range(num_topics):
            bkv_sum[k] =np.sum(bkv[k])
        return(bkv,bkv_sum)

    def gamma_init(self):
    #初始化文档-主题比例参数gamma,维数M*n_max*K
        corpus = self.corpus
        corpus_len = self.corpus_len
        vocab_len = self.vocab_len
        num_topics = self.num_topics
        n_list = [len(x) for x in corpus]
        n_max = max(n_list)#所有文档中出现最多不同编号词的数量,比如三篇文档分别有5,6,7个不同编号的词，我们的gamma维数为3*7*K
        gamma_word =np.full(shape=[corpus_len,n_max,num_topics],fill_value=0,dtype="float")
        nkw = np.full(shape=[vocab_len,num_topics],fill_value=0,dtype="float")#对应双稀疏Nkr
        nkw_sum = np.full(shape=[num_topics],fill_value=0,dtype="float")#对应双稀疏Nk
        nmk = np.full(shape=[corpus_len,num_topics],fill_value=0,dtype="float")#对应双稀疏Njk
        nmk_sum = np.full(shape=[corpus_len],fill_value=0,dtype="float")#对应双稀疏Nm
        for m in range(corpus_len):
            n_per = n_list[m]
            for n in range(n_per):
                temp = np.random.normal(0.5,0.5,num_topics)
                gamma_word[m][n] = np.exp(temp)/sum(np.exp(temp))
                word_id = corpus[m][n][0]
                word_freq = corpus[m][n][1]
                nmk[m] += gamma_word[m][n]*word_freq
                nmk_sum[m] += np.sum(gamma_word[m][n])*word_freq
                nkw[word_id] += gamma_word[m][n]*word_freq
                nkw_sum += gamma_word[m][n]*word_freq
        return(gamma_word,nmk,nmk_sum,nkw,nkw_sum)

    def amk_update(self,amk,amk_sum,nmk,nmk_sum):
    #更新amk
        num_topics = self.num_topics
        corpus_len = self.corpus_len
        s = self.s
        v = self.v
        alpha = self.alphas[0]
        alpha_bar = self.alpha_smooth[0]
        log_a1 = np.zeros((corpus_len,num_topics))
        log_a0 = np.zeros((corpus_len,num_topics))
        
        for m in range(corpus_len):
            prev_a = np.zeros(num_topics)
            for k in range(num_topics):
                prev_a[k] = amk[m][k]
                Am = amk_sum[m] - amk[m][k]
                #nmk为第m篇文档主题k下所有词的gamma估计值之和，nmk_sum为第m篇文档下所有词的gamma估计值之和
                log_a1[m][k] = np.log(s + Am) + gammaln(nmk[m][k] + alpha + alpha_bar) + betaln(alpha + alpha * Am + num_topics * alpha_bar,nmk_sum[m] + alpha * Am + num_topics * alpha_bar)
                log_a0[m][k] = np.log(v + num_topics - 1.0 - Am) + gammaln(alpha + alpha_bar) + betaln(alpha * Am + num_topics * alpha_bar,nmk_sum[m] + alpha * Am + alpha + num_topics * alpha_bar)
                amk[m][k] = float(mpmath.exp(log_a1[m][k]) / (mpmath.exp(log_a1[m][k]) + mpmath.exp(log_a0[m][k])))
            #for k in range(num_topics):
                #if np.exp(log_a1[m][k])>np.exp(700):
                #    amk[m][k] = np.exp(700)/(np.exp(700)+np.exp(log_a0[m][k]))
                #else:
                #   amk[m][k] = np.exp(log_a1[m][k])/(np.exp(log_a1[m][k])+np.exp(log_a0[m][k]))

            for k in range(num_topics):
                amk_sum[m] += amk[m][k] - prev_a[k]
            return(log_a1,log_a0,amk,amk_sum)
    def bkv_update(self,bkv,bkv_sum,nkw,nkw_sum):
        num_topics = self.num_topics
        vocab_len = self.vocab_len
        x = self.x
        y = self.y
        beta = self.beta[0]
        beta_bar = self.beta_smooth[0]
        #更新bkv
        log_b1 = np.zeros((num_topics,vocab_len))
        log_b0 = np.zeros((num_topics,vocab_len))

        for k in range(num_topics):
            Bk = bkv_sum[k]-bkv[k]
            prev_b = bkv[k].copy()
            log_b1[k] = np.log(x + Bk) + gammaln(nkw[:,k] + beta + beta_bar) + betaln(
                beta + beta * Bk + vocab_len * beta_bar, nkw_sum[k] + beta * Bk + vocab_len * beta_bar)
            log_b0[k] = np.log(y + vocab_len - 1.0 - Bk) + gammaln(beta + beta_bar) + betaln(
                beta * Bk + vocab_len * beta_bar, nkw_sum[k] + beta * Bk + beta + vocab_len * beta_bar)
            bkv[k] = np.exp(log_b1[k]) / (np.exp(log_b1)[k] + np.exp(log_b0[k]))
            if np.isnan(bkv[k]).any() or np.isinf(bkv[k]).any():
                for v in range(vocab_len):
                    #nkw为主题k第w个词的gamma估计值之和，nkw_sum为主题k下所有词的gamma估计值之和
                    bkv[k][v] = float(mpmath.exp(log_b1[k][v])/(mpmath.exp(log_b1[k][v])+mpmath.exp(log_b0[k][v])))
            # for v in range(vocab_len):
                # if np.exp(log_b1[k][v])>np.exp(700):
                #     bkv[k][v] = np.exp(700)/(np.exp(700)+np.exp(log_b0[k][v]))
                # else:
                #     bkv[k][v] = np.exp(log_b1[k][v])/(np.exp(log_b1[k][v])+np.exp(log_b0[k][v]))
            for v in range(vocab_len):
                bkv_sum[k] += bkv[k][v] - prev_b[v]
        return(log_b1,log_b0,bkv,bkv_sum)

    def gamma_update(self,gamma_word,nkw,nkw_sum,nmk,nmk_sum,amk,amk_sum,bkv,bkv_sum):
        corpus = self.corpus
        corpus_len = self.corpus_len
        vocab_len = self.vocab_len
        num_topics = self.num_topics
        alpha = self.alphas[0]
        alpha_bar = self.alpha_smooth[0]
        beta = self.beta[0]
        beta_bar = self.beta_smooth[0]
        n_list = [len(x) for x in corpus]
        prev_gamma_word = gamma_word.copy()
        for m in range(corpus_len):
            n_per = n_list[m]
            for n in range(n_per):
                word_id = corpus[m][n][0]
                word_freq = corpus[m][n][1]
                # 初始化临时参数
                gamma_word_temp = np.full(shape=[num_topics], fill_value=0, dtype="float")
                temp1 = nkw[word_id] - gamma_word[m][n]
                temp2 = nmk[m] - gamma_word[m][n]
                temp3 = nkw_sum - gamma_word[m][n]
                gamma_word_temp = (temp1 + beta * bkv[:, word_id] + beta_bar) * (temp2 + amk[m] * alpha + alpha_bar) / (
                            temp3 + beta * bkv_sum + vocab_len * beta_bar)
                gamma_word[m][n] = gamma_word_temp / sum(gamma_word_temp)
                nkw[word_id] += (gamma_word[m][n] - prev_gamma_word[m][n]) * word_freq
                nmk[m] += (gamma_word[m][n] - prev_gamma_word[m][n]) * word_freq
                nkw_sum += (gamma_word[m][n] - prev_gamma_word[m][n]) * word_freq
                nmk_sum[m] += np.sum(gamma_word[m][n] - prev_gamma_word[m][n]) * word_freq
        return(gamma_word,nmk,nmk_sum,nkw,nkw_sum)
    def fit_DSTM_model(self,iterations):
        np.random.seed(self.seed)
        amk,amk_sum=self.a_init()
        bkv,bkv_sum = self.b_init()
        gamma_word,nmk,nmk_sum,nkw,nkw_sum=self.gamma_init()
        for iters in range(iterations):
            print('iteration' + str(iters))
            t1 = time.time()
            log_a1,log_a0,amk,amk_sum = self.amk_update(amk,amk_sum,nmk,nmk_sum)
            t2 = time.time()
            # print('更新a,', t2-t1)
            log_b1,log_b0,bkv,bkv_sum=self.bkv_update(bkv,bkv_sum,nkw,nkw_sum)
            t3 = time.time()
            # print('更新b,', t3-t2)
            gamma_word,nmk,nmk_sum,nkw,nkw_sum=self.gamma_update(gamma_word,nkw,nkw_sum,nmk,nmk_sum,amk,amk_sum,bkv,bkv_sum)
            t4 = time.time()
            # print('更新gamma,',t4-t3)
        self.amk = amk
        self.amk_sum = amk_sum
        self.bkv = bkv
        self.bkv_sum = bkv_sum
        self.gamma_word = gamma_word
        self.nmk = nmk
        self.nmk_sum = nmk_sum
        self.nkw = nkw
        self.nkw_sum = nkw_sum
        #print('amk:',amk)
        #print('bkv:',bkv)
        #print('gamma_word:',gamma_word)
        return(gamma_word,amk,bkv)
    def estimate_theta(self):
        alpha = self.alphas[0]
        alpha_bar = self.alpha_smooth[0]
        corpus_len = self.corpus_len
        num_topics = self.num_topics
        theta = np.zeros((corpus_len,num_topics))
        amk = self.amk
        amk_sum = self.amk_sum
        nmk = self.nmk
        nmk_sum = self.nmk_sum
        for m in range(corpus_len):
            for k in range(num_topics):
                theta[m][k] = (nmk[m][k] + amk[m][k]*alpha+ alpha_bar) / (nmk_sum[m]+alpha*amk_sum[m]+num_topics* alpha_bar)
        return(theta)
    def estimate_Phi(self):
        beta = self.beta[0]
        beta_bar = self.beta_smooth[0]
        vocab_len = self.vocab_len
        num_topics = self.num_topics
        bkv = self.bkv
        bkv_sum = self.bkv_sum
        nkw = self.nkw
        nkw_sum = self.nkw_sum
        Phi = np.zeros((num_topics,vocab_len))
        for k in range(num_topics):
            for v in range(vocab_len):
                Phi[k][v] = (nkw[v][k]+bkv[k][v]*beta+beta_bar)/(nkw_sum[k]+beta*bkv_sum[k]+vocab_len*beta_bar)
        self.Phi = Phi
        return(Phi)
    def print_document_topic(self,theta,Phi):
        """Get the information needed to visualize the corpus model , using the pyLDAvis format.

        Parameters
        ----------
        corpus : {iterable of list of (int, float), scipy.sparse.csc}, optional
            The corpus we want to visualize at the given time slice.

        Returns
        -------
        doc_topics : list of length `self.num_topics`
            Probability for each topic in the mixture (essentially a point in the `self.num_topics - 1` simplex.
        topic_term : numpy.ndarray
            The representation of each topic as a multinomial over words in the vocabulary,
            expected shape (`num_topics`, vocabulary length).
        doc_lengths : list of int
            The number of words in each document. These could be fixed, or drawn from a Poisson distribution.
        term_frequency : numpy.ndarray
            The term frequency matrix (denoted as beta in the original Blei paper). This could also be the TF-IDF
            representation of the corpus, expected shape (number of documents, length of vocabulary).
        vocab : list of str
            The set of unique terms existing in the cropuse's vocabulary.

        """
        doc_topic = theta 
        K = self.num_topics

        def normalize(x):
            return x / x.sum()

        topic_term = [normalize(np.exp(Phi[k])) for k in range(K)]

        doc_lengths = []
        term_frequency = np.zeros(self.vocab_len)
        for doc_no, doc in enumerate(self.corpus):
            doc_lengths.append(len(doc))

            for term, freq in doc:
                term_frequency[term] += freq

        vocab = [self.id2word[i] for i in range(len(self.id2word))]

        return doc_topic, np.array(topic_term), doc_lengths, term_frequency, vocab
    
    def print_topics(self,top_terms=20):
        """Get the most relevant words for every topic.

        Parameters
        ----------
        top_terms : int, optional
            Number of most relevant words to be returned for each topic.

        Returns
        -------
        list of list of (str, float)
            Representation of all topics. Each of them is represented by a list of pairs of words and their assigned
            probability.

        """
        return [self.print_topic(topic, top_terms) for topic in range(self.num_topics)]

    def print_topic(self, topic, top_terms=20):
        """Get the list of words most relevant to the given topic.

        Parameters
        ----------
        topic : int
            The index of the topic to be inspected.
     
        top_terms : int, optional
            Number of words associated with the topic to be returned.

        Returns
        -------
        list of (str, float)
            The representation of this topic. Each element in the list includes the word itself, along with the
            probability assigned to it by the topic.

        """
        topic = self.Phi[topic]
        topic = np.transpose(topic)
        topic = np.exp(topic)
        topic = topic / topic.sum()
        bestn = matutils.argsort(topic, top_terms, reverse=True)
        beststr = [(self.id2word[id_], topic[id_]) for id_ in bestn]
        return beststr
    
    def dstm_coherence(self):
        """Get the coherence for each topic.

        Can be used to measure the quality of the model, or to inspect the convergence through training via a callback.

        Parameters
        ----------
        time : int
            The time slice.

        Returns
        -------
        list of list of str
            The word representation for each topic, for each time slice. This can be used to check the time coherence
            of topics as time evolves: If the most relevant words remain the same then the topic has somehow
            converged or is relatively static, if they change rapidly the topic is evolving.

        """
        coherence_topics = []
        for topics in self.print_topics():
            coherence_topic = []
            for word, dist in topics:
                coherence_topic.append(word)
            coherence_topics.append(coherence_topic)

        return coherence_topics
