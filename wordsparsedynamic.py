#!/usr/bin/env python
# coding: utf-8

# In[73]:


from gensim import utils,matutils
from gensim.models import ldamodel
from scipy.special import gammaln, psi, digamma
from scipy import optimize
from six.moves import range, zip
import numpy as np
import pandas as pd
import mpmath
import math
import gensim
from gensim.corpora import Dictionary
from gensim.test.utils import datapath
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamodel import CoherenceModel
from gensim.models import LdaSeqModel
import pickle
import sys
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from itertools import permutations, combinations
import logging
logger = logging.getLogger(__name__)
from wordsparsestatic import wordsparseTM


# In[2]:


#生成模拟数据
def get_WSTM_data(nTerms = 100,T = 3, M=100,K = 3,N = 75,x=2,y=2,mean = 0.5, sd = 0.5, seed =1234,seed2=1234):
    """
    参数说明：
    *nTerms：单词表的总词数
    *M:文档数
    *K:主题数
    *ND:每篇文档词数
    *s,v,x,y:主题以及词稀疏Beta分布的参数
    """
    np.random.seed(seed)
    pi = 1/K
    beta_bar = 1e-7

    Mt=np.full(shape=[T], fill_value = M, dtype = int)
    M = np.sum(Mt)   #文档总数
    Nd=np.full(shape=[M],fill_value=N,dtype=int)
    Pi = np.full(K,pi)

    Beta = np.zeros((T, K, nTerms))
    Beta[0] = np.random.normal(np.full(shape = [K, nTerms], fill_value = mean), sd)
    for t in range(1, T):
        Beta[t] = np.random.normal(Beta[t-1],sd)

    #gamma--beta分布得到的所有词表达的概率，维数T*K
    p2 = np.random.beta(x,y,[T, K])

    #b--词稀疏参数,维数T*K*nTerms
    b = np.zeros((T, K, nTerms), dtype = int)
    for w in range(nTerms):
        b[:, :, w] = np.random.binomial(1, p2)

    #pi平滑--dir先验参数 M*K维
    Pi_final = [Pi for i in range(M)]

    #Theta[d]--第d篇文档的主题分布
    Theta = [np.random.dirichlet(Pi_final[i]) for i in range(M)]

    #Phi[k]--第k个主题的词分布
    Phi = np.exp(Beta)*b+beta_bar
    for t in range(T):
        for k in range(K):
            Phi[t, k, :] = Phi[t, k, :]/np.sum(Phi[t, k, :])

    np.random.seed(seed2)
    #z_[d][n]--第d篇文档第n个词的主题
    z = np.zeros((M,N,K))
    z_ = np.zeros((M,N))
    K_topic=range(K)
    for d in range(M):
        for n in range(Nd[d]):
            z[d][n] = np.random.multinomial(1, Theta[d])
            z_[d][n] =np.dot(z[d][n],K_topic)

    #w_[d][n]--第d篇文档第n个词的内容
    w = np.zeros((M,N,nTerms))
    w_ = np.zeros((M,N))
    for t in range(T):
        if t == 0:
            d_slice = range(Mt[t])
        else:
            d_slice = range(np.cumsum(Mt)[t-1],np.cumsum(Mt)[t])

        for d in d_slice:
            for n in range(Nd[d]):
                z_dn = int(z_[d][n])
                w[d][n] = np.random.multinomial(1,Phi[t, z_dn])
                w_[d][n] = int(np.dot(w[d][n],range(nTerms)))

    d = dict()
    d['p2'] = p2
    d['b']=b
    d['phi']=Phi
    d['theta']=Theta
    d['z']=z_
    d['w']=w_
    d['pi']=Pi_final
    d['beta']=Beta
    return d


# In[3]:

#生成模拟数据
def get_data_sp(nTerms = 100,T = 3, M=100,K = 3,N = 75,s = 2,v=2,x=2,y=2, mean = 0.5, sd = 0.5, seed =1234):
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

    Mt=np.full(shape=[T], fill_value = M, dtype = int)
    M = np.sum(Mt)   #文档总数
    Nd=np.full(shape=[M],fill_value=N,dtype=int)
    Pi = np.full(K,pi)

    Beta = np.zeros((T, K, nTerms))
    Beta[0] = np.random.normal(np.full(shape = [K, nTerms], fill_value = mean), sd)
    for t in range(1, T):
        Beta[t] = np.random.normal(Beta[t-1],sd)

    #gamma--beta分布得到的所有词表达的概率，维数T*K
    p2 = np.random.beta(x,y,[T, K])

    # b--词稀疏参数,维数T*K*nTerms
    b = np.zeros((T, K, nTerms), dtype = int)
    i = int(nTerms/K)
    for t in range(T):
        for k in range(K):
            b[t, k, range(i * k, i * (k+1))] = np.repeat(1, i)

    #pi平滑--dir先验参数 M*K维
    Pi_final = [Pi for i in range(M)]

    #Theta[d]--第d篇文档的主题分布
    Theta = [np.random.dirichlet(Pi_final[i]) for i in range(M)]

    #Phi[k]--第k个主题的词分布
    Phi = np.exp(Beta)*b+beta_bar
    for t in range(T):
        for k in range(K):
            Phi[t, k, :] = Phi[t, k, :]/np.sum(Phi[t, k, :])

    #z_[d][n]--第d篇文档第n个词的主题
    z = np.zeros((M,N,K))
    z_ = np.zeros((M,N))
    K_topic=range(K)
    for d in range(M):
        for n in range(Nd[d]):
            z[d][n] = np.random.multinomial(1, Theta[d])
            z_[d][n] =np.dot(z[d][n],K_topic)

    #w_[d][n]--第d篇文档第n个词的内容
    w = np.zeros((M,N,nTerms))
    w_ = np.zeros((M,N))
    nkw = np.zeros((T, nTerms, K))
    for t in range(T):
        if t == 0:
            d_slice = range(Mt[t])
        else:
            d_slice = range(np.cumsum(Mt)[t-1],np.cumsum(Mt)[t])

        for d in d_slice:
            for n in range(Nd[d]):
                z_dn = int(z_[d][n])
                w[d][n] = np.random.multinomial(1,Phi[t, z_dn])
                w_[d][n] = int(np.dot(w[d][n],range(nTerms)))
                nkw[t, int(w_[d][n]), z_dn] += 1

    d = dict()
    d['p2'] = p2
    d['b']=b
    d['phi']=Phi
    d['theta']=Theta
    d['z']=z_
    d['w']=w_
    d['pi']=Pi_final
    d['beta']=Beta
    d['nkw'] = nkw
    return d

#将生成的数据格式转化为主题模型可以读入的形式
def get_corpus(data):
    words=np.array(np.array(data['w'],dtype=int),dtype=str)
    words_series = pd.DataFrame()
    word_series = pd.Series(dtype=object)
    words1=[str(doc) for doc in words]
    words2 = pd.Series(words1)
    words3 = pd.Series([words2[i].replace("\n","").replace(" ",",").replace("\'","").replace('[',"").replace(']',"") for i in range(len(words2))])
    word_series.reset_index(inplace=True,drop=True)
    words_series['word']=words3
    words_series['words']=[words_series['word'][i].split(',') for i in range(len(words_series))]
    return(words_series)

def betaln(a, b):
    # 计算log Beta(a,b)，为后面公式更新准备
    betaln = gammaln(a) + gammaln(b) - gammaln(a + b)
    return (betaln)

class sslm(utils.SaveLoad):
    """Encapsulate the inner State Space Language Model for DTM.

    Some important attributes of this class:

        * `obs` is a matrix containing the document to topic ratios.
        * `e_log_prob` is a matrix containing the topic to word ratios.
        * `mean` contains the mean values to be used for inference for each word for a time slice.
        * `variance` contains the variance values to be used for inference of word in a time slice.
        * `fwd_mean` and`fwd_variance` are the forward posterior values for the mean and the variance.
        * `zeta` is an extra variational parameter with a value for each time slice.
        * `bvt` is an matrix with an expected shape (`self.vocab_len`, `num_time_slices`).
    """

    def __init__(self, vocab_len=None, num_time_slices=None, num_topics=None, bvt = None, obs_variance=0.5, chain_variance=0.005, beta_bar = 1e-5):
        self.vocab_len = vocab_len
        self.num_time_slices = num_time_slices
        self.obs_variance = obs_variance
        self.chain_variance = chain_variance
        self.num_topics = num_topics
        bvt[bvt<1e-5] = 1e-5
        self.bvt = bvt
        self.beta_bar = beta_bar

        # setting up matrices
        self.obs = np.zeros((vocab_len, num_time_slices))
        self.e_log_prob = np.zeros((vocab_len, num_time_slices))
        self.mean = np.zeros((vocab_len, num_time_slices + 1))
        self.fwd_mean = np.zeros((vocab_len, num_time_slices + 1))
        self.fwd_variance = np.zeros((vocab_len, num_time_slices + 1))
        self.variance = np.zeros((vocab_len, num_time_slices + 1))
        self.zeta = np.zeros(num_time_slices)

        # the following are class variables which are to be integrated during Document Influence Model
        self.m_update_coeff = None
        self.mean_t = None
        self.variance_t = None
        self.influence_sum_lgl = None
        self.w_phi_l = None
        self.w_phi_sum = None
        self.w_phi_l_sq = None
        self.m_update_coeff_g = None

    def update_zeta(self):
        """Update the Zeta variational parameter.

        Zeta is described in the appendix and is equal to sum (exp(mean[word] + Variance[word] / 2)),
        over every time-slice. It is the value of variational parameter zeta which maximizes the lower bound.

        Returns
        -------
        list of float
            The updated zeta values for each time slice.

        """
        for j, val in enumerate(self.zeta):
#             self.zeta[j] = np.sum(np.exp(self.mean[:, j + 1] + self.variance[:, j + 1] / 2))
            self.zeta[j] = np.sum(self.bvt[:, j] * np.exp(self.mean[:, j + 1] + self.variance[:, j + 1] / 2))
        return self.zeta

    def compute_post_variance(self, word, chain_variance):
        r"""Get the variance, based on the `Variational Kalman Filtering approach for Approximate Inference (section 3.1)
        <https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.

        This function accepts the word to compute variance for, along with the associated sslm class object,
        and returns the `variance` and the posterior approximation `fwd_variance`.

        Notes
        -----
        This function essentially computes Var[\beta_{t,w}] for t = 1:T

        .. :math::

            fwd\_variance[t] \equiv E((beta_{t,w}-mean_{t,w})^2 |beta_{t}\ for\ 1:t) =
            (obs\_variance / fwd\_variance[t - 1] + chain\_variance + obs\_variance ) *
            (fwd\_variance[t - 1] + obs\_variance)

        .. :math::

            variance[t] \equiv E((beta_{t,w}-mean\_cap_{t,w})^2 |beta\_cap_{t}\ for\ 1:t) =
            fwd\_variance[t - 1] + (fwd\_variance[t - 1] / fwd\_variance[t - 1] + obs\_variance)^2 *
            (variance[t - 1] - (fwd\_variance[t-1] + obs\_variance))

        Parameters
        ----------
        word: int
            The word's ID.
        chain_variance : float
            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The first returned value is the variance of each word in each time slice, the second value is the
            inferred posterior variance for the same pairs.

        """
        INIT_VARIANCE_CONST = 1000

        T = self.num_time_slices
        variance = self.variance[word]
        fwd_variance = self.fwd_variance[word]
        # forward pass. Set initial variance very high
        fwd_variance[0] = chain_variance * INIT_VARIANCE_CONST
        for t in range(1, T + 1):
            if self.obs_variance:
                c = self.obs_variance / (fwd_variance[t - 1] + chain_variance + self.obs_variance)
            else:
                c = 0
            fwd_variance[t] = c * (fwd_variance[t - 1] + chain_variance)

        # backward pass
        variance[T] = fwd_variance[T]
        for t in range(T - 1, -1, -1):
            if fwd_variance[t] > 0.0:
                c = np.power((fwd_variance[t] / (fwd_variance[t] + chain_variance)), 2)
            else:
                c = 0
            variance[t] = (c * (variance[t + 1] - chain_variance)) + ((1 - c) * fwd_variance[t])

        return variance, fwd_variance

    def compute_post_mean(self, word, chain_variance):
        """Get the mean, based on the `Variational Kalman Filtering approach for Approximate Inference (section 3.1)
        <https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.

        Notes
        -----
        This function essentially computes E[\beta_{t,w}] for t = 1:T.

        .. :math::

            Fwd_Mean(t) ≡  E(beta_{t,w} | beta_ˆ 1:t )
            = (obs_variance / fwd_variance[t - 1] + chain_variance + obs_variance ) * fwd_mean[t - 1] +
            (1 - (obs_variance / fwd_variance[t - 1] + chain_variance + obs_variance)) * beta

        .. :math::

            Mean(t) ≡ E(beta_{t,w} | beta_ˆ 1:T )
            = fwd_mean[t - 1] + (obs_variance / fwd_variance[t - 1] + obs_variance) +
            (1 - obs_variance / fwd_variance[t - 1] + obs_variance)) * mean[t]

        Parameters
        ----------
        word: int
            The word's ID.
        chain_variance : float
            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The first returned value is the mean of each word in each time slice, the second value is the
            inferred posterior mean for the same pairs.

        """
        T = self.num_time_slices
        obs = self.obs[word]
        fwd_variance = self.fwd_variance[word]
        mean = self.mean[word]
        fwd_mean = self.fwd_mean[word]

        # forward
        fwd_mean[0] = 0
        for t in range(1, T + 1):
            c = self.obs_variance / (fwd_variance[t - 1] + chain_variance + self.obs_variance)
            fwd_mean[t] = c * fwd_mean[t - 1] + (1 - c) * obs[t - 1]

        # backward pass
        mean[T] = fwd_mean[T]
        for t in range(T - 1, -1, -1):
            if chain_variance == 0.0:
                c = 0.0
            else:
                c = chain_variance / (fwd_variance[t] + chain_variance)
            mean[t] = c * fwd_mean[t] + (1 - c) * mean[t + 1]
        return mean, fwd_mean

    def compute_expected_log_prob(self):
        """Compute the expected log probability given values of m.

        The appendix describes the Expectation of log-probabilities in equation 5 of the DTM paper;
        The below implementation is the result of solving the equation and is implemented as in the original
        Blei DTM code.

        Returns
        -------
        numpy.ndarray of float
            The expected value for the log probabilities for each word and time slice.

        """
        for (w, t), val in np.ndenumerate(self.e_log_prob):
            self.e_log_prob[w][t] = np.log(self.bvt[w][t]) + self.mean[w][t + 1] - np.log(self.zeta[t])
#             self.e_log_prob[w][t] = self.bvt[w][t] * self.mean[w][t + 1] + (1 - self.bvt[w][t]) * np.log(self.beta_bar) \
#                                     - np.log(self.zeta[t])
#             self.e_log_prob[w][t] = np.log(self.bvt[w][t] * np.exp(self.mean[w][t + 1] + self.variance[w][t + 1]/2) \
#                                            + self.beta_bar) - np.log(self.zeta[t])
        return self.e_log_prob

    def sslm_counts_init(self, obs_variance, chain_variance, sstats):
        """Initialize the State Space Language Model with LDA sufficient statistics.

        Called for each topic-chain and initializes initial mean, variance and Topic-Word probabilities
        for the first time-slice.

        Parameters
        ----------
        obs_variance : float, optional
            Observed variance used to approximate the true and forward variance.
        chain_variance : float
            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.
        sstats : numpy.ndarray
            Sufficient statistics of the LDA model. Corresponds to matrix beta in the linked paper for time slice 0,
            expected shape (`self.vocab_len`, `num_topics`).

        """
        W = self.vocab_len
        T = self.num_time_slices

        log_norm_counts = np.copy(sstats)
        log_norm_counts /= sum(log_norm_counts)
        log_norm_counts += 1.0 / W
        log_norm_counts /= sum(log_norm_counts)
        log_norm_counts = np.log(log_norm_counts)

        # setting variational observations to transformed counts
        self.obs = (np.repeat(log_norm_counts, T, axis=0)).reshape(W, T)
        # set variational parameters
        self.obs_variance = obs_variance
        self.chain_variance = chain_variance

        # compute post variance, mean
        for w in range(W):
            self.variance[w], self.fwd_variance[w] = self.compute_post_variance(w, self.chain_variance)
            self.mean[w], self.fwd_mean[w] = self.compute_post_mean(w, self.chain_variance)

        self.zeta = self.update_zeta()
        self.e_log_prob = self.compute_expected_log_prob()

    def fit_sslm(self, sstats):
        """Fits variational distribution.

        This is essentially the m-step.
        Maximizes the approximation of the true posterior for a particular topic using the provided sufficient
        statistics. Updates the values using :meth:`~gensim.models.ldaseqmodel.sslm.update_obs` and
        :meth:`~gensim.models.ldaseqmodel.sslm.compute_expected_log_prob`.

        Parameters
        ----------
        sstats : numpy.ndarray
            Sufficient statistics for a particular topic. Corresponds to matrix beta in the linked paper for the
            current time slice, expected shape (`self.vocab_len`, `num_topics`).

        Returns
        -------
        float
            The lower bound for the true posterior achieved using the fitted approximate distribution.

        """
        W = self.vocab_len
        bound = 0
        old_bound = 0
        sslm_fit_threshold = 1e-6
        sslm_max_iter = 2
        converged = sslm_fit_threshold + 1

        # computing variance, fwd_variance
        self.variance, self.fwd_variance =             (np.array(x) for x in zip(*(self.compute_post_variance(w, self.chain_variance) for w in range(W))))

        # column sum of sstats
        totals = sstats.sum(axis=0)
        iter_ = 0

        model = "DTM"
        if model == "DTM":
            bound = self.compute_bound(sstats, totals)
        if model == "DIM":
            bound = self.compute_bound_fixed(sstats, totals)

        logger.info("initial sslm bound is %f", bound)

        while converged > sslm_fit_threshold and iter_ < sslm_max_iter:
            iter_ += 1
            old_bound = bound
            self.obs, self.zeta = self.update_obs(sstats, totals)
            if model == "DTM":
                bound = self.compute_bound(sstats, totals)
            if model == "DIM":
                bound = self.compute_bound_fixed(sstats, totals)
            converged = np.fabs((bound - old_bound) / old_bound)
            logger.info("iteration %i iteration lda seq bound is %f convergence is %f", iter_, bound, converged)

        self.e_log_prob = self.compute_expected_log_prob()
        return bound

    def compute_bound(self, sstats, totals):
        """Compute the maximized lower bound achieved for the log probability of the true posterior.

        Uses the formula presented in the appendix of the DTM paper (formula no. 5).

        Parameters
        ----------
        sstats : numpy.ndarray
            Sufficient statistics for a particular topic. Corresponds to matrix beta in the linked paper for the first
            time slice, expected shape (`self.vocab_len`, `num_topics`).
        totals : list of int of length `len(self.time_slice)`
            The totals for each time slice.

        Returns
        -------
        float
            The maximized lower bound.

        """
        w = self.vocab_len
        t = self.num_time_slices

        term_1 = 0
        term_2 = 0
        term_3 = 0

        val = 0
        ent = 0

        chain_variance = self.chain_variance
        # computing mean, fwd_mean
        self.mean, self.fwd_mean =             (np.array(x) for x in zip(*(self.compute_post_mean(w, self.chain_variance) for w in range(w))))
        self.zeta = self.update_zeta()

        val = sum(self.variance[w][0] - self.variance[w][t] for w in range(w)) / 2 * chain_variance

        logger.info("Computing bound, all times")

        for t in range(1, t + 1):
            term_1 = 0.0
            term_2 = 0.0
            ent = 0.0
            for w in range(w):

                m = self.mean[w][t]
                prev_m = self.mean[w][t - 1]

                v = self.variance[w][t]

                # w_phi_l is only used in Document Influence Model; the values are always zero in this case
                # w_phi_l = sslm.w_phi_l[w][t - 1]
                # exp_i = np.exp(-prev_m)
                # term_1 += (np.power(m - prev_m - (w_phi_l * exp_i), 2) / (2 * chain_variance)) -
                # (v / chain_variance) - np.log(chain_variance)

                term_1 +=                     (np.power(m - prev_m, 2) / (2 * chain_variance)) - (v / chain_variance) - np.log(chain_variance)
                term_2 += sstats[w][t - 1] * m
                # term_2 += sstats[w][t - 1] * self.bvt[w][t - 1]* m
                ent += np.log(v) / 2  # note the 2pi's cancel with term1 (see doc)

            term_3 = -totals[t - 1] * np.log(self.zeta[t - 1])
            val += term_2 + term_3 + ent - term_1

        return val

    def update_obs(self, sstats, totals):
        """Optimize the bound with respect to the observed variables.

        TODO:
        This is by far the slowest function in the whole algorithm.
        Replacing or improving the performance of this would greatly speed things up.

        Parameters
        ----------
        sstats : numpy.ndarray
            Sufficient statistics for a particular topic. Corresponds to matrix beta in the linked paper for the first
            time slice, expected shape (`self.vocab_len`, `num_topics`).
        totals : list of int of length `len(self.time_slice)`
            The totals for each time slice.

        Returns
        -------
        (numpy.ndarray of float, numpy.ndarray of float)
            The updated optimized values for obs and the zeta variational parameter.

        """
        OBS_NORM_CUTOFF = 2
        STEP_SIZE = 0.01
        TOL = 1e-3

        W = self.vocab_len
        T = self.num_time_slices

        runs = 0
        mean_deriv_mtx = np.zeros((T, T + 1))
        norm_cutoff_obs = None
        for w in range(W):
            w_counts = sstats[w]
            counts_norm = 0
            # now we find L2 norm of w_counts
            for i in range(len(w_counts)):
                counts_norm += w_counts[i] * w_counts[i]

            counts_norm = np.sqrt(counts_norm)
            if counts_norm < OBS_NORM_CUTOFF and norm_cutoff_obs is not None:
                obs = self.obs[w]
                norm_cutoff_obs = np.copy(obs)
            else:
                if counts_norm < OBS_NORM_CUTOFF:
                    w_counts = np.zeros(len(w_counts))

                # TODO: apply lambda function
                for t in range(T):
                    mean_deriv_mtx[t] = self.compute_mean_deriv(w, t, mean_deriv_mtx[t])

                deriv = np.zeros(T)
                args = self, w_counts, totals, mean_deriv_mtx, w, deriv
                obs = self.obs[w]
                model = "DTM"

                if model == "DTM":
                    # slowest part of method
                    obs = optimize.fmin_cg(
                        f=f_obs, fprime=df_obs, x0=obs, gtol=TOL, args=args, epsilon=STEP_SIZE, disp=0
                    )
                if model == "DIM":
                    pass
                runs += 1

                if counts_norm < OBS_NORM_CUTOFF:
                    norm_cutoff_obs = obs

                self.obs[w] = obs
        self.zeta = self.update_zeta()
        return self.obs, self.zeta

    def compute_mean_deriv(self, word, time, deriv):
        """Helper functions for optimizing a function.

        Compute the derivative of:

        .. :math::

            E[\beta_{t,w}]/d obs_{s,w} for t = 1:T.

        Parameters
        ----------
        word : int
            The word's ID.
        time : int
            The time slice.
        deriv : list of float
            Derivative for each time slice.

        Returns
        -------
        list of float
            Mean derivative for each time slice.

        """

        T = self.num_time_slices
        fwd_variance = self.variance[word]

        deriv[0] = 0

        # forward pass
        for t in range(1, T + 1):
            if self.obs_variance > 0.0:
                w = self.obs_variance / (fwd_variance[t - 1] + self.chain_variance + self.obs_variance)
            else:
                w = 0.0
            val = w * deriv[t - 1]
            if time == t - 1:
                val += (1 - w)
            deriv[t] = val

        for t in range(T - 1, -1, -1):
            if self.chain_variance == 0.0:
                w = 0.0
            else:
                w = self.chain_variance / (fwd_variance[t] + self.chain_variance)
            deriv[t] = w * deriv[t] + (1 - w) * deriv[t + 1]

        return deriv

    def compute_obs_deriv(self, word, word_counts, totals, mean_deriv_mtx, deriv):
        """Derivation of obs which is used in derivative function `df_obs` while optimizing.

        Parameters
        ----------
        word : int
            The word's ID.
        word_counts : list of int
            Total word counts for each time slice.
        totals : list of int of length `len(self.time_slice)`
            The totals for each time slice.
        mean_deriv_mtx : list of float
            Mean derivative for each time slice.
        deriv : list of float
            Mean derivative for each time slice.

        Returns
        -------
        list of float
            Mean derivative for each time slice.

        """

        # flag
        init_mult = 1000

        T = self.num_time_slices

        mean = self.mean[word]
        variance = self.variance[word]
        bt = self.bvt[word]
        # only used for DIM mode
        # w_phi_l = self.w_phi_l[word]
        # m_update_coeff = self.m_update_coeff[word]

        # temp_vector holds temporary zeta values
        self.temp_vect = np.zeros(T)

        for u in range(T):
            self.temp_vect[u] = np.exp(mean[u + 1] + variance[u + 1] / 2)

        for t in range(T):
            mean_deriv = mean_deriv_mtx[t]
            term1 = 0
            term2 = 0
            term3 = 0
            term4 = 0

            for u in range(1, T + 1):
                mean_u = mean[u]
                mean_u_prev = mean[u - 1]
                dmean_u = mean_deriv[u]
                dmean_u_prev = mean_deriv[u - 1]

                term1 += (mean_u - mean_u_prev) * (dmean_u - dmean_u_prev)
#                 term2 += (word_counts[u - 1] - (totals[u - 1] * self.temp_vect[u - 1] / self.zeta[u - 1])) * dmean_u
                term2 += (word_counts[u - 1] - (totals[u - 1] * bt[u - 1] * self.temp_vect[u - 1] / self.zeta[u - 1])) * dmean_u

                model = "DTM"
                if model == "DIM":
                    # do some stuff
                    pass

            if self.chain_variance:
                term1 = - (term1 / self.chain_variance)
                term1 = term1 - (mean[0] * mean_deriv[0]) / (init_mult * self.chain_variance)
            else:
                term1 = 0.0

            deriv[t] = term1 + term2 + term3 + term4

        return deriv

# the following functions are used in update_obs as the objective function.
def f_obs(x, *args):
    """Function which we are optimising for minimizing obs.

    Parameters
    ----------
    x : list of float
        The obs values for this word.
    sslm : :class:`~gensim.models.ldaseqmodel.sslm`
        The State Space Language Model for DTM.
    word_counts : list of int
        Total word counts for each time slice.
    totals : list of int of length `len(self.time_slice)`
        The totals for each time slice.
    mean_deriv_mtx : list of float
        Mean derivative for each time slice.
    word : int
        The word's ID.
    deriv : list of float
        Mean derivative for each time slice.

    Returns
    -------
    list of float
        The value of the objective function evaluated at point `x`.

    """
    sslm, word_counts, totals, mean_deriv_mtx, word, deriv = args
    # flag
    init_mult = 1000

    T = len(x)
    val = 0
    term1 = 0
    term2 = 0

    # term 3 and 4 for DIM
    term3 = 0
    term4 = 0

    sslm.obs[word] = x
    sslm.mean[word], sslm.fwd_mean[word] = sslm.compute_post_mean(word, sslm.chain_variance)

    mean = sslm.mean[word]
    variance = sslm.variance[word]
    bt = sslm.bvt[word]
    # only used for DIM mode
    # w_phi_l = sslm.w_phi_l[word]
    # m_update_coeff = sslm.m_update_coeff[word]

    for t in range(1, T + 1):
        mean_t = mean[t]
        mean_t_prev = mean[t - 1]

        val = mean_t - mean_t_prev
        term1 += val * val
        term2 += word_counts[t - 1] * mean_t - totals[t - 1] * bt[t - 1] * np.exp(mean_t + variance[t] / 2) / sslm.zeta[t - 1]

        model = "DTM"
        if model == "DIM":
            # stuff happens
            pass

    if sslm.chain_variance > 0.0:

        term1 = - (term1 / (2 * sslm.chain_variance))
        term1 = term1 - mean[0] * mean[0] / (2 * init_mult * sslm.chain_variance)
    else:
        term1 = 0.0

    final = -(term1 + term2 + term3 + term4)

    return final


def df_obs(x, *args):
    """Derivative of the objective function which optimises obs.

    Parameters
    ----------
    x : list of float
        The obs values for this word.
    sslm : :class:`~gensim.models.ldaseqmodel.sslm`
        The State Space Language Model for DTM.
    word_counts : list of int
        Total word counts for each time slice.
    totals : list of int of length `len(self.time_slice)`
        The totals for each time slice.
    mean_deriv_mtx : list of float
        Mean derivative for each time slice.
    word : int
        The word's ID.
    deriv : list of float
        Mean derivative for each time slice.

    Returns
    -------
    list of float
        The derivative of the objective function evaluated at point `x`.

    """
    sslm, word_counts, totals, mean_deriv_mtx, word, deriv = args

    sslm.obs[word] = x
    sslm.mean[word], sslm.fwd_mean[word] = sslm.compute_post_mean(word, sslm.chain_variance)

    model = "DTM"
    if model == "DTM":
        deriv = sslm.compute_obs_deriv(word, word_counts, totals, mean_deriv_mtx, deriv)
    elif model == "DIM":
        deriv = sslm.compute_obs_deriv_fixed(
            p.word, p.word_counts, p.totals, p.sslm, p.mean_deriv_mtx, deriv)  # noqa:F821

    return np.negative(deriv)

class WSTMdynamic(utils.SaveLoad):

    def __init__(self, corpus=None, time_slice=None, id2word=None, x = 1, y = 1, pi = 1, beta_bar = 1e-5,
                 obs_variance=0.5, chain_variance=0.005, iters_E_max=20, iters_E_min = 5, iters_EM_max = 20,
                 init_iters = 10, iters_EM_min = 5, threshold_E = 1, threshold_EM = 1, num_topics=5, random_state=123):

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
        self.time_slice = time_slice
        if self.time_slice is not None:
            self.num_time_slices = len(time_slice)
        self.num_time_slices = len(time_slice)
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
        self.obs_variance = obs_variance
        self.chain_variance = chain_variance
        self.random_state = random_state
        self.init_iters = init_iters

        self.init_model()
        self.fit_model()

    def beta_init(self, staticmodel):
        num_time_slices = self.num_time_slices
        num_topics = self.num_topics
        vocab_len = self.vocab_len
        beta = np.zeros((num_time_slices, num_topics, vocab_len))
        for t in range(num_time_slices):
            beta[t] = staticmodel.beta
        return beta

    def b_init(self, staticmodel):
        num_time_slices = self.num_time_slices
        num_topics = self.num_topics
        vocab_len = self.vocab_len
        btkv = np.zeros((num_time_slices, num_topics, vocab_len))
        for t in range(num_time_slices):
            btkv[t] = staticmodel.bkv
        return btkv

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
            #         temp = np.random.normal(0.5,0.5,(n_per, num_topics))
            for n in range(n_per):
                #             phi[m][n]= np.exp(temp[n])/np.sum(np.exp(temp[n]))
                phi[m][n] = 1.0 / num_topics
        return phi

    def gamma_init(self, phi):
        pi = self.pi
        gamma = pi + np.sum(phi, axis=1)
        return gamma

    def topic_chains_init(self, btkv):
        num_topics = self.num_topics
        num_time_slices = self.num_time_slices
        vocab_len = self.vocab_len
        chain_variance = self.chain_variance
        obs_variance = self.obs_variance
        topic_chains = []
        for k in range(num_topics):
            sslm_ = sslm(
                num_time_slices=num_time_slices, vocab_len=vocab_len, num_topics=num_topics,
                chain_variance=chain_variance, obs_variance=obs_variance, bvt=btkv[:, k, :].T
            )
            topic_chains.append(sslm_)
        return topic_chains

    def sslm_init(self, init_suffstats):
        topic_chain_variance = self.chain_variance
        topic_obs_variance = self.obs_variance
        for k, chain in enumerate(self.topic_chains):
            sstats = init_suffstats[:, k]
            sslm.sslm_counts_init(chain, topic_obs_variance, topic_chain_variance, sstats)

    def calculate_Bk_t(self, bkv, beta, k):
        beta_bar = self.beta_bar
        vocab_len = self.vocab_len
        # 计算E_q[log(\sum_r{bkr*betakr}+V*beta_bar)]
        b = bkv[k]
        be = beta[k]
        # 先用近似的做法，有待改进
        Bk = np.log(np.sum(b * be) + vocab_len * beta_bar)
        return Bk

    def calculate_deriv_Bk_t(self, bkv, beta, k, v):
        beta_bar = self.beta_bar
        vocab_len = self.vocab_len
        # 计算\partial{E_q[log(\sum_r{bkr*betakr}+V*beta_bar)]} {bkv}
        vid = list(set(list(range(vocab_len))) - set([v]))
        b = bkv[k, vid]
        be = beta[k, vid]
        # 先用近似的做法，有待改进
        dBkv = np.log(beta[k, v] + np.sum(b * be) + vocab_len * beta_bar) - np.log(
            np.sum(b * be) + vocab_len * beta_bar)
        return dBkv

    def update_phi_t(self, phi_t, bkv_t, beta_t, gamma_t, corpus_t, corpus_t_len):
        beta_bar = self.beta_bar
        num_topics = self.num_topics
        temp0 = beta_t * bkv_t + beta_bar
        for k in range(num_topics):
            temp0[k] = temp0[k] / np.sum(temp0[k])
        n_list = [len(x) for x in corpus_t]
        for m in range(corpus_t_len):
            n_per = n_list[m]
            wordidlist = [wordid for wordid, count in corpus_t[m]]
            temp1 = np.log(temp0[:, wordidlist])
            temp2 = psi(gamma_t[m])
            temp_phi_t = np.exp(temp1 + np.tile(temp2, (n_per, 1)).T).T  # n_per*K维
            phi_t[m, range(n_per)] = (temp_phi_t.T / temp_phi_t.sum(axis=1)).T
        return phi_t

    def update_gamma_t(self, phi_t):
        pi = self.pi
        gamma_t = pi + np.sum(phi_t, axis=1)
        return gamma_t

    def calculate_nkw_t(self, phi_t, corpus_t, corpus_t_len):
        num_topics = self.num_topics
        vocab_len = self.vocab_len
        nkw_t = np.zeros((num_topics, vocab_len))
        n_list = [len(x) for x in corpus_t]
        for m in range(corpus_t_len):
            n_per = n_list[m]
            wordidlist = [corpus_t[m][n][0] for n in range(n_per)]
            wordfreqlist = [corpus_t[m][n][1] for n in range(n_per)]
            nkw_t[:, wordidlist] += (phi_t[m, range(n_per), :] * np.tile(wordfreqlist, (num_topics, 1)).T).T
        return nkw_t

    def calculate_ntkw(self, phi):
        corpus = self.corpus
        num_topics = self.num_topics
        vocab_len = self.vocab_len
        num_time_slices = self.num_time_slices
        time_slice = self.time_slice
        ntkw = np.zeros((num_time_slices, num_topics, vocab_len))
        for t in range(num_time_slices):
            if t == 0:
                d_slice = range(np.cumsum(time_slice)[t])
            else:
                d_slice = range(np.cumsum(time_slice)[t - 1], np.cumsum(time_slice)[t])
            phi_t = phi[d_slice].copy()
            corpus_t = [corpus[d] for d in d_slice]
            corpus_t_len = len(corpus_t)
            ntkw[t] = self.calculate_nkw_t(phi_t, corpus_t, corpus_t_len)
        return ntkw

    def update_bkv_t(self, bkv_t, beta_t, x_hat_t, y_hat_t, nkw_t):
        num_topics = self.num_topics
        vocab_len = self.vocab_len
        beta_bar = self.beta_bar
        for k in range(num_topics):
            dBk = beta_t[k] / np.sum(bkv_t[k] * beta_t[k] + vocab_len * beta_bar)  # V维
            temp = nkw_t[k] * (np.log(beta_t[k] + beta_bar) - np.log(beta_bar) - dBk) + psi(x_hat_t[k]) - psi(
                y_hat_t[k])
            temp[temp > 700] = 700
            bkv_t[k] = np.exp(temp) / (np.exp(temp) + 1)
        return bkv_t

    def update_xy_t(self, bkv_t):
        x = self.x
        y = self.y
        vocab_len = self.vocab_len
        x_hat_t = x + np.sum(bkv_t, axis=1)
        y_hat_t = y + vocab_len - np.sum(bkv_t, axis=1)
        return (x_hat_t, y_hat_t)

    def update_topic_chains(self, ntkw):
        topic_chains = self.topic_chains
        num_topics = self.num_topics
        num_time_slices = self.num_time_slices
        vocab_len = self.vocab_len
        topic_suffstats = []
        topic_word = np.zeros((num_time_slices, num_topics, vocab_len))
        for k in range(num_topics):
            topic_suffstats.append(ntkw[:, k, :].T)
        lhood = 0
        for k, chain in enumerate(topic_chains):
            print("Fitting topic number %i" % k)
            lhood_term = sslm.fit_sslm(chain, topic_suffstats[k])
            lhood += lhood_term
            temp = chain.e_log_prob.T  # T*V
            for t in range(num_time_slices):
                topic_word[t, k] = np.exp(temp[t]) / np.sum(np.exp(temp[t]))
        return lhood, topic_word

    def update_beta(self, btkv, topic_word):
        beta = topic_word / btkv
        return beta

    def cal_ELBO_t(self, phi, nkw, gamma, bkv, x_hat, y_hat, beta):
        beta_bar = self.beta_bar
        x = self.x
        y = self.y
        pi = self.pi
        vocab_len = self.vocab_len
        num_topics = self.num_topics
        # 单期的ELBO
        B = np.asarray([self.calculate_Bk_t(bkv, beta, k) for k in range(num_topics)])
        term1 = np.sum(
            nkw * (bkv * np.log(beta + beta_bar) + (1 - bkv) * np.log(beta_bar) - np.tile(B, (vocab_len, 1)).T))
        term2 = np.sum((np.sum(bkv, axis=1) + x - 1) * (psi(x_hat) - psi(x_hat + y_hat)) + (
                    vocab_len - np.sum(bkv, axis=1) + y - 1) * (psi(y_hat) - psi(x_hat + y_hat)))
        term3 = np.sum(
            (pi + np.sum(phi, axis=1) - 1) * (psi(gamma) - psi(np.tile(np.sum(gamma, axis=1), (num_topics, 1)).T)))
        term5 = - np.sum(phi[phi > 0] * np.log(phi[phi > 0]))
        term6 = - np.sum((gamma - 1) * psi(gamma)) + np.sum(
            (np.sum(gamma, axis=1) - 1) * psi(np.sum(gamma, axis=1)))
        term7 = - np.sum(gammaln(np.sum(gamma, axis=1))) + np.sum(gammaln(gamma))
        term9 = - np.sum(bkv[bkv > 0] * np.log(bkv[bkv > 0])) - np.sum(
            (1 - bkv[bkv < 1]) * np.log(1 - bkv[bkv < 1]))
        term11 = np.sum(betaln(x_hat, y_hat) - (x_hat - 1) * (psi(x_hat) - psi(x_hat + y_hat)) - (y_hat - 1) * (
                    psi(y_hat) - psi(x_hat + y_hat)))
        ELBO = term1 + term2 + term3 + term5 + term6 + term7 + term9 + term11
        return ELBO

    def cal_ELBO_beta_t(self, nkw_t, bkv_t, beta_t):
        beta_bar = self.beta_bar
        vocab_len = self.vocab_len
        num_topics = self.num_topics
        B_t = np.asarray([self.calculate_Bk_t(bkv_t, beta_t, k) for k in range(num_topics)])
        term1 = np.sum(nkw_t * (bkv_t * np.log(beta_t + beta_bar) + (1 - bkv_t) * np.log(beta_bar) - np.tile(B_t, (
        vocab_len, 1)).T))
        return term1

    def cal_ELBO_beta_whole(self, ntkw, btkv, beta):
        beta_bar = self.beta_bar
        vocab_len = self.vocab_len
        num_topics = self.num_topics
        num_time_slices = self.num_time_slices
        term = 0
        for t in range(num_time_slices):
            nkw_t = ntkw[t].copy()
            bkv_t = btkv[t].copy()
            beta_t = beta[t].copy()
            term += self.cal_ELBO_beta_t(nkw_t, bkv_t, beta_t)
        return term

    def init_model(self):
        # E步初始化
        staticmodel = wordsparseTM(corpus = self.corpus, id2word = self.id2word, num_topics = self.num_topics, x = self.x, y = self.y, iters_EM_max= self.init_iters, random_state= self.random_state)
        beta = self.beta_init(staticmodel)
        btkv = self.b_init(staticmodel)
        topic_chains = self.topic_chains_init(btkv)
        self.topic_chains = topic_chains
        sstats = staticmodel.nkw.T
        self.sslm_init(sstats)
        self.beta = beta

    def fit_model(self):
        iters_EM_max = self.iters_EM_max
        iters_EM_min = self.iters_EM_min
        iters_E_max = self.iters_E_max
        iters_E_min = self.iters_E_min
        threshold_E = self.threshold_E
        threshold_EM = self.threshold_EM
        num_time_slices = self.num_time_slices
        num_topics = self.num_topics
        vocab_len = self.vocab_len
        time_slice = self.time_slice
        corpus = self.corpus
        beta = self.beta
        # EM
        ELBO_EM = 0
        for it in range(iters_EM_max):
            ELBO_E = 0
            ELBO_EM_old = ELBO_EM
            # initialize
        #     btkv = b_init(num_time_slices, num_topics, vocab_len, x, y)
            btkv = np.full([num_time_slices, num_topics, vocab_len], 0.5)
            phi = self.phi_init()
            gamma = self.gamma_init(phi)
            ntkw = self.calculate_ntkw(phi)
            x_hat = np.full([num_time_slices, num_topics], 1)
            y_hat = np.full([num_time_slices, num_topics], 1)
            ## E-step
            for t in range(num_time_slices):
                if t == 0:
                    d_slice = range(np.cumsum(time_slice)[t])
                else:
                    d_slice = range(np.cumsum(time_slice)[t-1],np.cumsum(time_slice)[t])
                phi_t = phi[d_slice].copy()
                gamma_t = gamma[d_slice].copy()
                nkw_t = ntkw[t].copy()
                bkv_t = btkv[t].copy()
                x_hat_t = x_hat[t].copy()
                y_hat_t = y_hat[t].copy()
                beta_t = beta[t].copy()
                corpus_t = [corpus[d] for d in d_slice]
                corpus_t_len = len(corpus_t)
                ELBO_E_t = 0
                for it_E in range(iters_E_max):
                    ELBO_E_t_old = ELBO_E_t
                    phi_t = self.update_phi_t(phi_t, bkv_t, beta_t, gamma_t, corpus_t, corpus_t_len)
                    gamma_t = self.update_gamma_t(phi_t)
                    nkw_t = self.calculate_nkw_t(phi_t, corpus_t, corpus_t_len)
                    bkv_t = self.update_bkv_t(bkv_t, beta_t, x_hat_t, y_hat_t, nkw_t)
            #         x_hat_t, y_hat_t = update_xy_t(bkv_t, x, y, vocab_len)
                    ELBO_E_t = self.cal_ELBO_t(phi_t, nkw_t, gamma_t, bkv_t, x_hat_t, y_hat_t, beta_t)
                    if (ELBO_E_t - ELBO_E_t_old <= threshold_E and it_E >= iters_E_min - 1) or (it_E == iters_E_max - 1):
                        break
                phi[d_slice] = phi_t
                gamma[d_slice] = gamma_t
                ntkw[t] = nkw_t
                btkv[t] = bkv_t
                x_hat[t] = x_hat_t
                y_hat[t] = y_hat_t
                beta[t] = beta_t
                ELBO_E += ELBO_E_t

            ELBO_EM = ELBO_E
            term1_old = self.cal_ELBO_beta_whole(ntkw, btkv, beta)
            ## M-step
            lhood, topic_word = self.update_topic_chains(ntkw)
            beta = self.update_beta(btkv, topic_word)
            term1 = self.cal_ELBO_beta_whole(ntkw, btkv, beta)
            ELBO_EM += term1 - term1_old
            print('iter', it, 'completed!', 'ELBO:', round(ELBO_EM, 4), 'lhood:', round(lhood, 4))

            self.phi = phi
            self.gamma = gamma
            self.ntkw = ntkw
            self.btkv = btkv
            self.beta = beta
            self.topic_word = topic_word

            # 判断收敛
            if (ELBO_EM - ELBO_EM_old <= threshold_EM and it >= iters_EM_min - 1) or (it == iters_EM_max - 1):
                break

    def get_theta(self):
        gamma = self.gamma
        corpus_len = self.corpus_len
        num_topics = self.num_topics
        theta = np.zeros((corpus_len, num_topics))
        for m in range(corpus_len):
            theta[m] = gamma[m] / np.sum(gamma[m])
        return theta

    def print_topic_times(self, topic, top_terms=20):
        """Get the most relevant words for a topic, for each timeslice. This can be used to inspect the evolution of a
        topic through time.

        Parameters
        ----------
        topic : int
            The index of the topic.
        top_terms : int, optional
            Number of most relevant words associated with the topic to be returned.

        Returns
        -------
        list of list of str
            Top `top_terms` relevant terms for the topic for each time slice.

        """
        topics = []
        for time in range(self.num_time_slices):
            topics.append(self.print_topic(topic, time, top_terms))

        return topics

    def print_topics(self, time=0, top_terms=20):
        """Get the most relevant words for every topic.

        Parameters
        ----------
        time : int, optional
            The time slice in which we are interested in (since topics evolve over time, it is expected that the most
            relevant words will also gradually change).
        top_terms : int, optional
            Number of most relevant words to be returned for each topic.

        Returns
        -------
        list of list of (str, float)
            Representation of all topics. Each of them is represented by a list of pairs of words and their assigned
            probability.

        """
        return [self.print_topic(topic, time, top_terms) for topic in range(self.num_topics)]

    def print_topic(self, topic, time=0, top_terms=20):
        """Get the list of words most relevant to the given topic.

        Parameters
        ----------
        topic : int
            The index of the topic to be inspected.
        time : int, optional
            The time slice in which we are interested in (since topics evolve over time, it is expected that the most
            relevant words will also gradually change).
        top_terms : int, optional
            Number of words associated with the topic to be returned.

        Returns
        -------
        list of (str, float)
            The representation of this topic. Each element in the list includes the word itself, along with the
            probability assigned to it by the topic.

        """
        topic = self.topic_chains[topic].e_log_prob
        topic = np.transpose(topic)
        topic = np.exp(topic[time])
        topic = topic / topic.sum()
        bestn = matutils.argsort(topic, top_terms, reverse=True)
        beststr = [(self.id2word[id_], topic[id_]) for id_ in bestn]
        return beststr