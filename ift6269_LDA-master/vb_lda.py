import numpy as np
import scipy
from scipy import special
import time
from scipy import misc
import pickle as pkl
import os
import utils

def dirichlet_log_expectation(param):
    if len(param.shape) == 1:
        return scipy.special.psi(param) - scipy.special.psi(np.sum(param))
    else:
        return scipy.special.psi(param) - scipy.special.psi(np.sum(param, 1))[:, np.newaxis]

# Get the variational parameters estimation
def getVariationalVar(alpha, lb, wordsOfDoc):

    nTopics = len(alpha)
    nWords = len(wordsOfDoc)

    #phis_t = np.ones((nTopics, nWords)) * 1./ nTopics
    log_phis_t = np.log(np.ones((nTopics, nWords)) * 1./ nTopics)
    otherPhis_t = alpha + nWords / float(nTopics)

    #phis_tp1 = phis_t
    log_phis_tp1 = log_phis_t
    otherPhis_tp1 = otherPhis_t

    nbIter = 0
    B_log_expectation = dirichlet_log_expectation(lb)
    #print "B_log:"
    #print B_log_expectation

    while True:

        # sum_otherPhis = special.digamma(otherPhis_t.sum())
        # phis_tp1 = B[:, wordsOfDoc.astype(int)]
        # phis_tp1 *= np.expand_dims(np.exp(special.digamma(otherPhis_t) - sum_otherPhis), -1)
        # phis_tp1 /= phis_tp1.sum(axis=0)
        log_phis_tp1 = B_log_expectation[:, wordsOfDoc.astype(int)]
        #print "plop", B_log_expectation[:, wordsOfDoc.astype(int)]
        #log_phis_tp1 += np.expand_dims(special.digamma(otherPhis_t) - special.digamma(otherPhis_t.sum()), -1)
        log_phis_tp1 += np.expand_dims(special.digamma(otherPhis_t), -1)
        #print "plop2", log_phis_tp1
        # Normalization
        #print "plop3", misc.logsumexp(log_phis_tp1, axis=0)
        log_phis_tp1 -= np.expand_dims(misc.logsumexp(log_phis_tp1, axis=0), 0)
        #print "hop hop:"
        #print np.exp(log_phis_tp1).sum(axis=1) #+ alpha

        otherPhis_tp1 = alpha + np.exp(log_phis_tp1).sum(axis=1)

        if np.linalg.norm(log_phis_t - log_phis_tp1) + np.linalg.norm(otherPhis_tp1 - otherPhis_t) < 0.0001:
            # print "Done estimating the variational parameters in {} iterations".format(nbIter)
            break

        if nbIter > 100:
            #print "Took to much time!"
            # print otherPhis_tp1, phis_tp1
            break

        nbIter += 1

        log_phis_t = log_phis_tp1
        otherPhis_t = otherPhis_tp1

    return log_phis_tp1, otherPhis_tp1

# estimate the B
def getParamB(phis, wordsOfDocs, vocabSize):
    nTopics = phis[0].shape[0]
    nDocs = len(phis)

    B = np.zeros((nTopics, vocabSize))

    for i in range(nTopics):
        for d in range(nDocs):
            for n, word in enumerate(wordsOfDocs[d]):
                B[i, int(word)] += phis[d][i, n]

        B[i] += 1.
        B[i] = B[i] / B[i].sum()

    return B

# estimate the lambda
def getVarLambda(log_phis, smoothing, wordsOfDocs, vocabSize):
    nTopics = log_phis[0].shape[0]
    nDocs = len(log_phis)

    lb = np.zeros((nTopics, vocabSize))

    for i in range(nTopics):
        for d in range(nDocs):
            for n, word in enumerate(wordsOfDocs[d]):
                lb[i, int(word)] += np.exp(log_phis[d][i, n])

        #B[i] += 1.
        #B[i] = B[i] / B[i].sum()
    lb += smoothing

    return lb

# alpha stuff
def alphaFirstDeriv(alpha, otherPhis):
    g = np.zeros((otherPhis[0].shape[0],))
    M = len(otherPhis)
    k = len(alpha)
    suf = np.zeros((otherPhis[0].shape[0],))

    for i in range(k):
        psi_sum = special.psi(alpha.sum())
        g[i] = M * (psi_sum - special.psi(alpha[i]))
        tmp = np.array([special.psi(otherPhis[d][i]) for d in range(M)]).sum()
        tmp -= np.array([special.psi(otherPhis[d].sum()) for d in range(M)]).sum()
        suf[i] = tmp
        g[i] += tmp

    return g  # , suf


def alphaSecondDeriv(alpha, M):
    k = len(alpha)

    h = -M * special.polygamma(1, alpha)
    z = M * special.polygamma(1, alpha.sum())

    return h, z


def getParamAlpha(alpha, otherPhis):


    k = otherPhis[0].shape[0]
    M = len(otherPhis)

    alpha_old = alpha  # np.ones((k,))
    alpha_new = alpha_old
    nbIter = 0
    decay = 1.
    decay_factor = 0.9

    while True:

        g = alphaFirstDeriv(alpha_old, otherPhis)
        # print "The gradient:", g
        h, z = alphaSecondDeriv(alpha_old, M)
        c = np.sum(g / h) / (1. / z + np.sum(1. / h))
        step = (g - c) / h

        while np.any(alpha_old <= np.power(decay_factor, decay) * step):
            # print "hoho!"
            decay += 1.
            if decay > 15.:
                print "singular hessian, I quit!"
                break

        if decay > 10.:
            # print "singular hessian, I quit!"
            break

        alpha_new = alpha_old - np.power(decay_factor, decay) * step

        # print alpha_old, alpha_new

        if np.linalg.norm(alpha_new - alpha_old) < 0.000001:
            #print "Done estimating alpha in {} steps".format(nbIter)
            break

        if nbIter > 100:
            print "Took to long to converge :S ({})".format(nbIter)
            break

        alpha_old = alpha_new

        nbIter += 1

    return alpha_new



def log_likelyhood(alpha, lb, vocabSize, corpus, smoothing = None, log_phis = None, otherPhis = None):

    M = len(corpus)
    k = len(alpha)

    if log_phis == None or otherPhis == None:
        log_phis = []
        otherPhis = []
        for d in range(M):
            tmp1, tmp2 = getVariationalVar(alpha, lb, corpus[d])
            log_phis.append(tmp1)
            otherPhis.append(tmp2)

    if smoothing is None:
        smoothing = np.zeros((k, vocabSize)) + 1
        #lb = getVarLambda(log_phis, smoothing, corpus, vocabSize)

    log_like = 0.
    B_log_expectation = dirichlet_log_expectation(lb)
    #  Log likely hood for the documents
    for d in range(M):
        nWords = len(corpus[d])
        doc_gamma_exp = dirichlet_log_expectation(otherPhis[d])
        # alpha term
        log_like += special.gammaln(alpha.sum()) - special.gammaln(alpha).sum()
        log_like += ((alpha - 1)* doc_gamma_exp).sum()

        # phis
        log_like += np.array([np.exp(log_phis[d][:,n]) * doc_gamma_exp for n in range(nWords)]).sum()

        # words
        log_like += np.array([np.exp(log_phis[d][:,n]) * B_log_expectation[:, int(corpus[d][n])] for n in range(nWords)]).sum()

        # otherPhis
        log_like += -special.gammaln(otherPhis[d].sum()) + special.gammaln(otherPhis[d]).sum()
        log_like += -((otherPhis[d] - 1) * doc_gamma_exp).sum()

        # Phis
        log_like += - np.array([np.exp(log_phis[d][:,n]) * log_phis[d][:,n] for n in range(nWords)]).sum()

    topic_likelyhood = 0.
    # Log-likelyhood of the topics
    for k in range(k):
        log_B_expectation = dirichlet_log_expectation(lb[k])

        # with eta
        topic_likelyhood += (special.gammaln(smoothing[k].sum()) - special.gammaln(smoothing[k]).sum())
        topic_likelyhood += (((smoothing[k] - 1)* log_B_expectation).sum())

        topic_likelyhood -= (special.gammaln(lb[k].sum()) - special.gammaln(lb[k]).sum())
        topic_likelyhood -= (((lb[k] - 1) * log_B_expectation).sum())

    #print "The topic likelyhood:", topic_likelyhood
    #print "The total", topic_likelyhood + log_like

    return log_like + topic_likelyhood #/sum([len(d) for d in corpus])

def Var_EM(k, vocabSize, corpus, maxIter=50, batchSize = -1, smoothing=None, seed=1995, file_name="testing123.pkl"):


    np.random.seed(seed)
    M = len(corpus)
    alpha_old = np.ones((k,)) / k
    #B_old = np.random.dirichlet(np.ones((vocabSize,)) / float(vocabSize), k)

    if smoothing == None:
        smoothing = np.zeros((k, vocabSize)) + 1./vocabSize

    #lb_old = np.random.gamma(100., 1/100., (k, vocabSize))
    lb_old = np.random.gamma(vocabSize, 1./vocabSize, (k, vocabSize))
    #print "initial lb:", lb_old

    alpha_new = alpha_old
    #B_new = B_old
    lb_new = lb_old
    log_like = -np.inf
    likes = []
    log_phis = []
    otherPhis = []

    print "start training"
    for nbIter in range(maxIter):

        start_time = time.time()
        #For the batch
        start_doc = 0
        end_doc = M
        doc_ids = range(start_doc, end_doc)

        if batchSize != -1:
            start_doc = (nbIter * batchSize) % M
            end_doc = (start_doc + batchSize) % M
            if end_doc < start_doc:
                doc_ids = range(start_doc, M) + range(0, end_doc)
            else:
                doc_ids = range(start_doc, end_doc)

        sub_corpus = [corpus[i] for i in doc_ids]


        # Get our variatinal parameters (E-step)
        log_phis = []
        otherPhis = []
        var_time = time.time()

        for id, doc in enumerate(sub_corpus):


            #phi, otherPhi = getVariationalVar(alpha_old, B_old, doc)
            log_phi, otherPhi = getVariationalVar(alpha_old, lb_old, doc)

            # print phi
            log_phis.append(log_phi)
            otherPhis.append(otherPhi)
            #print "For ", id
            #print "Log  phi:", log_phi
            #print "otherphi:", otherPhi





        # print otherPhis
        log_like_p1 = log_likelyhood(alpha_old, lb_old, vocabSize, sub_corpus, smoothing, log_phis, otherPhis)
        print "The normalized log-likelyhood is:", log_like_p1
        likes.append(log_like_p1)

        # save all
        print "Saving averything."
        # save_file = open(file_name, 'wr')
        # obj = {'log_phis': log_phis,
        #        'gammas': otherPhis,
        #        'alpha': alpha_new,
        #        'lambda': lb_new,
        #        'loglikelyhood': likes,
        #         "iter": nbIter}
        # pkl.dump(obj, save_file)
        # save_file.close()
        utils.save_LDA(file_name, log_phis, otherPhis, alpha_new, lb_new, likes, nbIter)


        if(np.abs(log_like_p1 - log_like) /np.abs(log_like_p1)  <= 0.00001): #we stop if the improvement is less then X%
            print "We converged!"
            break

        log_like = log_like_p1

        # M-step
        #B_new = getParamB(phis, sub_corpus, vocabSize)
        lb_new = getVarLambda(log_phis, smoothing, sub_corpus, vocabSize)
        alpha_new = getParamAlpha(alpha_old, otherPhis)

        # print "Our new B:"
        #print "New B:", B_new
        # print B_new - B_old
        #print "New alpha:", alpha_new

        #B_old = B_new
        lb_old = lb_new
        alpha_old = alpha_new

        print "Done interation {} in {:.1f} seconds".format(nbIter+1, time.time() - start_time)

    #return alpha_new, lb_new, likes
    return alpha_new, lb_new, likes, log_phis, otherPhis

def get_B(lb):
    B = np.zeros_like(lb)
    #B_log_expectation = dirichlet_log_expectation(lb)
    for k in range(B.shape[0]):
        B_log_expectation = dirichlet_log_expectation(lb[k])
        B[k] = np.exp(B_log_expectation - scipy.misc.logsumexp(B_log_expectation))
    return B