
import vb_lda
from scipy import misc
import numpy as np
from collections import Counter

def get_mostlikely_topic(doc, alpha, lb, log_phi=None, gamma=None, voc=None):

    if log_phi is None:
        log_phi, gamma = vb_lda.getVariationalVar(alpha, lb, doc)

    k = len(alpha)
    topics = {i :Counter() for i in range(k)}
    for n in range(len(doc)):
        word = doc[n]
        probs = log_phi[:, n]
        probs -= misc.logsumexp(probs)
        if np.max(np.exp(probs)) > 0.9:
            if voc is not None:
                topics[np.argmax(np.exp(probs))].update({voc[word]:1})
            else:
                topics[np.argmax(np.exp(probs))].update({word:1})

                # topics[voc[word]] = [np.argmax(np.exp(probs)), np.max(np.exp(probs))]

    return topics, gamma


def get_topics(data, alpha, lb, log_phis=None, gammas=None, voc=None):
    k = len(alpha)
    topics = {i: Counter() for i in range(k)}
    for i, doc in enumerate(data):

        if log_phis is not None:
            topic, gamma = get_mostlikely_topic(doc, alpha, lb,log_phi=log_phis[i], gamma=gammas[i], voc=voc)
        else:
            topic, gamma = get_mostlikely_topic(doc, alpha, lb, voc=voc)

        for i in range(k):
            topics[i].update(topic[i])

    return topics


def save_LDA(file_name, log_phis, gammas, alpha, lb, likes, nbIter):

    #save_file = open(file_name, 'wr')
    # obj = {'log_phis': log_phis,
    #        'gammas': gammas,
    #        'alpha': alpha,
    #        'lambda': lb,
    #        'loglikelyhood': likes,
    #        "iter": nbIter}
    #print len(log_phis)
    #print log_phis[0].shape
    np.savez_compressed(file_name,
                        *log_phis,
                        gammas=np.array(gammas),
                        alpha=alpha,
                        lb=lb,
                        loglikelyhood=np.array(likes),
                        iter=nbIter)

    #pkl.dump(obj, save_file)
    #save_file.close()

def load_LDA(file_name):

    obj = np.load(file_name)
    gammas = obj['gammas']
    alpha = obj['alpha']
    lb = obj['lb']
    likes = obj['loglikelyhood']
    nbIter = obj['iter']

    log_phis = []
    i = 0
    cont = True

    while cont:
        try:
            tmp = obj['arr_{}'.format(i)]
            log_phis.append(tmp)
            i+=1
        except:
            cont = False


    return log_phis, gammas, alpha, lb, likes, nbIter


