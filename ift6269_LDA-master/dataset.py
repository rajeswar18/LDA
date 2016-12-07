# Generate the data
import numpy as np
import os
import collections
import pickle as pkl

def generateRandomDocuments(alphas, B, nWordsPerDoc):

    nDocs = len(nWordsPerDoc)
    topicsDist = np.random.dirichlet(alphas, nDocs)

    topicsOfDocs = []
    wordsOfDocs = []

    for d in range(nDocs):
        tmpTopics = np.zeros((nWordsPerDoc[d],))
        tmpWords = np.zeros((nWordsPerDoc[d],))

        for n in range(nWordsPerDoc[d]):
            z = np.random.multinomial(1, topicsDist[d], size=1).argmax()
            w = np.random.multinomial(1, B[z], size=1).argmax()
            tmpTopics[n] = z
            tmpWords[n] = w

        topicsOfDocs.append(tmpTopics)
        wordsOfDocs.append(tmpWords)

    return topicsDist, topicsOfDocs, wordsOfDocs


def generateRandomCorpus(nDocs=50, nTopics=10, vocabSize=100, averageLength=100, alpha=None, B=None):
    np.random.seed(1993)

    if alpha == None:
        alpha =np.ones((nTopics,)) / nTopics

    if B == None:
        B = np.random.dirichlet(np.ones((vocabSize,)) / vocabSize,
                                (nTopics,))

    nWordsPerDoc = [averageLength] * nDocs
    #print nWordsPerDoc, alpha
    #return nWordsPerDoc, alpha
    return generateRandomDocuments(alpha, B, nWordsPerDoc), (alpha, B)


def load_20news(path="./20news-bydate", vocabSize=30000, cache_file="cache_data.pkl"):
    # http://qwone.com/~jason/20Newsgroups/

    def read_dataset(name, vocab = None):

        createVocab = vocab == None

        print "loading the {} set".format(name)
        # load the data
        data = open(os.path.join(path, "matlab/{}.data".format(name))).read()
        data = data.split("\n")[:-1]  # The last line is empty
        nb_lines = len(data)

        # make a np.array
        file_np = np.zeros((nb_lines, 3))
        for no, line in enumerate(data):

            try:
                line = [int(i) for i in line.split()]
                file_np[no] = np.array(line)
            except ValueError:
                pass

        if createVocab:
            vocab = collections.Counter()

        M = len(set(file_np[:, 0]))
        docs_pre = []

        # Get the vocabulary
        for d in range(1, M + 1):
            doc = file_np[np.where(file_np[:, 0] == d)]
            doc = doc[:, 1:].astype(int)  # remove the doc id

            if d % 1000 == 0:
                print "Done loading {} documents".format(d)

            doc = dict(doc)

            if createVocab:
                vocab.update(doc)

            #doc_array = np.array(list(collections.Counter(dict(doc)).elements()))
            docs_pre.append(doc)

        #Decollapse the documents
        docs_post = []
        if createVocab:
            vocab = set(dict(vocab.most_common(vocabSize)).keys())
            vocab = {key:i for i, key in enumerate(vocab)}

        for doc in docs_pre:

            doc_post = {vocab[key] : value for key, value in doc.iteritems() if key in vocab}
            doc_array = np.array(list(collections.Counter(doc_post).elements()))
            docs_post.append(doc_array)

        return docs_post, vocab

    def read_labels(name):
        print "loading the labels for {} set".format(name)
        labels = open(os.path.join(path, "matlab/{}.label".format(name))).read()
        labels = labels.split("\n")[:-1]  # The last line is empty
        labels = np.array(labels).astype(int)
        return labels

    def load_map(name):
        print "loading the map for {} set".format(name)

        mapping = open(os.path.join(path, "matlab/{}.map".format(name))).read()
        mapping = mapping.split("\n")[:-1]

        mapping = [[int(line.split()[1]), line.split()[0]] for line in mapping]
        return dict(mapping)

    def load_vocabulary():

        print "loading the vocabulary"
        vocab = open(os.path.join(path, "vocabulary.txt")).read()
        vocab = vocab.split('\n')[:-1]
        vocab = {i+1: v for i, v in enumerate(vocab)}


        return vocab

    if cache_file is not None and os.path.exists(cache_file):

        print "Loading the cache version..."

        f = open(cache_file, 'rb')
        obj = pkl.load(f)
        train_data = obj['train_data']
        train_labels = obj['train_labels']
        train_map = obj['train_map']

        test_data = obj['test_data']
        test_labels = obj['test_labels']
        test_map = obj['test_map']
        vocab = obj['vocab']

        print "Done!"
    else:
        train_data, real_voc = read_dataset('train')
        train_labels = read_labels('train')
        train_map = load_map('train')

        test_data, _= read_dataset('test', real_voc)
        test_labels = read_labels('test')
        test_map = load_map('test')

        vocab = load_vocabulary()
        vocab = {real_voc[key] : value for key, value in vocab.iteritems() if key in real_voc}


    if cache_file is not None and not os.path.exists(cache_file):

        print "Saving the cache version..."

        f = open(cache_file, 'wb')
        obj = {}
        obj['train_data'] = train_data
        obj['train_labels'] = train_labels
        obj['train_map'] = train_map

        obj['test_data'] = test_data
        obj['test_labels'] = test_labels
        obj['test_map'] = test_map
        obj['vocab'] = vocab

        pkl.dump(obj, f)

        print "Done!"

    return train_data, train_labels, train_map, test_data, test_labels, test_map, vocab