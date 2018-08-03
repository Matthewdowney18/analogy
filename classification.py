import vecto
import vecto.embeddings
import os
import requests
import random
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
import json
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from nltk import word_tokenize


class Classification:
    def __init__(self, embedding_dir, dataset_dir, normalize=True,
                 ignore_oov=True,
                 do_top5=True,
                 need_subsample=False,
                 size_cv_test=1,
                 set_aprimes_test=None,
                 inverse_regularization_strength=1,
                 exclude=True,
                 name_classifier='NN',
                 name_kernel="rbf",
                 hidden_layer_sizes = ()):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.normalize = normalize
        self.ignore_oov = ignore_oov
        self.do_top5 = do_top5
        self.need_subsample = need_subsample
        self.normalize = normalize
        self.size_cv_test = size_cv_test
        self.set_aprimes_test = set_aprimes_test
        self.inverse_regularization_strength = inverse_regularization_strength
        self.exclude = exclude
        self.name_classifier = name_classifier
        self.name_kernel = name_kernel

        self.precision_total = list()
        self.recall_total = list()
        self.fscore_total = list()

        self.embedding_dir = embedding_dir
        self.dataset_dir = dataset_dir
        self.embedding = vecto.embeddings.load_from_dir(self.embedding_dir)

    def make_dict(self):
        dict = {}
        dict["name_classifier"] = self.name_classifier
        dict["normalize"] = self.normalize
        dict["size_cv_test"] = self.size_cv_test
        dict["ignore_oov"] = self.ignore_oov
        dict["name_kernel"] = self.name_kernel
        dict["inverse_regularization_strength"] = \
            self.inverse_regularization_strength
        dict["exclude"] = self.exclude
        dict["do_top5"] = self.do_top5
        dict["hidden layer size "] = self.hidden_layer_sizes
        return dict

    def save_json(self, results, path):
        basedir = os.path.dirname(path)
        os.makedirs(basedir, exist_ok=True)
        s = json.dumps(results, ensure_ascii=False, indent=4, sort_keys=False)
        f = open(path, 'w')
        f.write(s)
        f.close()

    def plot(self, precision, recall, fscore, testsizes, category):
        plt.subplot(3, 1, 1)
        plt.plot(testsizes, precision)
        plt.title(category)
        plt.ylabel('precision')
        plt.subplot(3, 1, 2)
        plt.plot(testsizes, recall)
        plt.ylabel('recall')
        plt.subplot(3, 1, 3)
        plt.plot(testsizes, fscore)
        plt.ylabel('f-score')
        plt.xlabel('test size')
        plt.show()


    def get_precision_recall(self, true, pred):
        X = true - pred
        tp = 0.0
        fp = 0.0
        fn = 0.0
        for i in range(0, len(true)):
            if X[i] == 0 and true[i] == 1:
                tp += 1
            elif X[i] == 1:
                fn += 1
            elif X[i] == -1:
                fp += 1
        if tp != 0.0:
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            fscore = 2 * precision * recall / (precision + recall)
        else:
            precision = 0
            recall = 0
            fscore = 0

        self.precision_total.append(precision)
        self.recall_total.append(recall)
        self.fscore_total.append(fscore)

        return precision, recall, fscore


    def get_results(self, predictions, Y_test):
        results = {}
        predictions2 = []
        num_correct = 0

        for i in range(0, len(predictions)):
            dictonary = {}
            dictonary['word'] = self.test_words[i]
            dictonary['predicted class'] = str(predictions[i])
            dictonary['actual class'] = str(Y_test[i])
            predictions2.append(dictonary)
            if Y_test[i] == predictions[i]:
                num_correct += 1
        precision, recall, fscore  = self.get_precision_recall(Y_test,
                                                               predictions)
        #pr = precision_recall_fscore_support(Y_test, predictions, average='macro')
        results['precision'] = precision
        results['recall'] = recall
        results['fscore'] = fscore
        results['train_size'] = self.train_size
        results['test_size'] = self.test_size
        results['predictions'] = predictions2
        results['num_correct'] = num_correct
        return results


    def get_vectors(self, word_list, testsize):
        words = []
        for word in word_list:
            words.append(word)
        xtest = list()
        ytest = list()
        self.test_words = []
        for i in range(0, testsize):
            if i % 2 == 0:
                c = 1
            for key in words:
                if self.dict[key]['class'] == c:
                    xtest.append(self.dict[key]['embedding'])
                    ytest.append(self.dict[key]['class'])
                    self.test_words.append(key)
                    words.remove(key)
                    break
            c = 0

        xtrain = list()
        ytrain = list()
        for key in words:
            xtrain.append(self.dict[key]['embedding'])
            ytrain.append(self.dict[key]['class'])
        xtest = np.array(xtest)
        ytest = np.array(ytest)
        xtrain = np.array(xtrain)
        ytrain = np.array(ytrain)
        self.train_size = len(ytrain)
        self.test_size = len(xtest)
        return xtrain, ytrain, xtest, ytest

    def get_negative_examples(self, length):
        dict = {}
        word_site = "http://svnweb.freebsd.org/csrg/share/dict/" \
                    "words?view=co&content-type=text/plain"
        response = requests.get(word_site)
        words = [str(i).split("'")[1] for i in response.content.splitlines()]
        i = 0
        while i < length:
            rand = random.randint(0, len(words))
            if self.embedding.has_word(words[rand]):
                i+=1
                dict[words[rand]] = {}
                vector = list(self.embedding.get_vector(str(words[rand])))
                dict[str(words[rand])]['embedding'] = vector
                dict[words[rand]]['class'] = 0
        return dict

    def load_embedding(self, words):
        #for word in words:
        dict = {}
        for word in words:
            if self.embedding.has_word(word):
                dict[word] = {}
                vector = list(self.embedding.get_vector(str(word)))
                dict[str(word)]['embedding'] = vector
                dict[word]['class'] = 1
        return dict

    def make_data(self, file, category):
        x_true_words = [(line).split('\t')[1][:-1].split('/')[0] for line in file]
        dict = self.load_embedding(x_true_words)
        dict.update(self.get_negative_examples(len(dict)))
        return dict

    def run(self):
        results = {}
        results['embeddings'] = self.embedding.metadata
        results['classifier'] = self.make_dict()
        for file_name in os.listdir(self.dataset_dir):
            category = file_name[:-3]
            print(category)
            file = open(self.dataset_dir + file_name, "r")
            self.dict = self.make_data(file, category)
            words = list(self.dict.keys())

            random.shuffle(words)
            x_train, y_train, x_test, y_test = self.get_vectors(words, 10)
            if self.name_classifier == 'LR':
                model_regression = LogisticRegression(
                    class_weight='balanced',
                    C=1)
            if self.name_classifier == "SVM":
                model_regression = sklearn.svm.SVC(
                    C=self.inverse_regularization_strength,
                    kernel=self.name_kernel,
                    cache_size=200,
                    class_weight='balanced',
                    probability=True)
            if self.name_classifier == "NN":
                model_regression = MLPClassifier(
                    activation='logistic',
                    learning_rate='adaptive',
                    max_iter=5000,
                    hidden_layer_sizes=self.hidden_layer_sizes
                )

            model_regression.fit(x_train, y_train)
            prediction = model_regression.predict(x_test)
            results[category] = self.get_results(prediction, y_test)
        results['average precision'] = sum(self.precision_total)/len(
            self.precision_total)
        results['average recall'] = sum(self.recall_total) / len(
            self.recall_total)
        results['average f-score'] = sum(self.fscore_total) / len(
            self.fscore_total)
        self.save_json(results, '/home/downey/PycharmProjects/word_'
                                'classifacation/data1.json')

def main():
    embedding_dir  ='/home/downey/PycharmProjects/vecto_analogies/embeddings/' \
                    'structured_linear_cbow_500d'
    #embedding_dir = '/home/mattd/projects/tmp/pycharm_project_18/embeddings/
    # structured_linear_cbow_500d'
    dataset_dir = '/home/downey/PycharmProjects/vecto_analogies/BATS/' \
                  'BATS_collective/'
    #dataset_dir = '/home/mattd/projects/tmp/pycharm_project_18/
    # BATS/BATS_collective/'
    classification = Classification(embedding_dir,
                                    dataset_dir,
                                    name_classifier = 'LR')
    classification.run()

    print(embedding_dir)


main()