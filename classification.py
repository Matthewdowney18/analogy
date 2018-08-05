import os
import random
import json
import vecto
import vecto.embeddings
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def save_json(results, path):
    basedir = os.path.dirname(path)
    os.makedirs(basedir, exist_ok=True)
    info = json.dumps(results, ensure_ascii=False, indent=4, sort_keys=False)
    file = open(path, 'w')
    file.write(info)
    file.close()


def plot(precision, recall, fscore, testsizes, category):
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
                 hidden_layer_sizes=()):
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
        self.test_words = []
        self.train_size = 0
        self.test_size = 0
        self.dict = {}
        self.embedding_dir = embedding_dir
        self.dataset_dir = dataset_dir
        self.embedding = vecto.embeddings.load_from_dir(self.embedding_dir)

    def make_dict(self):
        dictionary = dict()
        dictionary["name_classifier"] = self.name_classifier
        dictionary["normalize"] = self.normalize
        dictionary["size_cv_test"] = self.size_cv_test
        dictionary["ignore_oov"] = self.ignore_oov
        dictionary["name_kernel"] = self.name_kernel
        dictionary["inverse_regularization_strength"] = \
            self.inverse_regularization_strength
        dictionary["exclude"] = self.exclude
        dictionary["do_top5"] = self.do_top5
        dictionary["hidden layer size "] = self.hidden_layer_sizes
        return dictionary

    def get_precision_recall(self, true, pred):
        difference = true - pred
        true_positive = 0.0
        false_positive = 0.0
        false_negative = 0.0
        for i in range(0, len(true)):
            if difference[i] == 0 and true[i] == 1:
                true_positive += 1
            elif difference[i] == 1:
                false_negative += 1
            elif difference[i] == -1:
                false_positive += 1
        if true_positive != 0.0:
            precision = true_positive/(true_positive+false_positive)
            recall = true_positive/(true_positive+false_negative)
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
        precision, recall, fscore = self.get_precision_recall(Y_test,
                                                               predictions)
        # pr = precision_recall_fscore_support(Y_test,
        #  predictions, average='macro')
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
        x_test = list()
        y_test = list()
        self.test_words = []
        for count_1 in range(0, testsize):
            if count_1 % 2 == 0:
                odd = 1
            for key in words:
                if self.dict[key]['class'] == odd:
                    x_test.append(self.dict[key]['embedding'])
                    y_test.append(self.dict[key]['class'])
                    self.test_words.append(key)
                    words.remove(key)
                    break
            odd = 0

        x_train = list()
        y_train = list()
        for key in words:
            x_train.append(self.dict[key]['embedding'])
            y_train.append(self.dict[key]['class'])
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        self.train_size = len(y_train)
        self.test_size = len(x_test)
        return x_train, y_train, x_test, y_test

    def get_negative_examples(self, length):
        dictionary = dict()
        words = self.embedding.vocabulary.lst_words
        for i in range(0, length):
            rand = random.randint(0, len(words))
            dictionary[words[rand]] = {}
            vector = list(self.embedding.get_vector(str(words[rand])))
            dictionary[str(words[rand])]['embedding'] = vector
            dictionary[words[rand]]['class'] = 0
        return dictionary

    def load_embedding(self, words):
        # for word in words:
        dictionary = dict()
        for word in words:
            if self.embedding.has_word(word):
                dictionary[word] = {}
                vector = list(self.embedding.get_vector(str(word)))
                dictionary[str(word)]['embedding'] = vector
                dictionary[word]['class'] = 1
        return dictionary

    def make_data(self, file):
        x_true_words = [line.split('\t')[1][:-1].split('/')[0] for line in file]
        dict = self.load_embedding(x_true_words)
        dict.update(self.get_negative_examples(len(dict)))
        return dict

    def run(self):
        results = dict()
        results['embeddings'] = self.embedding.metadata
        results['classifier'] = self.make_dict()
        for file_name in os.listdir(self.dataset_dir):
            category = file_name[:-3]
            print(category)
            file = open(self.dataset_dir + file_name, "r")
            self.dict = self.make_data(file)
            words = list(self.dict.keys())

            random.shuffle(words)
            x_train, y_train, x_test, y_test = self.get_vectors(words, 10)
            if self.name_classifier == 'LR':
                model_regression = LogisticRegression(
                    class_weight='balanced',
                    C=self.inverse_regularization_strength)
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
        save_json(results, '/home/downey/PycharmProjects/word_'
                                'classifacation/data7.json')


def main():
    embedding_dir = '/home/downey/PycharmProjects/vecto_analogies/embeddings/' \
                    'structured_linear_cbow_500d'
    # embedding_dir = '/home/mattd/projects/tmp/pycharm_project_18/embeddings/
    # structured_linear_cbow_500d'
    dataset_dir = '/home/downey/PycharmProjects/vecto_analogies/BATS/' \
                  'BATS_collective/'
    # dataset_dir = '/home/mattd/projects/tmp/pycharm_project_18/
    # BATS/BATS_collective/'
    classification = Classification(embedding_dir,
                                    dataset_dir,
                                    name_classifier='LR',
                                    inverse_regularization_strength=1)
    classification.run()
    print(embedding_dir)


main()
