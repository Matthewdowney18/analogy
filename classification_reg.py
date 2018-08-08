import os
import random
import json
import vecto
import vecto.embeddings
import numpy as np
import matplotlib.pyplot as plt
import pandas

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def save_json(results, path):
    '''
    saves the jason file
    Args:
        results:the dictionary
        path:the path to save it tt
    '''
    basedir = os.path.dirname(path)
    os.makedirs(basedir, exist_ok=True)
    info = json.dumps(results, ensure_ascii=False, indent=4, sort_keys=False)
    file = open(path, 'w')
    file.write(info)
    file.close()

def frange(start, stop, step):
    num = start
    nums = []
    while num < stop:
        nums.append(float(int(num*100))/100)
        num += step
    return nums

def plot(results):
    print(results)
    df = pandas.DataFrame(results)
    df = df.T
    df.plot()
    plt.xlabel('inverse regularization')
    plt.ylabel('score')
    plt.show()
    plt.savefig('/home/mattd/Pycharm/word_'
                'classification/plots_reg/plot1.png')
    x = list(df.index.values)
    print(x)
    #plt.plot(df[:, 1])
    #plt.show()

    #ax2 = df.plot(kind='scatter', y='d', color='g', ax=ax1)
    print(df.T)



class Classifier:
    def __init__(self, normalize=True,
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
        self.neg_accuracy_total = list()
        self.pos_accuracy_total = list()

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

    def get_accuracy(self, true, pred):
        neg_examples = list()
        pos_examples = list()
        for i in range(0, len(pred)):
            if true[i] == 0:
                neg_examples.append(pred[i])
            else:
                pos_examples.append(pred[i])

        neg_accuracy = float(neg_examples.count(0)) / len(neg_examples)
        pos_accuracy = float(pos_examples.count(1)) / len(pos_examples)
        self.neg_accuracy_total.append(neg_accuracy)
        self.pos_accuracy_total.append(pos_accuracy)
        return neg_accuracy, pos_accuracy

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

    def get_results(self, predictions, y_test, data):
        results = dict()
        predictions2 = list()
        num_correct = int(0)

        for i, prediction in enumerate(predictions, 0):
            dictonary = dict()
            dictonary['word'] = data.test_words[i]
            dictonary['predicted class'] = str(predictions[i])
            dictonary['actual class'] = str(y_test[i])
            predictions2.append(dictonary)
            if y_test[i] == predictions[i]:
                num_correct += 1
        precision, recall, fscore = self.get_precision_recall(y_test,
                                                              predictions)
        neg_accuracy, pos_accuracy = self.get_accuracy(y_test,
                                                              predictions)
        results['neg_accuracy'] = neg_accuracy
        results['pos_accuracy'] = pos_accuracy
        results['precision'] = precision
        results['recall'] = recall
        results['fscore'] = fscore
        results['train_size'] = data.train_size
        results['test_size'] = data.test_size
        results['predictions'] = predictions2
        results['num_correct'] = num_correct
        return results

    def run(self, embedding_dir, dataset_dir, test_size, result_file_name):
        data = Data(embedding_dir)
        results = dict()
        results['embeddings'] = data.embedding.metadata
        results['classifier'] = self.make_dict()
        inverse_reg = frange(0, 1, 0.01)
        totals_dict = {}
        for reg in inverse_reg:
            for file_name in os.listdir(dataset_dir):
                category = file_name[:-3]
                file = open(dataset_dir + file_name, "r")

                data.make_data(file)
                words = list(data.dictionary.keys())

                random.shuffle(words)

                x_train, y_train, x_test, y_test = data.get_vectors(
                    words, test_size)

                if self.name_classifier == 'LR':
                    model_regression = LogisticRegression(
                        class_weight='balanced',
                        C=self.inverse_regularization_strength)
                elif self.name_classifier == "SVM":
                    model_regression = sklearn.svm.SVC(
                        C=reg,
                        kernel=self.name_kernel,
                        cache_size=200,
                        class_weight='balanced',
                        probability=True)
                elif self.name_classifier == "NN":
                    model_regression = MLPClassifier(
                        activation='logistic',
                        learning_rate='adaptive',
                        max_iter=5000,
                        hidden_layer_sizes=self.hidden_layer_sizes
                    )

                model_regression.fit(x_train, y_train)
                prediction = model_regression.predict(x_test)
                results[category] = self.get_results(prediction, y_test, data)
            results['totals'] = {}
            results['totals']['average negative accuracy'] = sum(
                self.neg_accuracy_total) / len(self.neg_accuracy_total)
            results['totals']['average positive accuracy'] = sum(
                self.pos_accuracy_total) / len(self.pos_accuracy_total)
            results['totals']['average precision'] = sum(
                self.precision_total) / len(self.precision_total)
            results['totals']['average recall'] = sum(
                self.recall_total) / len(self.recall_total)
            results['totals']['average f-score'] = sum(
                self.fscore_total) / len(self.fscore_total)
            totals_dict[str(reg)] = results['totals']
        plot(totals_dict)


class Data:
    def __init__(self, embedding_dir):
        self.test_words = []
        self.train_size = 0
        self.test_size = 0
        self.dictionary = {}

        self.embedding_dir = embedding_dir
        self.embedding = vecto.embeddings.load_from_dir(self.embedding_dir)

    def get_vectors(self, word_list, testsize):
        words = []
        for word in word_list:
            words.append(word)
        x_test = list()
        y_test = list()
        self.test_words = []
        for count_1 in range(0, testsize):
            odd = 0
            if count_1 % 2 == 0:
                odd = 1
            for key in words:
                if self.dictionary[key]['class'] == odd:
                    x_test.append(self.dictionary[key]['embedding'])
                    y_test.append(self.dictionary[key]['class'])
                    self.test_words.append(key)
                    words.remove(key)
                    break
        x_train = list()
        y_train = list()
        for key in words:
            x_train.append(self.dictionary[key]['embedding'])
            y_train.append(self.dictionary[key]['class'])
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
            rand = random.randint(0, len(words)-1)
            dictionary[words[rand]] = {}
            vector = list(self.embedding.get_vector(str(words[rand])))
            dictionary[str(words[rand])]['embedding'] = vector
            dictionary[words[rand]]['class'] = 0
        return dictionary

    def load_positive_examples(self, words):
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
        self.dictionary = self.load_positive_examples(x_true_words)
        self.dictionary.update(self.get_negative_examples(len(self.dictionary)))
        return


def main():
    embedding_dir = '/home/mattd/Pycharm/vecto_analogies/embeddings/' \
                    'structured_linear_cbow_500d'
    # embedding_dir = '/home/mattd/projects/tmp/pycharm_project_18/embeddings/
    # structured_linear_cbow_500d'
    dataset_dir = '/home/mattd/Pycharm/vecto_analogies/BATS/' \
                  'BATS_collective/'
    # dataset_dir = '/home/mattd/projects/tmp/pycharm_project_18/
    # BATS/BATS_collective/'
    classifier = Classifier(name_classifier='LR')

    classifier.run(embedding_dir, dataset_dir, 10, 'data9')

main()