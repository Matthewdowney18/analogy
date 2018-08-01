import vecto
import vecto.embeddings
import os
import requests
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
import json
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from nltk import word_tokenize


class Classification:
    def __init__(self, embedding_dir, dataset_dir):
        self.embedding_dir = embedding_dir
        self.dataset_dir = dataset_dir
        self.embedding = vecto.embeddings.load_from_dir(self.embedding_dir)

    def save_json(self, results, path):
        basedir = os.path.dirname(path)
        os.makedirs(basedir, exist_ok=True)
        s = json.dumps(results, ensure_ascii=False, indent=4, sort_keys=False)
        f = open(path, 'w')
        f.write(s)
        f.close()

    def get_results(self, predictions, Y_test, words):
        results = {}
        predictions2 = []
        num_correct = 0
        for i in range(0, len(predictions)):
            dictonary = {}
            dictonary['word'] = words[(-10 + i)]
            dictonary['predicted class'] = str(predictions[i])
            dictonary['actual class'] = str(Y_test[i])
            predictions2.append(dict)

            if Y_test[i] == predictions[i]:
                num_correct += 1

        results['predictions'] = predictions2
        results['num_correct'] = num_correct
        return results


    def get_vectors(self, list):
        x = []
        y= []
        for key in list:
            x.append(self.dict[key]['embedding'])
            y.append(self.dict[key]['class'])
        X = np.array(x)
        Y = np.array(y)
        return X, Y

    def get_negative_examples(self):
        dict = {}
        word_site = "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
        response = requests.get(word_site)
        words = [str(i).split("'")[1] for i in response.content.splitlines()]
        i = 0
        while i < 50:
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
        dict.update(self.get_negative_examples())
        return dict

    def run(self):
        results = {}
        results['embeddings'] = self.embedding.metadata
        for file_name in os.listdir(self.dataset_dir):
            category = file_name[:-3]
            print(category)
            file = open(self.dataset_dir + file_name, "r")
            self.dict = self.make_data(file, category)
            words = list(self.dict.keys())
            random.shuffle(words)
            X_train, Y_train = self.get_vectors(words[:-11])
            X_test, Y_test = self.get_vectors(words[-10:])

            model_regression = LogisticRegression(
                class_weight='balanced',
                C=1)
            model_regression.fit(X_train, Y_train)
            prediction = model_regression.predict(X_test)
            results[category] = self.get_results(prediction, Y_test, words)
            print(results)
        self.save_json(results, '/home/downey/PycharmProjects/word_classifacation/data.json')

def main():
    embedding_dir  ='/home/downey/PycharmProjects/vecto_analogies/embeddings/structured_linear_cbow_500d'
    #embedding_dir = '/home/mattd/projects/tmp/pycharm_project_18/embeddings/structured_linear_cbow_500d'
    dataset_dir = '/home/downey/PycharmProjects/vecto_analogies/BATS/BATS_collective/'
    #dataset_dir = '/home/mattd/projects/tmp/pycharm_project_18/BATS/BATS_collective/'
    classification = Classification(embedding_dir, dataset_dir)
    classification.run()

    print(embedding_dir)


main()