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
                 class_weight="balanced",
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
        self.class_weight=class_weight

        self.stats = {}
        self.cnt_total_correct = 0
        self.cnt_total_total = 0


        self.results = {}
        self.embedding_dir = embedding_dir
        self.dataset_dir = dataset_dir
        self.embedding = vecto.embeddings.load_from_dir(self.embedding_dir)

    def make_dict(self):
        d = dict()
        d["name_classifier"] = self.name_classifier
        d["normalize"] = self.normalize
        d["size_cv_test"] = self.size_cv_test
        d["ignore_oov"] = self.ignore_oov
        d["name_kernel"] = self.name_kernel
        d["inverse_regularization_strength"] = self.inverse_regularization_strength
        d["exclude"] = self.exclude
        d["do_top5"] = self.do_top5
        d["hidden layer size "] =self.hidden_layer_sizes
        return d

    def save_json(self, results, path):
        basedir = os.path.dirname(path)
        os.makedirs(basedir, exist_ok=True)
        s = json.dumps(results, ensure_ascii=False, indent=4, sort_keys=False)
        f = open(path, 'w')
        f.write(s)
        f.close()

    def get_result(self, test1, scores, class_score, score_sim, p_train=[]):
        ids_max = np.argsort(scores)[::-1]
        result = dict()
        cnt_answers_to_report = 6
        extr = ""
        if len(p_train) == 1:
            extr = "as {} is to {}".format(p_train[0][1], p_train[0][0])
            set_exclude = set([p_train[0][0]]) | set(p_train[0][1])
        else:
            set_exclude = set()
        set_exclude.add(test1['word']['word'])
        result["question verbose"] = "What is to {} {}".format(test1['word']['word'], extr)
        result["b"] = test1['word']['word']
        result["expected answer"] = test1['class_word']['word']
        result["predictions"] = []
        result['set_exclude'] = [e for e in set_exclude]

        cnt_reported = 0
        for i in ids_max[:10]:
            prediction = dict()
            ans = self.embedding.vocabulary.get_word_by_id(i)
            if self.exclude and (ans in set_exclude):
                continue
            cnt_reported += 1
            prediction["score"] = float(scores[i])
            prediction["answer"] = ans
            if ans in test1['class_word']['word']:
                prediction["hit"] = True
            else:
                prediction["hit"] = False
            result["predictions"].append(prediction)
            if cnt_reported >= cnt_answers_to_report:
                break
        rank = 0
        for i in range(ids_max.shape[0]):
            ans = self.embedding.vocabulary.get_word_by_id(ids_max[i])
            if self.exclude and (ans in set_exclude):
                continue
            if ans in test1['class_word']['word']:
                break
            rank += 1
        result["rank"] = rank
        if rank == 0:
            self.cnt_total_correct += 1
        self.cnt_total_total += 1

        # vec_b_prime = self.embs.get_vector(p_test_one[1][0])
        # result["closest words to answer 1"] = get_distance_closest_words(vec_b_prime,1)
        # result["closest words to answer 5"] = get_distance_closest_words(vec_b_prime,5)
        # where prediction lands:
        ans = self.embedding.vocabulary.get_word_by_id(ids_max[0])
        if ans == test1['word']['word']:
            result["landing_b"] = True
        else:
            result["landing_b"] = False

        if ans in test1['class_word']['word']:
            result["landing_b_prime"] = True

        else:
            result["landing_b_prime"] = False
        return result

    def test(self, X_train, Y_train, test, category):
        if self.name_classifier == 'LR':
            model_regression = LogisticRegression(
                class_weight=self.class_weight,
                C=self.inverse_regularization_strength)
        if self.name_classifier == 'NN':
            model_regression = MLPClassifier(
                activation='logistic',
                learning_rate='adaptive',
                max_iter=5000,
                hidden_layer_sizes=self.hidden_layer_sizes)

        model_regression.fit(X_train, Y_train)

        if self.name_classifier == 'NN':
            plt.plot(model_regression.loss_curve_)
            plt.show()
        class_score = model_regression.predict_proba(self.embedding.matrix)[:, 1]
        details = []
        for test1 in test:
            word_vec = test1['word']['embedding']
            class_word_vec = test1['class_word']['embedding']
            v = word_vec / np.linalg.norm(word_vec)
            emb2 = self.embedding
            emb2.normalize()
            score_sim = v @ emb2.matrix.T
            scores = score_sim * class_score
            result = self.get_result(test1, scores, class_score, score_sim)
            result["similarity b to b_prime cosine"] = float(self.embedding.cmp_vectors(word_vec, class_word_vec))
            details.append(result)
        score = float(self.cnt_total_correct) / self.cnt_total_total
        details.append({'score': score})
        self.results[category] = details

    def get_vectors(self, m):
        x = []
        y= []
        for key in m:
            x.append(key['embedding'])
            y.append(key['class'])
        X = np.array(x)
        Y = np.array(y)
        return X, Y

    def get_negative_examples(self):
        list_words = list()
        word_site = "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
        response = requests.get(word_site)
        words = [str(i).split("'")[1] for i in response.content.splitlines()]
        for i in range(0, self.pair_num):
            rand = random.randint(0, len(words))
            if self.embedding.has_word(words[rand]):
                word = dict()
                word['word'] = words[rand]
                word['embedding'] = list(self.embedding.get_vector(str(words[rand])))
                word['class'] = '0'
                list_words.append(word)
        return list_words

    def load_bats_embeddings(self, file):
        pairs = list()
        self.pair_num = 0
        for line in file:
            word = (line).split('\t')[0]
            class_word = line.split('\t')[1][:-1].split('/')[0]
            if self.embedding.has_word(word) and self.embedding.has_word(class_word):
                pair = {'word': {}, 'class_word': {}}
                pair['word']['word'] = word
                pair['word']['embedding'] = list(self.embedding.get_vector(str(word)))
                pair['word']['class'] = '0'
                pair['class_word']['word'] = class_word
                pair['class_word']['embedding'] = list(self.embedding.get_vector(str(class_word)))
                pair['class_word']['class'] = '1'
                pairs.append(pair)
                self.pair_num += 1
        random.shuffle(pairs)
        return pairs

    def make_data(self, file, category):
        pairs = self.load_bats_embeddings(file)
        negative_examples = self.get_negative_examples()
        words = list()
        for pair in pairs[:-6]:
            words.append(pair['class_word'])
        for example in negative_examples:
            words.append(example)
        random.shuffle(words)
        X_train, Y_train = self.get_vectors(words)
        test = pairs[-5:]
        return X_train, Y_train, test

    def run(self):
        self.results['embeddings'] = self.embedding.metadata
        self.results['test_setup'] = self.make_dict()
        for file_name in os.listdir(self.dataset_dir)[:5]:
            self.cnt_total_total = 0
            self.cnt_total_correct = 0
            category = file_name[:-3]
            print(category)
            file = open(self.dataset_dir + file_name, "r")
            X_train, Y_train, test = self.make_data(file, category)
            self.test(X_train, Y_train, test, category)
        self.save_json(self.results, '/home/downey/PycharmProjects/word_classifacation/data3.json')

def main():
    embedding_dir  ='/home/downey/PycharmProjects/vecto_analogies/embeddings/structured_linear_cbow_500d'
    #embedding_dir = '/home/mattd/projects/tmp/pycharm_project_18/embeddings/structured_linear_cbow_500d'
    dataset_dir = '/home/downey/PycharmProjects/vecto_analogies/BATS/BATS_collective/'
    #dataset_dir = '/home/mattd/projects/tmp/pycharm_project_18/BATS/BATS_collective/'
    classification = Classification(embedding_dir, dataset_dir)
    classification.run()

    print(embedding_dir)

main()