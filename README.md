# analogy

classifying words n stuff and analogy

the bats data set contains lists of words and can be downloaded at this 

website:https://my.pcloud.com/publink/show?code=XZOn0J7Z8fzFMt7Tw1mGS6uI1SYfCfTyJQTV

That contains two lists: a list of words of the same type(noun, verb), and a second list of similar words that are in a
certain class. 

the first task, classifacation does the following in cllassifacation.py
the task is to determine if a word should be a part of the class (list on the right hand side of the file).
to do this, the word embedding for each of the words in the class, and for some random words that are not in that class 
are loaded(for positive and negative examples when training and testing your classifier) and then a classifier is 
trained with the embeddings. 

The library that is used to get the embeddings can be downloaded here:
http://vecto.readthedocs.io/en/docs/tutorial/basic.html

the embedding files can be found here:
http://vecto.readthedocs.io/en/docs/tutorial/basic.html
 
A library called sklearn is used to train and test classifiers:
http://scikit-learn.org/stable/ 