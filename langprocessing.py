import nltk

import numpy as np

#nltk.download('punkt')

from nltk.stem.porter import PorterStemmer

stemmer=PorterStemmer()

def tokenization(sentence):
    return nltk.word_tokenize(sentence)

def stemming(word):
    return stemmer.stem(word.lower())


def word_dir(tokenize_sentence, all_words):
    tokenize_sentence=[stemming(w) for w in tokenize_sentence]
    bag=np.zeros(len(all_words),dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenize_sentence:
            bag[idx]=1.0
    return bag

