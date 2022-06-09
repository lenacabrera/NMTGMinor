# https://stackoverflow.com/questions/27412881/python-compare-n-grams-across-multiple-text-files

import nltk
from nltk.util import ngrams


text1 = 'Hello my name is Jason'
text2 = 'My name is not Mike'

n = 3
trigrams1 = list(ngrams(text1.lower().split(), n))
trigrams2 = list(ngrams(text2.lower().split(), n))

def compare(trigrams1, trigrams2):
    common=[]
    for grams1 in trigrams1:
        if grams1 in trigrams2:
            common.append(grams1)
    return common

print(compare(trigrams1, trigrams2))
