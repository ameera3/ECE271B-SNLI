
# coding: utf-8

# In[7]:

import pandas as pd
import numpy as np
from collections import defaultdict
import nltk
import string
from nltk.stem.porter import *
from sklearn import linear_model
import sys
import unicodedata
import string
import operator
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from scipy import sparse
from scipy.spatial.distance import cosine
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
stdout = sys.stdout
reload(sys)  
sys.setdefaultencoding('utf8')
sys.stdout = stdout


# In[4]:

import jsonlines
import numpy as np
import cPickle as pickle
import os

class Preprocessor:
    def __init__(self, dataDir = './'):
        self.data = []
        self.dataDir = dataDir 

    def isSampleValid(self, sample):
        return sample['gold_label'] != '-'
    
    def getDataFromJSONL(self, filename):
        numInvalid = 0
        with open(filename) as fp:
            reader = jsonlines.Reader(fp)
            for obj in reader.iter(type=dict, skip_invalid=True):
                if self.isSampleValid(obj):
                    sample = {}
                    sample['gold_label'] = obj['gold_label']
                    sample['sentence1'] = obj['sentence1']
                    sample['sentence2'] = obj['sentence2']
                    self.data.append(sample)
                else:
                    numInvalid+=1
        print len(self.data)
        print numInvalid
    
    


# In[5]:

pp = Preprocessor()
pp.getDataFromJSONL('./snli_1.0_train.jsonl')


# In[21]:

print pp.data[4]


# In[28]:

#nltk.download("stopwords")
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu
stop = set(stopwords.words('english'))
sentence1 = pp.data[4]['sentence1']
sentence1 = unicodedata.normalize('NFKD', sentence1).encode('ascii','ignore')
reference = [[i for i in sentence1.lower().split() if i not in stop]]
sentence2 = pp.data[4]['sentence2']
sentence2 = unicodedata.normalize('NFKD', sentence2).encode('ascii','ignore')
candidate = [i for i in sentence2.lower().split() if i not in stop]
print reference
print candidate
score = sentence_bleu(reference, candidate)
print(score)


# In[ ]:



