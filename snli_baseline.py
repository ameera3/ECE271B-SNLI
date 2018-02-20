
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


# In[80]:

#nltk.download("stopwords")
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu
stop = set(stopwords.words('english'))
BLEU_Score_List = []
for j in range(len(pp.data)):
    weights = [1./4., 1./4., 1./4., 1./4.]
    sentence1 = pp.data[j]['sentence1']
    sentence1 = unicodedata.normalize('NFKD', sentence1).encode('ascii','ignore')
    reference = [[i for i in sentence1.lower().split() if i not in stop]]
    sentence2 = pp.data[j]['sentence2']
    sentence2 = unicodedata.normalize('NFKD', sentence2).encode('ascii','ignore')
    candidate = [i for i in sentence2.lower().split() if i not in stop]
    length = min([len(reference[0]), len(candidate)]) 
    if length == 0:
        BLEU_Score_List.append(0)
    else:
        if length < 4:
            weights = ( 1. / length ,) * length
        score = sentence_bleu(reference, candidate, weights)
        BLEU_Score_List.append(score)      


# In[81]:

get_ipython().magic(u'store BLEU_Score_List > BLEU_Score_List.txt')


# In[109]:

BLEU_Score_Array = np.array(BLEU_Score_List)


# In[115]:

BLEU_Score_Array = BLEU_Score_Array[:, np.newaxis]


# In[93]:

Target_List = [pp.data[j]['gold_label'] for j in range(len(pp.data))]


# In[94]:

get_ipython().magic(u'store Target_List > Target_List.txt')


# In[95]:

Target_Array = np.array(Target_List)


# In[104]:

print Target_Array.shape


# In[116]:

print BLEU_Score_Array.shape


# In[120]:

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(BLEU_Score_Array, Target_Array).predict(BLEU_Score_Array)
y_pred_list = y_pred.tolist()
get_ipython().magic(u'store y_pred_list > y_pred_list.txt')


# In[121]:

print("Number of mislabeled points out of a total %d points : %d" % (BLEU_Score_Array.shape[0],(Target_Array != y_pred).sum()))


# In[123]:

print 549367 - 331625


# In[124]:

print 217742./549367.


# In[ ]:



