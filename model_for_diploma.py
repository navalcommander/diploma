
# coding: utf-8

# In[1]:


import csv
import pandas as pd
import numpy as np


# In[6]:


from __future__ import division, print_function
# отключим всякие предупреждения Anaconda
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# In[44]:


new_data = pd.read_csv('C:/Users/pro10/Downloads/qq.csv', sep=';',encoding="windows-1251", header=None)
#new_data1 = new_data1[[0,1]]
#new_data1.columns = ['text', 'label']


# In[39]:


new_data = pd.read_csv('C:/Users/pro10/Downloads/data.csv', sep=';',encoding="windows-1251", header=None)
new_data = new_data[[1,2]]
new_data.columns = ['text', 'label']


# In[60]:


new_data = pd.read_csv('C:/Users/pro10/Downloads/data.csv', sep=';',encoding="windows-1251", header=None)
new_data = new_data[[1,2]]
new_data.columns = ['text', 'label']


# In[61]:


new_data.drop(0, inplace=True)


# In[64]:


new_data['label'].dtype


# In[63]:


new_data['label'] = new_data['label'].astype(pd.np.int64)


# In[3]:


import pymorphy2
morph = pymorphy2.MorphAnalyzer()


# In[65]:


new_data.head()


# In[66]:


def text_cleaner(text):
    # к нижнему регистру
    text = text.lower()
    #if type(text) is str:
    #    text = text.lower()
    # оставляем в предложении только русские буквы (таким образом
    # удалим и ссылки, и имена пользователей, и пунктуацию и т.д.)
    alph = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    
    cleaned_text = ''
    for char in text:
        if (char.isalpha() and char[0] in alph) or (char == ' '):
            cleaned_text += char
        
    result = []
    for word in cleaned_text.split():
        # лемматизируем
        result.append(morph.parse(word)[0].normal_form)
                              
    return ' '.join(result)

new_data['text'] = new_data['text'].apply(text_cleaner)

new_data.to_csv('C:/Users/pro10/Downloads/cleaned_new_data.csv')


# In[67]:


new_data


# In[56]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit, cross_val_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

import pandas as pd


# In[57]:


ngram_schemes = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]


# In[68]:


for ngram_scheme in ngram_schemes:

    print('N-gram Scheme:', ngram_scheme)

    count_vectorizer = CountVectorizer(analyzer = "word", ngram_range=ngram_scheme) 
    tfidf_vectorizer = TfidfVectorizer(analyzer = "word", ngram_range=ngram_scheme)

    vectorizers = [count_vectorizer, tfidf_vectorizer]
    vectorizers_names = ['Count Vectorizer', 'TF-IDF Vectorizer']

    for i in range(len(vectorizers)):
        print(vectorizers_names[i])
        vectorizer = vectorizers[i]

        X = vectorizer.fit_transform(new_data['text'])
        y = new_data['label']

        cv = ShuffleSplit(len(y), n_iter=5, test_size=0.3, random_state=0)

            # наивный байес
        clf = MultinomialNB()
        NB_result = cross_val_score(clf, X, y, cv=cv).mean()

            # линейный классификатор
        clf = SGDClassifier()

        parameters = {
            'loss': ('log', 'hinge'),
            'penalty': ['none', 'l1', 'l2', 'elasticnet'],
            'alpha': [0.001, 0.0001, 0.00001, 0.000001]
        }

        gs_clf = GridSearchCV(clf, parameters, cv=cv, n_jobs=-1)
        gs_clf = gs_clf.fit(X, y)

        L_result = gs_clf.best_score_

        print('NB:', NB_result.mean())
        print('Linear:', L_result)
        print('Linear Parameters:', gs_clf.best_params_)
        print()

