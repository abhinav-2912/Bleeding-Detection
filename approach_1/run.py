
# coding: utf-8

# In[1]:


import os
import pickle
from time import time

import matplotlib.image as mpimg
# get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import (ImageDataGenerator, array_to_img,
                                       img_to_array, load_img)
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import (MiniBatchDictionaryLearning, SparseCoder,
                                   dict_learning, sparse_encode)
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC

# In[2]:


bleeding_images_path = ""


# In[3]:


bleeding_images = []

for filename in os.listdir(bleeding_images_path):
    if(filename != '.ipynb_checkpoints'):
        img_path = os.path.join(bleeding_images_path, filename)
        img = load_img(img_path)
        bleeding_images.append(img_to_array(img))
    
bleeding_images = np.array(bleeding_images)

print(bleeding_images.shape)


# In[4]:


normal_images_path = ""


# In[5]:


normal_images = []

for filename in os.listdir(normal_images_path):
    if(filename != '.ipynb_checkpoints'):
        img_path = os.path.join(normal_images_path, filename)
        img = load_img(img_path)
        normal_images.append(img_to_array(img))
    
normal_images = np.array(normal_images)

print(normal_images.shape)


# In[6]:


X = np.append(bleeding_images, normal_images, axis=0)

# Labels :
# Bleeding_images = 1   Normal_images = 0

# In[8]:


Y = np.append(np.ones(bleeding_images.shape[0]), np.zeros(normal_images.shape[0]))
Y.reshape(X.shape[0], 1)

# In[10]:


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

# from sklearn.utils import shuffle
X, Y = shuffle_in_unison(X, Y)


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)


# In[12]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[13]:


x1, x2, x3, x4 = X_train.shape
X_train = X_train.flatten().reshape(x1, x2 * x3 * x4)

# In[15]:


x1, x2, x3, x4 = X_test.shape
X_test = X_test.flatten().reshape(x1, x2 * x3 * x4)


# In[18]:


print('Learning the dictionary...')
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=50, batch_size=10, alpha=3, n_iter=200)
V = dico.fit(X_train).components_
dt = time() - t0
print('done in %.2fs.' % dt)


# In[26]:


t0 = time()
dico.set_params(transform_algorithm='omp')
X_train_sparse_code = dico.transform(X_train)
dt = time() - t0
print('done in %.2fs.' % dt)


# In[28]:


t0 = time()
dico.set_params(transform_algorithm='omp')
X_test_sparse_code = dico.transform(X_test)
dt = time() - t0
print('done in %.2fs.' % dt)

# In[40]:


clf = LinearSVC()
clf.fit(X_train_sparse_code, y_train)
predictions = clf.predict(X_test_sparse_code)


# In[42]:


print(classification_report(y_test, predictions))



with open('Dataset', 'wb') as f:
    pickle.dump([X_train, X_test, y_train, y_test], f)
    
with open('dictionary', 'wb') as f:
    pickle.dump(V, f)

with open('X_train_sparse_code', 'wb') as f:
    pickle.dump(X_train_sparse_code, f)

with open('X_test_sparse_code', 'wb') as f:
    pickle.dump(X_test_sparse_code, f)

    
print("...................saved all variables.............")
