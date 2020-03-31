import os
import pickle
from time import time

import cv2
import numpy as np
from keras.preprocessing.image import (ImageDataGenerator, array_to_img,
                                       img_to_array, load_img)
from scipy.stats import itemfreq
from skimage.feature import local_binary_pattern
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import (MiniBatchDictionaryLearning, SparseCoder,
                                   dict_learning, sparse_encode)
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from sklearn.svm import SVC, LinearSVC
from tqdm import tqdm



bleeding_images_path = ""

normal_images_path = ""

image_name = []
for filename in os.listdir(bleeding_images_path):
    if(filename != '.ipynb_checkpoints'):
        image_name.append(os.path.join(bleeding_images_path, filename))
for filename in os.listdir(normal_images_path = ""):
    if(filename != '.ipynb_checkpoints'):
        image_name.append(os.path.join(normal_images_path, filename))

Y = np.append(np.ones(456), np.zeros(456))
Y = Y.reshape(len(image_name), 1)

X_train, X_test, y_train, y_test = train_test_split(image_name, Y, test_size=0.30, random_state=42)


print('finding the descriptors.........')
t0 = time()

keypoints_count_train = []
descriptors = []
for img_path in tqdm(X_train):
    img = cv2.imread(img_path)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    keypoints_count_train.append(len(kp))
    descriptors.append(des)
    
descriptors = np.concatenate(descriptors, axis=0)

dt = time() - t0
print('done in %.2fs.' % dt)

print('descriptors.shape : ', descriptors.shape)

print('finding the test_descriptors..........')

t0 = time()

keypoints_count_test = []
test_descriptors = []

for img_path in X_test:
    img = cv2.imread(img_path)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    keypoints_count_test.append(len(kp))
    test_descriptors.append(des)
    
test_descriptors = np.concatenate(test_descriptors, axis=0)

dt = time() - t0
print('done in %.2fs.' % dt)

print('test_descriptors.shape : ', test_descriptors.shape)


print('Learning the dictionary...')
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=100, batch_size=10, alpha=1, n_iter=250)
dictionary = dico.fit(descriptors).components_
dt = time() - t0
print('done in %.2fs.' % dt)

print('dictionary.shape : ', dictionary.shape)

print('finding the X_train_sparse_code..........')

t0 = time()
dico.set_params(transform_algorithm='omp')
X_train_sparse_code = dico.transform(descriptors)
dt = time() - t0
print('done in %.2fs.' % dt)

print('X_train_sparse_code.shape : ', X_train_sparse_code.shape)

print('finding the X_test_sparse_code..........')

t0 = time()
dico.set_params(transform_algorithm='omp')
X_test_sparse_code = dico.transform(test_descriptors)
dt = time() - t0
print('done in %.2fs.' % dt)

print('X_test_sparse_code.shape : ', X_test_sparse_code.shape)


with open('descriptors', 'wb') as f:
    pickle.dump(descriptors, f)
    
with open('test_descriptors', 'wb') as f:
    pickle.dump(test_descriptors, f)

with open('X_train_sparse_code', 'wb') as f:
    pickle.dump(X_train_sparse_code, f)

with open('X_test_sparse_code', 'wb') as f:
    pickle.dump(X_test_sparse_code, f)

with open('dictionary', 'wb') as f:
    pickle.dump(dictionary, f)
    
with open('Dataset', 'wb') as f:
    pickle.dump([X_train, X_test, y_train, y_test], f)
    
print("...................saved all variables.............")


final_vectors_list_train = []
t0 = time()

last_keypoints_count_train = 0

for keypoints_count in keypoints_count_train:
    vector = np.amax(X_train_sparse_code[last_keypoints_count_train : keypoints_count + last_keypoints_count_train], axis = 0)
    final_vectors_list_train.append(vector)
    
    last_keypoints_count_train += keypoints_count
    
final_vectors_list_train = np.array(final_vectors_list_train)

dt = time() - t0
print('done in %.2fs.' % dt)

final_vectors_list_test = []
t0 = time()

last_keypoints_count_test = 0

for keypoints_count in keypoints_count_test:
    vector = np.amax(X_test_sparse_code[last_keypoints_count_test : keypoints_count + last_keypoints_count_test], axis = 0)
    final_vectors_list_test.append(vector)
    
    last_keypoints_count_test += keypoints_count
    
final_vectors_list_test = np.array(final_vectors_list_test)

dt = time() - t0
print('done in %.2fs.' % dt)

final_vectors_list_train.shape

final_vectors_list_test.shape

final_vectors_list_test


clf = LinearSVC()
clf.fit(final_vectors_list_train, y_train)
predictions = clf.predict(final_vectors_list_test)

print(classification_report(y_test, predictions))


with open('X_train_sparse_code_2', 'rb') as f:
    # Python 3: open(..., 'rb')
    X_train_sparse_code = pickle.load(f)
    
with open('X_test_sparse_code_2', 'rb') as f:
    # Python 3: open(..., 'rb')
    X_test_sparse_code = pickle.load(f)

with open('Dataset_2', 'rb') as f:# Python 3: open(..., 'rb')
     X_train, X_test, y_train, y_test = pickle.load(f)


print('finding the descriptors.........')
t0 = time()

keypoints_count_train = []
descriptors = []
for img_path in tqdm(X_train):
    img = cv2.imread(img_path)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    keypoints_count_train.append(len(kp))
    descriptors.append(des)
    
descriptors = np.concatenate(descriptors, axis=0)

dt = time() - t0
print('done in %.2fs.' % dt)

print('finding the test_descriptors..........')

t0 = time()

keypoints_count_test = []
test_descriptors = []

for img_path in tqdm(X_test):
    img = cv2.imread(img_path)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    keypoints_count_test.append(len(kp))
    test_descriptors.append(des)
    
test_descriptors = np.concatenate(test_descriptors, axis=0)

dt = time() - t0
print('done in %.2fs.' % dt)

with open('keypoints_count_train_2', 'wb') as f:
    pickle.dump(keypoints_count_train, f)

with open('keypoints_count_test_2', 'wb') as f:
    pickle.dump(keypoints_count_test, f)
    
final_vectors_list_train = []
t0 = time()

last_keypoints_count_train = 0

for keypoints_count in tqdm(keypoints_count_train):
    vector = np.amax(X_train_sparse_code[last_keypoints_count_train : keypoints_count + last_keypoints_count_train], axis = 0)
    final_vectors_list_train.append(vector)
    
    last_keypoints_count_train += keypoints_count
    
final_vectors_list_train = np.array(final_vectors_list_train)

dt = time() - t0
print('done in %.2fs.' % dt)

print('final_vectors_list_train.shape : ', final_vectors_list_train.shape)

final_vectors_list_test = []
t0 = time()

last_keypoints_count_test = 0

for keypoints_count in tqdm(keypoints_count_test):
    vector = np.amax(X_test_sparse_code[last_keypoints_count_test : keypoints_count + last_keypoints_count_test], axis = 0)
    final_vectors_list_test.append(vector)
    
    last_keypoints_count_test += keypoints_count
    
final_vectors_list_test = np.array(final_vectors_list_test)

dt = time() - t0
print('done in %.2fs.' % dt)

print('final_vectors_list_test.shape : ', final_vectors_list_test.shape)

with open('final_vectors_list_train_2', 'wb') as f:
    pickle.dump(final_vectors_list_test, f)

with open('final_vectors_list_train_2', 'wb') as f:
    pickle.dump(final_vectors_list_test, f)


clf = LinearSVC(C=0.001, loss='hinge', penalty='l2')
clf.fit(final_vectors_list_train, y_train)
predictions = clf.predict(final_vectors_list_test)

print(classification_report(y_test, predictions))

clf_2 = SVC(C=10.0, kernel='rbf', gamma=0.00001)

clf_2.fit(final_vectors_list_train, y_train)
predictions_2 = clf_2.predict(final_vectors_list_test)

print(classification_report(y_test, predictions_2))


radius = 3
no_points = 8 * radius

lbp_features_train = []

for img_path in tqdm(X_train):
    im = cv2.imread(img_path)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
    x = itemfreq(lbp.ravel())
    hist = x[:, 1] / sum(x[:, 1])
    lbp_features_train.append(hist)
    
lbp_features_train = np.array(lbp_features_train)
lbp_features_train.shape

X_train_lbp = np.concatenate((final_vectors_list_train, lbp_features_train), axis = 1)
X_train_lbp.shape

lbp_features_test = []

for img_path in tqdm(X_test):
    im = cv2.imread(img_path)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
    x = itemfreq(lbp.ravel())
    hist = x[:, 1] / sum(x[:, 1])
    lbp_features_test.append(hist)
    
X_test_lbp = np.concatenate((final_vectors_list_test, lbp_features_test), axis = 1)
X_test_lbp.shape

clf_3 = SVC(C=1.0, kernel='rbf', gamma=0.00001)

clf_3.fit(X_train_lbp, y_train)
predictions_3 = clf_3.predict(X_test_lbp)

print(classification_report(y_test, predictions_3))

final_vectors_list_train.shape

with open('X_train_lbp', 'wb') as f:
    pickle.dump(X_train_lbp, f)

with open('X_test_lbp', 'wb') as f:
    pickle.dump(X_test_lbp, f)
    
with open('lbp_features_train', 'wb') as f:
    pickle.dump(lbp_features_train, f)
    
with open('lbp_features_test', 'wb') as f:
    pickle.dump(lbp_features_test, f)
