import os
import pickle
from time import time

import cv2
import numpy as np
from keras.preprocessing.image import (ImageDataGenerator, array_to_img,
                                       img_to_array, load_img)
from matplotlib import pyplot
from scipy import misc, signal
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import (MiniBatchDictionaryLearning, SparseCoder,
                                   dict_learning, sparse_encode)
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC
from tqdm import tqdm

'''
dsift.py: this function implements some basic functions that 
does dense sift feature extraction.
The descriptors are defined in a similar way to the one used in
Svetlana Lazebnik's Matlab implementation, which could be found
at:
http://www.cs.unc.edu/~lazebnik/
Yangqing Jia, jiayq@eecs.berkeley.edu
'''


# sift features
Nangles = 8
Nbins = 4
Nsamples = Nbins**2
alpha = 9.0
angles = np.array(range(Nangles))*2.0*np.pi/Nangles

def gen_dgauss(sigma):
    '''
    generating a derivative of Gauss filter on both the X and Y
    direction.
    '''
    fwid = np.int(2*np.ceil(sigma))
    G = np.array(range(-fwid,fwid+1))**2
    G = G.reshape((G.size,1)) + G
    G = np.exp(- G / 2.0 / sigma / sigma)
    G /= np.sum(G)
    GH,GW = np.gradient(G)
    GH *= 2.0/np.sum(np.abs(GH))
    GW *= 2.0/np.sum(np.abs(GW))
    return GH,GW

class DsiftExtractor:
    '''
    The class that does dense sift feature extractor.
    Sample Usage:
        extractor = DsiftExtractor(gridSpacing,patchSize,[optional params])
        feaArr,positions = extractor.process_image(Image)
    '''
    def __init__(self, gridSpacing, patchSize,
                 nrml_thres = 1.0,\
                 sigma_edge = 0.8,\
                 sift_thres = 0.2):
        '''
        gridSpacing: the spacing for sampling dense descriptors
        patchSize: the size for each sift patch
        nrml_thres: low contrast normalization threshold
        sigma_edge: the standard deviation for the gaussian smoothing
            before computing the gradient
        sift_thres: sift thresholding (0.2 works well based on
            Lowe's SIFT paper)
        '''
        self.gS = gridSpacing
        self.pS = patchSize
        self.nrml_thres = nrml_thres
        self.sigma = sigma_edge
        self.sift_thres = sift_thres
        # compute the weight contribution map
        sample_res = self.pS / np.double(Nbins)
        sample_p = np.array(range(self.pS))
        sample_ph, sample_pw = np.meshgrid(sample_p,sample_p)
        sample_ph.resize(sample_ph.size)
        sample_pw.resize(sample_pw.size)
        bincenter = np.array(range(1,Nbins*2,2)) / 2.0 / Nbins * self.pS - 0.5 
        bincenter_h, bincenter_w = np.meshgrid(bincenter,bincenter)
        bincenter_h.resize((bincenter_h.size,1))
        bincenter_w.resize((bincenter_w.size,1))
        dist_ph = abs(sample_ph - bincenter_h)
        dist_pw = abs(sample_pw - bincenter_w)
        weights_h = dist_ph / sample_res
        weights_w = dist_pw / sample_res
        weights_h = (1-weights_h) * (weights_h <= 1)
        weights_w = (1-weights_w) * (weights_w <= 1)
        # weights is the contribution of each pixel to the corresponding bin center
        self.weights = weights_h * weights_w
        #pyplot.imshow(self.weights)
        #pyplot.show()
        
    def process_image(self, image, positionNormalize = True,\
                       verbose = True):
        '''
        processes a single image, return the locations
        and the values of detected SIFT features.
        image: a M*N image which is a numpy 2D array. If you 
            pass a color image, it will automatically be converted
            to a grayscale image.
        positionNormalize: whether to normalize the positions
            to [0,1]. If False, the pixel-based positions of the
            top-right position of the patches is returned.
        
        Return values:
        feaArr: the feature array, each row is a feature
        positions: the positions of the features
        '''

        image = image.astype(np.double)
        if image.ndim == 3:
            # we do not deal with color images.
            image = np.mean(image,axis=2)
        # compute the grids
        H,W = image.shape
        gS = self.gS
        pS = self.pS
        remH = np.mod(H-pS, gS)
        remW = np.mod(W-pS, gS)
        offsetH = int(remH/2)
        offsetW = int(remW/2)
#         print(type(remH), remH)
        gridH,gridW = np.meshgrid(range(offsetH,H-pS+1,gS), range(offsetW,W-pS+1,gS))
        gridH = gridH.flatten()
        gridW = gridW.flatten()
#         if verbose:
#             print ('Image: w {}, h {}, gs {}, ps {}, nFea {}'.\
#                     format(W,H,gS,pS,gridH.size))
        feaArr = self.calculate_sift_grid(image,gridH,gridW)
        feaArr = self.normalize_sift(feaArr)
        if positionNormalize:
            positions = np.vstack((gridH / np.double(H), gridW / np.double(W)))
        else:
            positions = np.vstack((gridH, gridW))
        return feaArr, positions

    def calculate_sift_grid(self,image,gridH,gridW):
        '''
        This function calculates the unnormalized sift features
        It is called by process_image().
        '''
        H,W = image.shape
        Npatches = gridH.size
        feaArr = np.zeros((Npatches,Nsamples*Nangles))

        # calculate gradient
        GH,GW = gen_dgauss(self.sigma)
        IH = signal.convolve2d(image,GH,mode='same')
        IW = signal.convolve2d(image,GW,mode='same')
        Imag = np.sqrt(IH**2+IW**2)
        Itheta = np.arctan2(IH,IW)
        Iorient = np.zeros((Nangles,H,W))
        for i in range(Nangles):
            Iorient[i] = Imag * np.maximum(np.cos(Itheta - angles[i])**alpha,0)
            #pyplot.imshow(Iorient[i])
            #pyplot.show()
        for i in range(Npatches):
            currFeature = np.zeros((Nangles,Nsamples))
            for j in range(Nangles):
                currFeature[j] = np.dot(self.weights,\
                        Iorient[j,gridH[i]:gridH[i]+self.pS, gridW[i]:gridW[i]+self.pS].flatten())
            feaArr[i] = currFeature.flatten()
        return feaArr

    def normalize_sift(self,feaArr):
        '''
        This function does sift feature normalization
        following David Lowe's definition (normalize length ->
        thresholding at 0.2 -> renormalize length)
        '''
        siftlen = np.sqrt(np.sum(feaArr**2,axis=1))
        hcontrast = (siftlen >= self.nrml_thres)
        siftlen[siftlen < self.nrml_thres] = self.nrml_thres
        # normalize with contrast thresholding
        feaArr /= siftlen.reshape((siftlen.size,1))
        # suppress large gradients
        feaArr[feaArr>self.sift_thres] = self.sift_thres
        # renormalize high-contrast ones
        feaArr[hcontrast] /= np.sqrt(np.sum(feaArr[hcontrast]**2,axis=1)).\
                reshape((feaArr[hcontrast].shape[0],1))
        return feaArr

class SingleSiftExtractor(DsiftExtractor):
    '''
    The simple wrapper class that does feature extraction, treating
    the whole image as a local image patch.
    '''
    def __init__(self, patchSize,
                 nrml_thres = 1.0,\
                 sigma_edge = 0.8,\
                 sift_thres = 0.2):
        # simply call the super class __init__ with a large gridSpace
        DsiftExtractor.__init__(self, patchSize, patchSize, nrml_thres, sigma_edge, sift_thres)   
    
    def process_image(self, image):
        return DsiftExtractor.process_image(self, image, False, False)[0]
    

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


train_descriptors = []

extractor = DsiftExtractor(8,16,1)
for image_path in tqdm(X_train):
    image = misc.imread(image_path)
    feaArr, positions = extractor.process_image(image)
    train_descriptors.append(feaArr)

train_descriptors = np.concatenate(train_descriptors, axis=0)
train_descriptors.shape

test_descriptors = []


extractor = DsiftExtractor(8,16,1)
for image_path in tqdm(X_test):
    image = misc.imread(image_path)
    feaArr, positions = extractor.process_image(image)
    test_descriptors.append(feaArr)

test_descriptors = np.concatenate(test_descriptors, axis=0)
test_descriptors.shape


print('Learning the dictionary...')
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=50, batch_size=10, alpha=3, n_iter=250)
dictionary = dico.fit(train_descriptors).components_
dt = time() - t0
print('done in %.2fs.' % dt)

print('dictionary.shape : ', dictionary.shape)

descriptor_count = 5041

print('finding the X_train_sparse_code..........')

t0 = time()
dico.set_params(transform_algorithm='omp')
X_train_sparse_code = dico.transform(train_descriptors)
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

int(X_train_sparse_code.shape[0] / descriptor_count)

final_vectors_list_train = []
t0 = time()

last_keypoints_count_train = 0

for i in tqdm(range(int(X_train_sparse_code.shape[0] / descriptor_count))):
    keypoints_count = descriptor_count
    vector = np.amax(X_train_sparse_code[last_keypoints_count_train : keypoints_count + last_keypoints_count_train], axis = 0)
    final_vectors_list_train.append(vector)
    
    last_keypoints_count_train += keypoints_count
    
final_vectors_list_train = np.array(final_vectors_list_train)

dt = time() - t0
print('done in %.2fs.' % dt)

final_vectors_list_test = []
t0 = time()

last_keypoints_count_test = 0

for i in tqdm(range(int(X_test_sparse_code.shape[0] / descriptor_count))):
    keypoints_count = descriptor_count
    vector = np.amax(X_test_sparse_code[last_keypoints_count_test : keypoints_count + last_keypoints_count_test], axis = 0)
    final_vectors_list_test.append(vector)
    
    last_keypoints_count_test += keypoints_count
    
final_vectors_list_test = np.array(final_vectors_list_test)

dt = time() - t0
print('done in %.2fs.' % dt)

print('final_vectors_list_train.shape : ', final_vectors_list_train.shape)

print('final_vectors_list_test.shape : ', final_vectors_list_test.shape)


clf = LinearSVC()
clf.fit(final_vectors_list_train, y_train)
predictions = clf.predict(final_vectors_list_test)

print(classification_report(y_test, predictions))


with open('test_descriptors', 'wb') as f:
    pickle.dump(test_descriptors, f)

with open('train_descriptors', 'wb') as f:
    pickle.dump(train_descriptors, f)

with open('X_train_sparse_code', 'wb') as f:
    pickle.dump(X_train_sparse_code, f)

with open('X_test_sparse_code', 'wb') as f:
    pickle.dump(X_test_sparse_code, f)

with open('dictionary', 'wb') as f:
    pickle.dump(dictionary, f)
    
with open('final_vectors_list_train', 'wb') as f:
    pickle.dump(final_vectors_list_train, f)
    
with open('final_vectors_list_test', 'wb') as f:
    pickle.dump(final_vectors_list_test, f)
    
with open('Dataset', 'wb') as f:
    pickle.dump([X_train, X_test, y_train, y_test], f)
    
    
print("...................saved all variables.............")
