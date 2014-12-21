# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 21:55:23 2014

1. Loads an existing database of image codebook histograms
2. Read images form "imagenesPlatos/*.png2" 
3. Classifies each images according to its euclidean distance to the elements in the database

@author: pbustos
"""


import cv2
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import glob
import scipy.stats
from matplotlib import pyplot as plt
import cPickle as pickle

files = glob.glob('imagenesPlatos/*.png')
des_array = np.zeros(32)
des = []

# number of clusters for kmeans
clusters = 100

def euc_dist(a,b):
    distance = (a-b)**2
    distance = distance.sum(axis=0)
    return np.sqrt(distance)                  

orb = cv2.ORB()
bf = cv2.BFMatcher()
database = []

with open('bin/database.p', 'rb') as fp:
    database = pickle.load(fp)
codebook = np.load("bin/codebook"+str(clusters)+".npy")

for i,f in enumerate(files[:100]): 

    print f
    img = cv2.imread(f)
    gray= cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
 
    kp, d = orb.detectAndCompute(gray,None)
    d = whiten(d)
    #des.append(d)
    #des_array = np.vstack((des_array,des[i]))
    
    # draw only keypoints location,not size and orientation
    #img2 = cv2.drawKeypoints(gray,kp,color=(0,255,0), flags=0)
    #plt.imshow(img2),plt.show() 

    query_code, distortions = vq(d, codebook)
    query_hist = np.histogram(query_code, range(clusters))

    # Search for nearest match, by highest frequency
    smallest_distance = 10000000
    for i,data in enumerate(database):
        distance = euc_dist(query_hist[0], data[1][0])
        if distance < smallest_distance:
                best_index = i
                smallest_distance = distance
    print best_index, distance
    print f, database[best_index][2]
    print "----------------"










