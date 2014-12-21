# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 22:43:10 2014

1. Reads a codebook from "bin/codebookXXX.npy"
2. Reads images from "imagedb/*.jpg"
3. Computes ORB descriptors for the images
4. Computes histogram of each image from codebook
5. Saves histograms in a file using pickle to be used as a matching database

@author: pbustos
"""
import cv2
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import glob
import scipy.stats
from matplotlib import pyplot as plt
import cPickle as pickle

files = glob.glob('imagedb/*.jpg')
print "files: ", len(files)

# number of clusters for kmeans
clusters = 100
orb = cv2.ORB()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
database = []
 
codebook = np.load("bin/codebook100.npy") 

for i,f in enumerate(files[:]):
 
    img = cv2.imread(f)
    gray= cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)  
    kp, d = orb.detectAndCompute(gray,None)
#    
#    for desc in enumerate(d):
#        matches = bf.match(desc,codebook)
#        # Sort them in the order of their distance.
#        matches = sorted(matches, key = lambda x:x.distance)        
#        
    d = whiten(d)
    code, distortions = vq(d, codebook)
    
    #draw only keypoints location,not size and orientation
    #img2 = cv2.drawKeypoints(gray,kp,color=(0,255,0), flags=0)
    #plt.imshow(img2),plt.show() 
 
    hist = np.histogram(code, range(clusters))
    database.append((hist, f))
    print i
    
with open('bin/databaseAll100.p', 'wb') as fp:
    pickle.dump(database, fp)
print "database saved to database.p"


# create BFMatcher object


