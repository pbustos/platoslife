# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 22:16:06 2014

1. Reads all image from "imagenes/Platos"
2. Computes ORB descriptors
3. Saves them using numpy persistence
4. Computes a codebook using numpy kmeans and clusters = 100
5. Saves de codebook using np persistence



@author: pbustos
"""
import cv2
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import glob
import scipy.stats
from matplotlib import pyplot as plt

files = glob.glob('imagenesPlatos/*.png')
#files = glob.glob('imagedb/*.jpg')
des_array = np.zeros(32)
des = []

# number of clusters for kmeans

clusters = 100              
orb = cv2.ORB()
 
#for i,f in enumerate(files[13701:20000]): 
for i,f in enumerate(files[:]): 
    print f, "\t", i
    img = cv2.imread(f)
    gray= cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    kp, d = orb.detectAndCompute(gray,None)
    des.append(d) 
    des_array = np.vstack((des_array,des[i]))
    
    # draw only keypoints location,not size and orientation
#    img2 = cv2.drawKeypoints(gray,kp,color=(0,255,0), flags=0)
#    plt.imshow(img2),plt.show() 

des_array = des_array[1:]
whitened = whiten(des_array)
np.save("bin/orbdescriptors",whitened)
print "ORB descriptors saved"

print "now the codebook..."
codebook, distortions = kmeans(whitened,clusters)
np.save("bin/codeboo"+str(clusters), codebook)
print "saved"