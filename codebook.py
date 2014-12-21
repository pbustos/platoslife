# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 22:16:06 2014

1. Loads file of descriptors form "bin/XXX" 
2. Computes codebook using kmean form numpy
3. Saves the codebook using np persistence

@author: pbustos
"""
import cv2
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import glob
import scipy.stats
from matplotlib import pyplot as plt

# number of clusters for kmeans
clusters = 100              

print "reading descriptors file"
whitened = np.load("bin/orbdescriptors.npy")
print len(whitened) , " ORB descriptors read"

print "now the codebook..."
codebook, distortions = kmeans(whitened,clusters)
np.save("bin/codebook" +str(clusters), codebook)
print "saved"