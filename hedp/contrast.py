#!/usr/bin/python
# -*- coding: utf-8 -*-
# hedp module
# Roman Yurchak, Laboratoire LULI, 11.2012

import numpy as np

def histeq(im,nbr_bins=256):
   """
    Global contrast equalisation
   """

   #get image histogram
   imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = 255 * cdf / cdf[-1] #normalize

   #use linear interpolation of cdf to find new pixel values
   im2 = np.interp(im.flatten(),bins[:-1],cdf)

   return im2.reshape(im.shape), cdf

