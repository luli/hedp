#!/usr/bin/python
#"""andor.py version 1.0
#November 2008
#Michael V. DePalatis <surname at gmail dot com>
#
#Contains functions to handle Andor SIF image files including
#conversion to other formats and reading basic information from the SIF
#file. Little is done in the way of error checking, but this is
#probably not a real issue as long as you actually pass it a valid SIF
#file.
#"""

import sys
import numpy as np

class SifFile:
    """SifFile is the Python representation of an Andor SIF image
    file. Image data is stored in a numpy array indexed as [row,
    column] instead of [x, y]."""

    def __init__(self, path=""):
        self.data = 0
        if path != "":
            self.open(path)

    def __add__(self, other):
        new_sif = self.__class__()
        new_sif.__dict__ = self.__dict__
        if isinstance(other, (SifFile, np.ndarray)):
            new_sif.data = new_sif.data + other.data
        elif isinstance(other, (int, float)):
            new_sif.data = new_sif.data + other
        else:
            raise TypeError("Addition of SIF data requires another SifFile instance, a numpy array, or a scalar.")
        return new_sif

    def __sub__(self, other):
        new_sif = self.__class__()
        new_sif.__dict__ = self.__dict__
        if isinstance(other, (SifFile, np.ndarray)):
            new_sif.data = new_sif.data - other.data
        elif isinstance(other, (int, float)):
            new_sif.data = new_sif.data - other
        else:
            raise TypeError("Subtraction of SIF data requires another SifFile instance, a numpy array, or a scalar.")
        return new_sif

    def __mul__(self, other):
        new_sif = self.__class__()
        new_sif.__dict__ = self.__dict__
        if isinstance(other, (SifFile, np.ndarray)):
            new_sif.data = new_sif.data * other.data
        elif isinstance(other, (int, float)):
            new_sif.data = new_sif.data * other
        else:
            raise TypeError("Multiplcation of SIF data requires another SifFile instance, a numpy array, or a scalar.")
        return new_sif

    def __rmul__(self, other):
        return self.__mul__(other)

    def open(self, path):
        """Opens the SIF file at path and stores the data in
        self.data."""
        sif = open(path, "rb")

        # Verify we have a SIF file
        if sif.readline().strip() != "Andor Technology Multi-Channel File":
            sif.close()
            raise Exception("File %s is not an Andor SIF file." % path)

        # Ignore lines until we get to camera model
        for i in range(2):
            sif.readline()

        # Get camera model
        self.cammodel = sif.readline().strip()

        # Get CCD dimension in pixels
        shape = sif.readline().split()
        self.ccd_size = (int(shape[0]), int(shape[1]))

        # Read superpixeling data
        line = sif.readline().split()
        #self.shape = (self.ccd_size[1]/int(line[5]), self.ccd_size[0]/int(line[6]))

        # Read data
        raw_data = sif.read()
        raw_data = raw_data[len(raw_data) - 4*np.prod(self.ccd_size):]
        self.data = np.fromstring(raw_data, dtype=np.float32)
        #self.data = self.data[:len(self.data)-2]
        #if line[3] < line[2]:
        #    self.shape = (len(self.data)/int(line[3]), int(line[3]))
        #else:
        #    # I'm not sure if this is correct...
        #    # Needs more testing.
        #    self.shape = (int(line[2]), len(self.data)/int(line[2]))
        self.data = np.reshape(self.data, self.ccd_size)

## Testing
if __name__ == "__main__":
    import pylab as plt
    import scipy.ndimage

    sif = SifFile(sys.argv[1])
    cs = plt.imshow(scipy.ndimage.filters.median_filter(sif.data, 3))#, cmap=plt.cm.Paired)
    plt.colorbar(cs)
    plt.show()    
