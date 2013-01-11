#!/usr/bin/python
# -*- coding: utf-8 -*-
# Roman Yurchak, Laboratoire LULI, 16.11.2012

import re
import numpy as np


class HamamatsuFile(object):
    def __init__(self, filename):
        """ A parser to read Hamamatsu streak camera's .img output files.
        This code was partly adapted from an ImageJ plugin.

        Parameters
        ----------
         - filename [str]: filepath to .img file to open

        Returns
        -------
         an HamammatsuReader object with following attributes:
           - HamammatsuReader.header  [dict] : dictionary with metada
             information.
           - HamammatsuReader.data [ndarray, ndim=2]: image data
           - HamammatsuReader.shape : returns the shape of the data
             array.

        Exemple of use
        --------------
        >> img = HamamatsuFile('/home/user/shot001.img')
        >> print img # gives just the most usefull information from img.header
        DeviceName: C7700
        CameraName: C4742-98-24NR
        DateTime: 28/03/2012 09:23:05.684
        Time Range: 20 ns
        Gain: 63
        Temperature: -50.1
        Shutter: Open
        Mode: Single
        Binning: 1,1
        Shape: (1280, 1024)
        >> print img.header['Gain']
        63
        >> plt.imshow(img.data)
        """
        self.filename = filename
        self._read_header()
        self._read_data()

    def _read_header(self):
        """Read the Hamamatsu header for the given filename
        Internal use only
        """
        f = open(self.filename, 'r')
        idx = 0
        header = ''
        # reading the header 
        while idx < 10: 
            header += f.readline()[:-2] # removes the "\n\r" at the end

            idx += 1
        # "magically" compute the data offset
        self._data_offset = ord(header[:10].decode('utf-8')[2]) + 1856
        # this removes the values between square [] e.g: [Grabber]
        header = re.sub(r'\[[^\]]+\]', '', header)
        base_regexp = r'(?P<key>[A-Z][a-zA-Z_0-9 ]+)='
        fh = re.findall(base_regexp + r'(?P<val>[^,"]+),', header) # e.g: NrTrigger=1
        fh += re.findall(base_regexp + r'"(?P<val>[^"]*)"', header) # e.g: pntOrigCh="0,0"
        self.header  = dict(fh)
        self.header['DateTime'] = self.header['Date'] +' ' + self.header['Time']
        del self.header['Date'], self.header['Time']

        self.shape = np.array(self.header['GRBScan'].split(',')[-2:]).astype(np.int)
        f.close()
        return self.header

    def _read_data(self):
        """Reading the binary data
        Internal use only.
        """
        with open(self.filename, 'rb') as f:
            f.seek(self._data_offset)
            self.data = np.fromfile(f, dtype=np.int16,
                    count=np.prod(self.shape)).reshape(self.shape[::-1])

    def __repr__(self):
        """Default representation of the class.
        This method is used when calling pring HamammatsuReader
        """
        stdout = ''
        for key in ['DeviceName', 'CameraName', 'DateTime',
                'Time Range', 'Gain','Temperature',
                'Shutter', 'Mode', 'Binning']:
            if key in self.header:
                stdout += '%s: %s\n' % (key, self.header[key])
        stdout += 'Shape: %s' % str(tuple(self.shape))
        return stdout

def test_hamamatsu():
    """Some basic unitest"""
    ref = np.load('test_image.npy').astype(np.int16)
    img = HamamatsuFile('test_image.img').data
    assert (np.flipud(ref) == img).all()



if __name__ == '__main__':
    # just call this file as
    # python hamamatsu.py filename.img
    import argparse
    import sys
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
            description="Open a Hamamatsu streak camera's .img file")
    parser.add_argument('filepath', type=str,
                               help='path to the .img file')

    args = parser.parse_args()
    sp = HamamatsuFile(args.filepath)
    cs = plt.imshow(sp.data, vmax=np.percentile(sp.data, 99.9))
    plt.colorbar(cs)
    plt.show()

