#!/usr/bin/python
# -*- coding: utf-8 -*-
# hedp module
# Roman Yurchak, Laboratoire LULI, 11.2012

import re
import numpy as np
#import codecs



class HamamatsuFile(object):
    def __init__(self, filename, offset='auto'):
        """ A parser to read Hamamatsu streak camera's .img output files.
        This code was partly adapted from an ImageJ plugin.

        Parameters
        ----------
         - filename [str]: filepath to .img file to open
         - offset [str or int]: the method to use when computing the
               offset. Can be :
                   * 'auto' : try to read the offset in the header.
                        Should work for most cases, but may occasionnaly fail.
                   * 'from_end': get offset as data_size - image_size
                   * 'from_end_4k': same as 'from_end' but additionnaly
                        substract 4092 (seems to be streak dependant..)
                   * int : manual value for the offset
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
        if (type(offset) is str and\
                offset not in ['auto', 'from_end', 'from_end_4k']) and\
                (type(offset) is not int):
            raise ValueError("Wrong input value for 'offset' input parameter!")
        self._offset_input = offset
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
        while idx < 13: 
            header += f.readline()[:-2] # removes the "\n\r" at the end
            idx += 1
        # "magically" compute the data offset
        self._offset_auto = ord(header[2]) + 1856

        header =  header[:self._offset_auto+300] # add an extra random header for offset
        header = re.sub(r'(?P<section>\[[^\]]+\])', '\n\g<section>', header)
        header = header.splitlines()[1:]
        self.header = dict([self._header_sect2dict(line) for line in header])
        self.shape = np.array(self.header['Acquisition']['areGRBScan'].split(',')[-2:]).astype(np.int)
        f.close()

        self._offset_whence = 0
        if type(self._offset_input) is str:
            offset_list = {'auto': self._offset_auto,
                           'from_end': -np.prod(self.shape)*2,
                           'from_end_4k': - np.prod(self.shape)*2 - 4092}

            self._offset_data = offset_list[self._offset_input]
            if self._offset_input.startswith('from_end'):
                # set the flag to seek from the end of the file.
                self._offset_whence = 2
        elif type(self._offset_input) is int:
            self._offset_data = self._offset_input


        return self.header

    @staticmethod
    def _header_sect2dict(line):
        try:
            sect_name = re.match(r'\[(?P<section>[^\]]+)\]', line).group('section')
            metadata = re.split(r'(?<=\d|"|\]|\w),(?=[a-zA-Z])', line)[1:]
            for idx, val in enumerate(metadata):
                metadata[idx] = val.split('=')
                mval =  re.sub('"', '', metadata[idx][1])
                if mval.isdigit():
                    mval = int(mval)
                metadata[idx][1] = mval
            return (sect_name, dict(metadata))
        except:
            return ('Error', 'here')

    def _read_data(self):
        """Reading the binary data
        Internal use only.
        """
        with open(self.filename, 'rb') as f:
            f.seek(self._offset_data, self._offset_whence)
            self.data = np.fromfile(f, dtype=np.int16,
                    count=np.prod(self.shape)).reshape(self.shape[::-1])

    def __repr__(self):
        """Default representation of the class.
        This method is used when calling pring HamammatsuReader
        """
        out = ""
        for section_name, section_data in sorted(self.header.iteritems()):
            if section_name== 'Error':
                continue
            out += '\n'.join(['='*80, " "*20 + section_name, '='*80]) + '\n'
            for key, val in sorted(section_data.iteritems()):
                out += '   - {0} : {1}\n'.format(key, val)
            out += '\n'
        return out


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
    #  some logic to determine offset mode depending on the folder
    offset = 'from_end'
    if sum([key in args.filepath for key in ['Rear_SOP_1D']]) and\
        not sum([key in args.filepath for key in ['ref.img', 'alignemts']]):
        offset =  'from_end_4k'
    elif sum([key in args.filepath for key in ['Transverse_SOP_1D']]):
        offset =  'from_end'
    print offset


    sp = HamamatsuFile(args.filepath, offset)
    cs = plt.imshow(sp.data, vmax=np.percentile(sp.data, 99.9))

    plt.colorbar(cs)
    plt.show()

