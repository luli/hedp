#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals


import re
import numpy as np
import warnings
import os.path

warnings.simplefilter('error', UserWarning)

class HamamatsuFile(object):
    def __init__(self, filename, offset='from_end', dtype='int32',
            ignore_header=False, shape=None):
        """ A parser to read Hamamatsu streak camera's .img output files.
        This code was partly adapted from an ImageJ plugin.

        Parameters
        ----------
         - filename [str]: filepath to .img file to open
         - offset [str or int]: the method to use when computing the
               offset. Can be :
                   * 'auto' : try to read the offset in the header.
                     That should be the normal way of getting to offset,
                     but it doesn't really work.
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
        if (offset not in ['auto', 'from_end', 'from_end_4k']) and (type(offset) is not int):
            raise ValueError("Wrong input value for 'offset' input parameter! Acceptable values are 'auto', 'from_end', 'from_end_4k'.")
        self._offset_input = offset
        self._offset_whence = 0
        self._dtype=dtype
        if dtype=='int32':
            self._nbytes = 4
        elif dtype=='int16':
            self._nbytes = 2
        else:
            raise ValueError("Wrong input value for dtype!")

        self._set_opener()

        if not ignore_header:
            self._read_header()
        else:
            self._offset_data = offset
            self.shape = shape
        #print(self._offset_data)
        self.heristic_analysis()
        self._read_data()

    def heristic_analysis(self):
        """ Try to determine whether the provided offset and dtype are
        consistent given the total file size """
        if self._compression:
            # don't bother with file size arguments if the file is compressed
            return

        img_len = np.prod(self.shape)*self._nbytes
        file_len = os.path.getsize(self.filename)
        flag_1 = file_len < img_len
        flag_2 = file_len > 1.5*img_len
        if  flag_1 or flag_2:
            print("""Warning: hedp.io.HamamatsuFile 
        File size {}, image size {}={}""".format(
                    img_len, file_len, self.shape))
            if flag_1:
                print(" "*9,"File length smaller than the expected size of the image!")
            if flag_2:
                print(" "*9,"File length larger by more then 50% the expected size of the image!")
            print(" "*9, 'The dtype (or the determined shape) are probably wrong')
        return


    def _set_opener(self):
        """ Internal function to open, optionally compressed, hamamatsu file """
        if self.filename.lower().endswith('.img'):
            opener = open
            self._compression = False
        elif self.filename.lower().endswith('.img.bz2'): # bz2 compressed sif files
            import bz2
            opener = bz2.BZ2File
            self._compression = True
        else:
            raise ValueError('Wrong extension.')

        self._open = opener



    def _read_header(self):
        """Read the Hamamatsu header for the given filename
        Internal use only
        """
        f = self._open(self.filename, 'r')
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

        offset_list = {'auto': self._offset_auto,
                       'from_end': -np.prod(self.shape)*self._nbytes,
                       'from_end_4k': - np.prod(self.shape)*self._nbytes - 4092}

        if self._offset_input in offset_list:

            self._offset_data = offset_list[self._offset_input]
            if self._offset_input.startswith('from_end'):
                # set the flag to seek from the end of the file.
                self._offset_whence = 2
        elif type(self._offset_input) is int:
            self._offset_data = self._offset_input
        else:
            raise ValueError

        

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
            return ('Error', {'error': 'error'})

    def _read_data(self):
        """Reading the binary data
        Internal use only.
        """
        with self._open(self.filename, 'rb') as f:
            try:
                f.seek(self._offset_data, self._offset_whence)
            except IOError:
                print('Error: hedp.io.HamamatsuFile seeking outside of file limits.')
                print('       Failed to parse file.')
                print("       Either the 'offset'  or 'dtype' input arguments must be wrong!")
                raise
            except:
                raise

            data_len = np.prod(self.shape)*np.dtype(self._dtype).itemsize
            data_str = f.read(data_len)
            if data_len != len(data_str):
                print(data_len, len(data_str))
                raise ValueError('File ended before all data was read. Probably wrong offset or dtype!')


            self.data = np.fromstring(data_str, dtype=self._dtype).reshape(self.shape[::-1])

            #self.data = np.fromfile(f, dtype=self._dtype,
            #        count=np.prod(self.shape)).reshape(self.shape[::-1])

    @property
    def time_range(self):
        """ Get time range (e.g. 50 ns) in seconds """
        tr_str = self['Time Range']
        val, unit = tr_str.split(' ')
        val = float(val)
        unit_multiplier = {'ps': 1e-12, 'ns': 1e-9, 'us': '1e-6'}[unit]
        return val*unit_multiplier


    def __getitem__(self, key):
        """
        Query an element from the header by key

            HamamatsuFile[key]
        """
        for section in self.header.values():
            for ckey, val in section.items():
                if ckey == key:
                    return val
        else:
            raise ValueError

    def __repr__(self):
        """Default representation of the class.
        This method is used when calling pring HamammatsuReader
        """
        out = ""
        for section_name, section_data in sorted(self.header.items()):
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
    offset = 'from_end_4'
    print(offset)


    sp = HamamatsuFile(args.filepath, offset, dtype="int16")
    print(sp._offset_data)
    print(sp.data.shape, sp._nbytes)

    d = sp.data
    cs = plt.imshow(sp.data, vmax=np.percentile(sp.data, 99.9))

    plt.colorbar(cs)
    plt.show()

