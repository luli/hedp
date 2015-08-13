#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

import collections
import os
import fnmatch


class MergedDict(collections.MutableMapping):
    def __init__(self, *objects):
        """
        A read only structure that can query values from multiple dictionaries,
        the ealier ones in the list having a higher priority

        Parameters:
        -----------
          a list of dictionaries (e.g. MergedDict(dict1, dict2, dict3) )
        """
        self.obj = objects

    @property
    def collection(self):
        """ Generate the collection every time"""
        # this is very unefficient, but allows to account for changes in
        # the self.obj that can't be controlled
        return [{self.__keytransform__(key): val for key, val in obj.items()} for obj in self.obj]

    def __getitem__(self, key):
        """ Get an element """
        key = self.__keytransform__(key)

        for obj in self.collection:
            if key in obj:
                return obj[key]
        else:
            raise KeyError


    def __setitem__(self, key):
        raise NotImplementedError('This is a read-only pobject')


    def __delitem__(self, key):
        raise NotImplementedError('This is a read-only pobject')


    @property
    def _merged(self):
        tmp = {}
        for obj in self.collection[::-1]:
            tmp.update(**obj)
        return tmp

    def __iter__(self):
        return iter(self._merged)


    def __len__(self):
        return len(self._merged)

    def __keytransform__(self, key):
        return key.lower()


def parse_flash_header(path):
    """
    Parse Flash.h looking for variable definitions of the form,
      #define variable value

    Returns a dictionary with the corresponding values.
    """
    import re
    out = {}
    regexp = re.compile(r'#define (\w+) (\d+)')
    with open(path, 'r') as f:
        for line in f:
            m = re.match(regexp, line)
            if m:
                key, val = m.groups()
                out[key] = int(val)
    return out


def find_files(locations_list, pattern): 
    """
    Given a list of folder find the files marching the pattern (unix regexp).
    """
    matches = []
    for base_dir in locations_list:
        for root, dirnames, filenames in os.walk(os.path.abspath(base_dir)):
          for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches




