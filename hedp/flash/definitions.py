#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

import collections


class MergedDict(collections.MutableMapping):
    def __init__(self, primary, secondary):
        """
        A read only structure that can query values from two dictionaries.

        Parameters:
          - primary: dict
          - secondary: dict
        """
        self.primary = {self.__keytransform__(key): val for key, val in primary.items()}
        self.secondary = {self.__keytransform__(key): val for key, val in secondary.items()}

    def __getitem__(self, key):
        """ Get an element """
        key = self.__keytransform__(key)

        if key in self.primary:
            return self.primary[key]
        elif key in self.secondary:
            return self.secondary[key]
        else:
            raise KeyError


    def __setitem__(self, key):
        raise NotImplementedError('This is a read-only pobject')


    def __delitem__(self, key):
        raise NotImplementedError('This is a read-only pobject')


    @property
    def _merged(self):
        tmp = self.primary.copy()
        tmp.update(self.secondary)
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




