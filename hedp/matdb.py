#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

import os, os.path

import json

if 'MATDB_PATH' in os.environ:
    MATDB_PATH = os.environ['MATDB_PATH']
else:
    raise RuntimeWarning('You should define a MATDB_PATH in your .bashrc !')

def matdb(element):
    path = os.path.join(MATDB_PATH, 'db', element+ '.json')
    if not os.path.exists(path):
        raise ValueError(
            'The specified element {0} does not exist in the database {1}'.format(
                                        element, os.path.join(MATDB_PATH, 'db')))
    with open(path, 'r') as f:
        d = json.load(f)
        for key in d:
            if type(d[key]) is dict:
                d[key] = Storage(d[key])
        return Storage(d)

class Storage(dict):
    """
    A Storage object is like a dictionary except `obj.foo` can be used
    in addition to `obj['foo']`, and setting obj.foo = None deletes item foo.
    Copied from web2py framework.

        >>> o = Storage(a=1)
        >>> print o.a
        1

        >>> o['a']
        1

        >>> o.a = 2
        >>> print o['a']
        2

        >>> del o.a
        >>> print o.a
        None
    """
    __slots__ = ()
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __getitem__ = dict.get
    __getattr__ = dict.get
    __repr__ = lambda self: '<Storage %s>' % dict.__repr__(self)
    # http://stackoverflow.com/questions/5247250/why-does-pickle-getstate-accept-as-a-return-value-the-very-instance-it-requi
    __getstate__ = lambda self: None
    __copy__ = lambda self: Storage(self)
