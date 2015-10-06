#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

import os, os.path

import json

#if 'MATDB_PATH' in os.environ:
#    MATDB_PATH = os.environ['MATDB_PATH']
#else:
#    MAT

_base_dir = os.path.dirname(os.path.realpath(__file__))
MATDB_PATH = os.path.join(_base_dir, 'data')


def matdb(element):
    path = os.path.join(MATDB_PATH, 'db', element+ '.json')
    if not os.path.exists(path):
        raise ValueError(
            'The specified element {0} does not exist in the database {1}'.format(
                                        element, os.path.join(MATDB_PATH, 'db')))
    with open(path, 'r') as f:
        try:
            d = json.load(f)
        except ValueError:
            print('Error: Syntax error in {}.json file!'.format(element))
            raise
        except:
            raise
        for key in d:
            if type(d[key]) is dict:
                d[key] = Storage(d[key])
        return Storage(d)


def load_material_database():
    """ Load the material database """
    material_keys = [os.path.splitext(el)[0] for el in os.listdir(os.path.join(MATDB_PATH, 'db')) if el.endswith('.json')]

    db = {}
    incorrect_files = []
    for key in material_keys:
        try:
            db[key] = matdb(key)
        except:
            incorrect_files.append(key)
    if incorrect_files:
        raise ValueError("Some materials in the database failed to parse: {}".format(incorrect_files))
    return db


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
    __getattr__ = dict.__getitem__
    __repr__ = lambda self: '<Storage %s>' % dict.__repr__(self)
    # http://stackoverflow.com/questions/5247250/why-does-pickle-getstate-accept-as-a-return-value-the-very-instance-it-requi
    __getstate__ = lambda self: None
    __copy__ = lambda self: Storage(self)
