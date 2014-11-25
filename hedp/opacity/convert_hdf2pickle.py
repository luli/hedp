#!/usr/bin/python
# -*- coding: utf-8 -*-
import tables
import pickle

FILEBASE = '/home/rth/.hedp_matdb/henke_op'
f = tables.open_file(FILEBASE+'.h5', 'r')

out_dir = {}
specs =  [key for key in dir(f.root) if not key.startswith('_')]
for el in specs:
    nu0 = getattr(f.root, el).nu[:]
    op0 = getattr(f.root, el).op[:]
    out_dir[el] = nu0, op0
with open(FILEBASE+'.pickle', 'wb') as handle:
  pickle.dump(out_dir, handle)







