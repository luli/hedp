#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import hedp


def test_matdb_database():
    """ Check that we can parse the database """
    hedp.load_material_database()

