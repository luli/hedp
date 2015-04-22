#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from hedp.viz.formatters import MetricFormatter

def test_metric_formatter():
    d = 1e-8
    f = MetricFormatter()
    assert f(d) == u"10 n"

