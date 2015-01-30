#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
# adapted from https://stackoverflow.com/questions/15733772/convert-float-number-to-string-with-engineering-notation-with-si-prefixe-in-py

class MetricFormatter:
    
    def __init__(self, fmt=u'{scaled:.0f} {prefix}', latex=False):
        self.fmt = fmt
        self.latex = latex

    def __call__(self, d):
        incPrefixes = ['k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
        decPrefixes = ['m', 'µ', 'n', 'p', 'f', 'a', 'z', 'y']
        if self.latex:
            decPrefixes[1] = r'\mu'
        if d  == 0.0:
            degree = 0
        else: 
            degree = int(math.floor(math.log10(math.fabs(d)) / 3))

        prefix = ''

        if degree!=0:
            ds = degree/math.fabs(degree)
            if ds == 1:
                if degree - 1 < len(incPrefixes):
                    prefix = incPrefixes[degree - 1]
                else:
                    prefix = incPrefixes[-1]
                    degree = len(incPrefixes)

            elif ds == -1:
                if -degree - 1 < len(decPrefixes):
                    prefix = decPrefixes[-degree - 1]
                else:
                    prefix = decPrefixes[-1]
                    degree = -len(decPrefixes)

            scaled = float(d * math.pow(1000, -degree))

            s = self.fmt.format(scaled=scaled, prefix=prefix)

        else:
            s = self.fmt.format(scaled=d, prefix='')

        return(s)


if __name__ == '__main__':
    d = 1e-8
    print MetricFormatter()(d)
