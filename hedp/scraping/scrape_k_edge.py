#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import re
if sys.version_info.major < 3:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen



def scrape_k_edge():
    """
    Scrapes the atomic and nuclear physics database at
        http://www.kayelaby.npl.co.uk/atomic_and_nuclear_physics/4_2/4_2_1.html
    and returns a pandas.DataFrame with the following columns,
        - index (atomic number)
        - element (symbol)
        - K_edge
        - K_alpha_1
    """
    from bs4 import BeautifulSoup
    import pandas as pd

    wiki = "http://www.kayelaby.npl.co.uk/atomic_and_nuclear_physics/4_2/4_2_1.html"
    page = urlopen(wiki)
    soup = BeautifulSoup(page)


    def filter_results(it):
        out = []
        for el in it:
            el = el.strip()
            if el:
                out.append(el.replace(u'\xa0', u''))
        if not out:
            out.append(u'')
        return out


    out = []
    table = soup.find("table", { "class" : "table" })
    for idx, row in enumerate(table.findAll("tr")):
        cells = row.findAll("td")
        if len(cells) < 20 or idx < 4: 
            continue

        tmp = {}
        res = cells[1].findAll(text=True)
        res = filter_results(res)
        try:
            tmp['K_edge'] = float(res[0])
        except:
            continue # this row does not contain data


        res = cells[0].findAll(text=True)
        res = filter_results(res)[0]
        
        match = re.match(r'(\d+)\s*(\w+)', res)
        Z, element = match.groups()

        tmp['Z'], tmp['element'] = int(Z), element


        offset_dict = {'K_alpha_1': 5}
        if cells[3].has_attr('colspan'):
            offset_dict['K_alpha_1'] -= 1

        for key in offset_dict:
            res = cells[offset_dict[key]].findAll(text=True)
            res = filter_results(res)[0]

            tmp[key] = float(res)




        out.append(tmp)

    df = pd.DataFrame(out)
    df.index = df.Z
    del df['Z']
    return df
