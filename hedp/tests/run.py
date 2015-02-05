#!/usr/bin/python
# -*- coding: utf-8 -*-
if __name__ == '__main__':
    import nose
    import sys
    result = nose.run()
    status = int(not result)
    print('Exit status: {}'.format(status))
    sys.exit(int(not result))




