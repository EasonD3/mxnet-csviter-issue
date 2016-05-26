import os
import sys
sys.path.insert(0, '~/mxnet/python')

import mxnet as mx
import numpy as np


def print_iter(iter):
    iter.reset()
    bi = 1
    for b in iter:
        print b.data[0].asnumpy()
        print 'batch {}: {} {}'.format(bi, len(b.data[0].asnumpy()), b.pad)
        bi += 1


if __name__ == '__main__':
    N = 10
    batch_size = 3

    data_iter = mx.io.CSVIter(data_csv='./data.csv', data_shape=(1,), batch_size=batch_size)
    print_iter(data_iter)
