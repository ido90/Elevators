
import sys
import time
import numpy as np

class PrintTime():
    def __init__(self):
        self.ti = time.time()
    def update_reference(self):
        self.ti = time.time()
    def print_time(self):
        tf = time.time()
        dt = tf - self.ti
        mins = int(dt//60)
        secs = int(dt%60)
        print('Elapsed runtime:\t{:d}:{:02d} minutes'.format(mins,secs))

nones = lambda n: [None for _ in range(n)]

def assert_integers(*args):
    for x in args:
        assert (x==int(x)), x

def assert_zero(*args, eps=sys.float_info.epsilon):
    for x in args:
        assert (abs(x)<eps), x

def dist(x, quantiles=(0,10,50,90,100), do_round=False):
    s = [len(x), np.mean(x)] + list(np.percentile(x,quantiles))
    return [int(z+np.sign(z)*0.5) for z in s] if do_round else s
