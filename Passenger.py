
import numpy as np
from MyTools import *

class Arrival:
    def __init__(self, t, n, d, xi, xf):
        assert(t>=0 and n>=1 and d>=0 and xi>=0 and xf>=0 and xi!=xf)
        assert_integers(n,xi,xf)
        self.t = t # arrival time
        self.n = n # number of passengers
        self.d = d # passengers delay in entrance
        self.xi = xi # initial floor
        self.xf = xf # destination floor
    def print(self, i=None):
        if i is None:
            print(f"{self.t:.1f}:\t{self.n:d}\t({self.d:.1f})\t{self.xi:d} -> {self.xf:d}")
        else:
            print(f"({i:03d})\t{self.t:.1f}:\t{self.n:d}\t({self.d:.1f})\t{self.xi:d} -> {self.xf:d}")


class Passenger:
    def __init__(self, a):
        self.t0 = a.t
        self.d = a.d
        self.xi = a.xi
        self.xf = a.xf
        # initialization
        self.assigned_el = -1
        self.t1 = np.inf # pick-up time
        self.t2 = np.inf # destination time
        self.indirect_motion = 0 # number of intervals of motion against destination
