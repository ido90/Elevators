
from MyTools import *
import numpy as np

class Elevator:

    def __init__(self, index, n_floors, capacity, speed, open_time):
        # Configuration
        self.idx = index
        self.N = n_floors
        self.capacity = capacity
        self.speed = speed
        self.open_time = open_time
        # Initialization
        self.x, self.is_open, self.motion = nones(3)
        self.missions, self.next_t = nones(2) # TODO put next_t out of this class
        self.total_opens, self.total_distance = nones(2)
        self.initialize()

    def __str__(self):
        return f"i={self.idx:d}:\tx={self.x:.2f}, open={self.is_open:b}, motion={self.motion:.0f}, next_t={self.next_t:.2f}\n{self.missions}"

    def initialize(self):
        self.x = 0
        self.is_open = False
        self.motion = 0 # +1 for up, -1 for down, 0 for standing
        self.missions = []
        self.next_t = np.inf
        self.total_opens = 0
        self.total_distance = 0

    def open(self, t, delay):
        self.next_t = t + 2 * self.open_time + delay
        self.motion = 0
        self.is_open = True
        self.total_opens += 1

    def move(self, t):
        self.next_t = t + abs(self.missions[0] - self.x) / self.speed
        self.motion = np.sign(self.missions[0] - self.x)
        self.is_open = False

    def sleep(self):
        self.next_t = np.inf
        self.motion = 0
        self.is_open = False

    def locate(self, i):
        '''Get elevator expected location after i missions.'''
        x = [l for l in self.missions[:i] if l is not None]
        return x[-1] if x else self.x
