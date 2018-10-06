
from MyTools import *
from time import sleep
import matplotlib.pyplot as plt

class SimPlotter:

    '''
    SimPlotter
    Real-time plotter for ElevatorSimulator.
    Written by Ido Greenberg, 2018
    '''

    def __init__(self, n_floors, n_elevators, sim_pace):
        '''Define all variables, open figure and plot the initial settings.'''
        # Structure
        self.H = n_floors
        self.W = n_elevators
        # Elevators
        self.y = self.W * [0]
        self.o = [False for _ in range(self.W)] # open/closed
        # Graphic objects
        self.pipes, self.el, self.wps, self.mps = nones(4)
        self.sim_pace = sim_pace
        self.t = 0

    def initialize(self):
        plt.show()
        self.t = 0
        # axes
        axes = plt.gca()
        axes.set_xlim(-2,self.W)
        axes.set_ylim(-0.5,self.H+0.5)
        plt.xticks(list(range(self.W)))
        plt.yticks(list(range(self.H+1)))
        plt.title("Elevators Simulator")
        self.update_time()
        # elevators
        self.pipes = [plt.plot(2*[i], [0,self.H], 'k-', linewidth=32)[0]
                      for i in range(self.W)]
        self.el = [plt.plot(i,self.y[i],'rs',markersize=29,
                            markerfacecolor='r',markeredgewidth=3)[0]
                   for i in range(self.W)]
        # waiting passengers
        self.wps = [plt.plot(-1, i, 'b.', markersize=0)[0]
                    for i in range(self.H+1)]
        # moving passengers
        self.mps = [plt.plot(i, self.y[i], 'b.', markersize=0)[0]
                    for i in range(self.W)]

    def update_plot(self, dt, el, waiting_passengers, moving_passengers):
        '''Update plot according to current state.'''
        self.y = [e.x for e in el]
        self.o = [e.is_open for e in el]
        for pl,y,o in zip(self.el,self.y,self.o):
            pl.set_ydata(y)
            pl.set_markerfacecolor('w' if o else 'r')
        for i in range(self.W):
            self.mps[i].set_ydata(self.y[i])
        for i,eps in enumerate(self.mps):
            ps_in_el = moving_passengers[i]
            eps.set_markersize(10*ps_in_el)
        for floor in range(self.H+1):
            ps_in_fl = waiting_passengers.count(floor)
            self.wps[floor].set_markersize(10*ps_in_fl)
        self.t += dt
        self.update_time()
        plt.draw()
        plt.pause(1e-17)
        sleep(dt / self.sim_pace)

    def update_time(self):
        plt.xlabel(f"time = {int(self.t):03d} [s]", fontsize=18)
