
import numpy as np
from math import factorial as fac
from math import log10 as log
import time

def count_states(H, N, capacity):

    # Calculate order of magnitude of number of states for system with
    # H floors and N elevators, each can load at most capacity passengers.
    #
    # Full calculation: all states.
    # Min calculation: only states with <=1 unassigned passengers.
    #
    # Min results (per H, N, capacity):
    # 40,8,8: 13+4+64+168+3~252
    # 20,4,4: 5+2+15+36+3~61
    # 10,2,4: 2+1+5+13+2~23
    # 4,2,3: 1+1+2+6+1~11
    #
    # Conclusion:
    # direct search in state-space (e.g. Value Iteration)
    # is impractical for any interesting configuration.

    # for each elevator - in which floor is it?
    locations = H**N
    # for each elevator - in which direction is it moving?
    directions = 3**N
    # for each elevator and each floor -
    # how many passengers are carried towards that floor?
    carried = sum([fac(H)/(fac(H-i)*fac(i))
                       for i in range(capacity+1)])**N
    # for each elevator and each pair of floors -
    # are there (should be how many...) passengers waiting
    # for the elevator to take them between the floors?
    waiting_full = (2**(H**2))**N
    waiting_min = sum([fac(H**2)/(fac(H**2-i)*fac(i))
                       for i in range(capacity+1)])**N
    # unassigned passengers: for each pair of floors,
    # how many passengers are waiting for assignment.
    new_full = 2**(H**2)
    new_min = H**2 + 1

    states_full = [locations, directions, carried,
                   waiting_full, new_full]
    states_min = [locations, directions, carried,
                   waiting_min, new_min]

    print(sum([log(x) for x in states_full]))
    print([log(x) for x in states_full])
    print(sum([log(x) for x in states_min]))
    print([log(x) for x in states_min])


def count_time(N, verbose=True):
    # N=25M=2.5e7 takes ~1sec.
    # N=100B=1e11 should take ~1h.
    N = round(N)
    x = 0
    t = time.time()
    for i in range(N):
        x+=1
    t = time.time()-t
    if verbose: print(t)
    return t
