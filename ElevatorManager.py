
from MyTools import *
import Elevator

class NaiveManager:
    '''
    ElevatorManager

    The manager is called by the simulator in certain events
    (initialization, arrival or finished mission)
    and returns new missions for the elevators.

    This manager is a pathetically-simple one intended as basis for inheritance.

    Missions format:
    {elevator_index : list_of_missions}

    where a single mission is encoded as a 3D-tuple:
    (n,True,-1)      = go to floor n and open.
    (n,False,-1)     = go to floor n (without opening).
    (n,True/False,k) = get in the middle of another mission -
                       go to n and push it as the k'th task of the elevator.
    (None,False,k)   = remove the current k'th mission of the elevator.

    In cases of new arrival, the output dict must also include:
    {-1 : elevator_assigned_to_arrival}

    Written by Ido Greenberg, 2018
    '''
    def __init__(self, n_floors, n_elevators,
                 capacity, speed, open_time,
                 arrivals_pace=None, p_up=0.5, p_down=0.5, p_between=0., size=1., delay=3.):
        assert_zero(p_up+p_down+p_between-1)
        self.H = n_floors
        self.N = n_elevators
        self.capacity = capacity
        self.speed = speed
        self.open_time = open_time
        self.arrivals_info = { # "semi-cheating" info
            'pace':arrivals_pace,
            'p_up':p_up,
            'p_down':p_down,
            'p_between':p_between,
            'size':size,
            'delay':delay
        }
    def version_info():
        return ("NaiveManager",
                "Use the first elevator to handle passengers arrivals sequentially.")
    def initialize(self):
        pass
    def handle_initialization(self):
        return {}
    def handle_arrival(self, t, el, xi, xf):
        return {
            -1:0, # assign arrival to elevator 0
            0:[(xi,True,-1),(xf,True,-1)]  # add missions to elevator 0: go to arrival floor and open, then go to arrival destination and open
        }
    def handle_no_missions(self, t, el, idx):
        return {}


class NaiveRoundRobin(NaiveManager):
    def __init__(self, n_floors, n_elevators,
                 capacity, speed, open_time,
                 arrivals_pace=None, p_up=0.5, p_down=0.5, p_between=0., size=1., delay=3.):
        NaiveManager.__init__(self, n_floors, n_elevators, capacity, speed, open_time,
                     arrivals_pace, p_up, p_down, p_between, size, delay)
        self.robin = 0
    def version_info():
        return ("NaiveRoundRobin",
                "Use the elevators in turns to handle passengers arrivals.")
    def handle_arrival(self, t, el, xi, xf):
        self.robin = (self.robin + 1) % self.N
        return {
            -1:self.robin,
            self.robin:[(xi,True,-1),(xf,True,-1)]
        }


class GreedyManager(NaiveManager):

    def __init__(self, n_floors, n_elevators,
                 capacity, speed, open_time,
                 arrivals_pace=None, p_up=0.5, p_down=0.5, p_between=0., size=1., delay=3.):
        NaiveManager.__init__(self, n_floors, n_elevators, capacity, speed, open_time,
                     arrivals_pace, p_up, p_down, p_between, size, delay)
        # waiting dispersion state
        self.n0 = int(np.ceil(self.N/3))
        self.nn = self.N - self.n0
        delta = self.H / (2 * self.nn + 1)
        self.bases = [int(b) for b in np.round(2 * delta * (1 + np.arange(self.nn)))]
        # elevators status
        #self.bindings = [[] for _ in range(self.N)] # binding destinations (of onboard passengers)

    def version_info():
        return ("GreedyManager",
                '''Try to disperse waiting elevators,
                and assign elevators to passengers greedily.''')

    def handle_initialization(self):
        return {
            **{i:[(0,False,-1)] for i in range(self.n0)},
            **{self.n0+i:[(self.bases[i],False,-1)] for i in range(self.nn)}
        }

    def handle_arrival(self, t, el, xi, xf):
        # TODO (1) find least busy in terms of ETA rather than n_missions
        # is anyone already there?
        for i, e in enumerate(el):
            if not e.missions and e.x==xi:
                return self.send_elevator(i,xi,xf)
        # for priority p: find someone with p missions or whose p+1 mission is on the way
        for priority in range(1+max([len(e.missions) for e in el])):
            dmin = np.inf
            amin = ()
            imin = None
            for i, e in enumerate(el):
                d,a = self.available(e, priority, xi, xf)
                if a is None or dmin<=d:
                    continue
                dmin = d
                amin = a
                imin = i
            if imin is not None:
                return {
                    -1: imin,
                    imin: amin
                }

        raise EOFError("No elevator was found fit for mission in any priority.")

    def available(self, el, p, xi, xf):
        # is available after p'th mission?
        if len(el.missions) <= p:
            return abs(xi-el.locate(p)), [(xi, True, -1), (xf, True, -1)]
        if len(el.missions) == p+1 and el.missions[p] is not None:
            return abs(xi-el.locate(p)), [(None, False, p), (xi, True, -1), (xf, True, -1)]
        # is available on the way of p+1'th mission?
        # look for exi <= xi <= xf,exf
        exf = el.missions[p]
        if exf is None:
            return None,None
        dests1 = [m for m in el.missions[:p] if m is not None]
        exi = dests1[-1] if dests1 else el.x
        if not ((exi<=xi and xi<=xf and xi<=exf) or
                (exi>=xi and xi>=xf and xi>=exf)):
            return None,None
        # floors order is fit! now find where to locate the new mission
        missions = []
        # go to xi
        di = 0
        if not (xi==exi and p>=1 and el.missions[p-1] is None):
            missions.append((xi,True,p))
            di += 2
        # go to xf
        for a in range(len(el.missions)-p):
            if el.missions[p+a] is None:
                continue
            if xf < el.missions[p+a]:
                missions.append((xf,True,p+di+a))
                return 0,missions
            if xf == el.missions[p+a]:
                if len(el.missions)>=p+a+2 and el.missions[p+a+1] is not None:
                    missions.append((xf,True,p+di+a))
                return 0,missions
        missions.append((xf,True,-1))
        return 0,missions

    def send_elevator(self, i, xi, xf):
        return {
            -1: i,
            i: [(xi, True, -1), (xf, True, -1)]
        }

    def handle_no_missions(self, t, el, idx):
        # where's everyone intended to be?
        dests = [[m for m in e.missions if m is not None] for e in el]
        xfs = [ds[-1] if ds else e.x for e,ds in zip(el,dests)]
        assert(xfs[idx]==el[idx].x), "'no mission' elevator seems to have more missions."
        # are there enough grounds?
        grounds = self.N - np.count_nonzero(xfs)
        if grounds < self.n0:
            return {} if el[idx].x==0 else {idx:0}
        # what's the loneliest base floor?
        dists = [min([abs(x-base) for i,x in enumerate(xfs) if i!=idx])
                 for base in self.bases]
        lonely_base = np.argmax(dists)
        x = self.bases[lonely_base]
        return {} if el[idx].x==x else {idx:x}


if __name__ == "__main__":
    import ElevatorTester, matplotlib.pyplot as plt
    c = ElevatorTester.ELEVATOR_TESTS_CONFS[-1]
    c['sim_len'] = 120
    x = ElevatorTester.ManagerTester(GreedyManager, c, 1)
    x.single_test(c)
    plt.show()
