
from MyTools import *
import ElevatorManager
from Elevator import Elevator
from Passenger import Arrival, Passenger
from SimulationPlotter import SimPlotter
import matplotlib.pyplot as plt
import numpy as np
import sys, pprint
from warnings import warn


class Simulator:
    '''
    Simulator
    This class implements a simulation of elevators which serve arriving passengers.

    Involved classes:
    Simulator       = simulation manager
    ElevatorManager = decision maker (very naive algorithm intended to be inherited
                      by smarter managers which would overwrite its methods)
    Elevator        = represent the elevators in the simulation
    Arrival         = encode a single event of arrival of passengers
    Passenger       = encode and track a single passenger
    SimPlot         = visualize the simulation dynamically via matplotlib

    Main flow:
    generate_scenario()
    run_simulation()
        sim_initialize()
            Manager.handle_initialization()
            update_missions()
        loop: handle_next_event()
            update the state to the time of the next event: sim_update()
            handle next event: handle_arrival() / end_mission() / sim_end()
                Manager.handle_arrival() / Manager.handle_no_missions()
                update_missions()

    Written by Ido Greenberg, 2018
    '''
    # TODO add stop length instead of the missing acceleration (and reduce it from the open & close time)

    def __init__(self, manager=ElevatorManager.NaiveManager, debug_mode=False, verbose=True,
                 sim_len=120, sim_pace=None, time_resolution=0.5, logfile=None, seed=1,
                 n_floors=3, n_elevators=2, capacity=4, speed=1, open_time=2,
                 arrival_pace=1/10, p_between=0.1, p_up=0.45, size=1.5, delay=3):

        # Note: default time unit is seconds.
        self.debug = debug_mode
        self.verbose = verbose

        ## Simulation world
        # conf
        self.sim_len = sim_len
        sim_pace = sim_pace # visualization's fast-forward; None for no visualization
        self.time_resolution = np.inf if sim_pace is None else time_resolution
        self.logfile = logfile
        self.seed = seed
        # init
        self.sim_time = 0
        self.real_time = None
        self.logger = None
        # stats init
        self.useless_opens = 0
        self.blocked_entrances = 0
        self.moves_without_open = 0

        ## Elevators
        # conf
        self.n_floors = n_floors
        self.n_elevators = n_elevators
        el_capacity = capacity
        el_speed = speed # floors per second. Note: acceleration is neglected
        el_open_time = open_time # = time to open = time to close
        # init
        self.el = [Elevator(i, self.n_floors, el_capacity, el_speed, el_open_time)
                   for i in range(self.n_elevators)]
        self.sim_plot = SimPlotter(self.n_floors, self.n_elevators, sim_pace) \
            if sim_pace is not None else None

        ## Passengers
        # conf
        self.arrivals_pace = arrival_pace # arrivals per second
        self.p_go_between = p_between
        self.p_go_up = p_up
        self.p_go_down = 1 - (self.p_go_between + self.p_go_up)
        self.arrival_size = size # mean number of passengers per arrival
        self.delay = delay # typical (not exactly mean) delay on passengers entrance
        # init
        self.scenario = []
        self.future_arrivals = []
        self.waiting_passengers = []
        self.moving_passengers = [[] for _ in range(self.n_elevators)]
        self.completed_passengers = []

        ## Manager
        self.manager_info = manager.version_info()
        self.manager = manager(self.n_floors, self.el,
                                       el_capacity, el_speed, el_open_time,
                                       self.arrivals_pace,
                                       self.p_go_up, self.p_go_down, self.p_go_between,
                                       self.arrival_size, self.delay)

    def generate_scenario(self, verbose=None):

        if verbose is None: verbose = self.debug
        if self.seed is not None: np.random.seed(self.seed)

        n_arrivals = np.random.poisson(self.sim_len * self.arrivals_pace)

        a_times = list(np.sort(self.sim_len * np.random.rand(n_arrivals)))
        a_sizes = list(np.floor(1+np.random.exponential(self.arrival_size-1,n_arrivals)).astype(int))
        a_delays = list(np.random.lognormal(0,np.sqrt(self.delay),n_arrivals))
        a_types = list(np.random.choice(['up', 'down', 'between'], n_arrivals, True,
                                               (self.p_go_up, self.p_go_down, self.p_go_between)))
        a_from = [(0 if tp=='up' else int(1+self.n_floors*np.random.rand()))
                  for tp in a_types]
        a_to = [(0 if tp=='down' else
                 int(1+self.n_floors*np.random.rand()) if tp=='up' else
                 np.random.choice(list(set(list(range(self.n_floors+1)))-{0,a_from[i]})) )
                for i,tp in zip(range(n_arrivals),a_types)]

        self.scenario = tuple([Arrival(t,n,d,xi,xf)
                         for (t,n,d,xi,xf) in zip(a_times,a_sizes,a_delays,a_from,a_to)])

        if verbose:
            for i,arr in enumerate(self.scenario):
                arr.print(i)

    def run_simulation(self):
        self.sim_initialize()
        end_sim = False
        while not end_sim:
            end_sim = self.handle_next_event()
        return end_sim

    def sim_initialize(self):
        if self.verbose: print("\n\nSIMULATION BEGAN\n")
        self.logger = open(self.logfile, 'w') if self.logfile else None
        # world
        self.sim_time = 0
        # arrivals
        self.future_arrivals = list(self.scenario)
        self.waiting_passengers = []
        self.moving_passengers = [[] for _ in range(self.n_elevators)]
        self.completed_passengers = []
        # elevators
        for el in self.el:
            el.initialize()
        # manager
        self.manager.initialize()
        missions = self.manager.handle_initialization()
        self.update_missions(missions)
        # stats
        self.useless_opens = 0
        self.blocked_entrances = 0
        self.moves_without_open = 0
        # visualization
        if self.sim_plot is not None:
            self.sim_plot.initialize()
        self.real_time = PrintTime()

    def handle_next_event(self):
        # Find next event's time
        t_arrival = self.future_arrivals[0].t if self.future_arrivals else np.inf
        t_finish_mission = min([el.next_t for el in self.el])
        t_forced_update = self.sim_time + self.time_resolution
        t = min(t_forced_update, t_arrival, t_finish_mission, self.sim_len)

        # update simulation to next event's time
        dt = self.sim_update(t)

        # handle next event
        if t_forced_update < min(self.sim_len, t_arrival, t_finish_mission):
            pass
        elif self.sim_len < min(t_arrival, t_finish_mission):
            self.update_plot(dt)
            summary = self.sim_end()
            return summary
        elif t_arrival < t_finish_mission:
            self.handle_arrival()
        else:
            self.end_mission()

        self.update_plot(dt)

        return 0

    def sim_update(self, t):
        dt = t - self.sim_time
        assert(dt>=0), dt
        if dt==0:
            return dt
        # update elevators location
        for el in self.el:
            if el.missions and el.missions[0] is not None:
                el.x += np.sign(el.missions[0]-el.x) * el.speed * dt
                el.total_distance += el.speed * dt
        self.sim_time = t
        return dt

    def handle_arrival(self):
        a = self.future_arrivals[0]
        if self.debug:
            self.log("Arrive", f"n={a.n:d}\t({a.d:.1f})\t{a.xi:d} -> {a.xf:d}")
        del(self.future_arrivals[0])
        new_passengers = [Passenger(a) for _ in range(a.n)]
        missions = self.manager.handle_arrival(self.sim_time, a.xi, a.xf)
        if self.debug:
            print(missions)
        if -1 in missions:
            for ps in new_passengers:
                ps.assigned_el = missions[-1]
        self.waiting_passengers.extend(new_passengers)
        self.update_missions(missions)

    def end_mission(self):
        i_el = int(np.argmin([el.next_t for el in self.el]))
        el = self.el[i_el]
        m = el.missions[0]
        del(el.missions[0])

        # end previous mission
        if m is not None:
            assert_zero(m-el.x, eps=1e-10)
            el.x = m
            if self.debug:
                self.log("Moved", f"#{i_el:02d}\t-> {el.x:d}")
            # detect move without open
            if el.missions and el.missions[0] is not None:
                self.moves_without_open += 1

        # begin new mission
        if not el.missions:
            missions = self.manager.handle_no_missions(self.sim_time, i_el)
            if self.debug:
                print(missions)
            if not i_el in missions or not missions[i_el]:
                el.sleep()
            self.update_missions(missions)
        elif el.missions[0] is None:
            delay = self.open_el(i_el)
            el.open(self.sim_time, delay)
        else:
            el.move(self.sim_time)
            for ps in self.moving_passengers[i_el]:
                if el.motion != np.sign(ps.xf-el.x):
                    ps.indirect_motion += 1
                    if self.debug:
                        self.log("INDIRECT", f"{el.motion:d} != {el.x:d}->{ps.xf:d}")
            if self.debug:
                self.log("Moving", f"#{i_el:02d}\t-> {el.missions[0]:d}")

    def update_missions(self, missions):
        '''
        Get manager missions per elevator in format:
          (destination, whether_to_open, mission_to_split/remove)
        convert to elevator missions in format:
          n for moving to floor n, None for opening and loading.
        and add to elevators missions lists.
        '''
        for i_el in missions:
            if i_el == -1: continue # elevator assignment rather than mission
            el = self.el[i_el]
            immediate_mission = missions[i_el] and not el.missions

            for m in missions[i_el]:
                if m[0] is None:
                    # remove mission m: (None, *, m)
                    assert(m[2]>=0)
                    del(el.missions[m[2]])
                    if m[2]==0: immediate_mission = True
                elif m[2]==-1:
                    # new mission: (destination floor, open/not, -1)
                    el.missions.append(m[0])
                    if m[1]: el.missions.append(None)
                else:
                    # split existing mission m: (destination floor, open/not, m)
                    assert(el.missions[m[2]] is not None), "Trying to split an OPEN mission."
                    el.missions.insert(m[2], m[0])
                    if m[1]: el.missions.insert(m[2]+1, None)
                    if m[2]==0: immediate_mission = True

            if immediate_mission:
                if el.missions[0] is None:
                    warn("Unexpected mission assignment: open which does not follow motion.")
                    delay = self.open_el(i_el)
                    el.open(self.sim_time, delay)
                else:
                    el.move(self.sim_time)

        if self.debug:
            print([el.missions for el in self.el])

    def open_el(self, i):
        el = self.el[i]
        any_activity = False
        blocked_entrance = False

        # exiting passengers
        picked_up = []
        for j,ps in enumerate(self.moving_passengers[i]):
            if ps.xf == el.x:
                any_activity = True
                picked_up.append(j)
                self.completed_passengers.append(ps)
                ps.t2 = self.sim_time
                if self.debug:
                    self.log("Exit", f"#{i:02d}\tt={ps.t2-ps.t0:.0f}s")
        for j in sorted(picked_up, reverse=True):
            del self.moving_passengers[i][j]

        # entering passengers
        delay = 0
        picked_up = []
        for j,ps in enumerate(self.waiting_passengers):
            if ps.xi == el.x and ps.assigned_el == i:
                if el.capacity <= len(self.moving_passengers[i]):
                    # Elevator is full - count block and re-push the button
                    blocked_entrance = True
                    missions = self.manager.handle_arrival(self.sim_time, ps.xi, ps.xf)
                    if -1 in missions: ps.assigned_el = missions[-1]
                    else: ps.assigned_el = -1
                    self.update_missions(missions)
                    if self.debug: self.log("Blocked", f"#{i:02d}")
                    continue
                any_activity = True
                picked_up.append(j)
                self.moving_passengers[i].append(ps)
                ps.t1 = self.sim_time
                delay = max(delay, ps.d)
                if self.debug: self.log("Enter", f"#{i:02d}\tt={ps.t1-ps.t0:.0f}s")
        for j in sorted(picked_up, reverse=True):
            del self.waiting_passengers[j]

        if blocked_entrance: self.blocked_entrances += 1
        if not any_activity:
            self.useless_opens += 1
            if self.debug: self.log("USELESS", f"#{i:02d}")

        return delay

    def sim_end(self, verbose=None):
        if verbose is None: verbose = self.verbose
        if self.verbose: print("\nSIMULATION FINISHED\n\n")
        # classify passengers
        n_ps_scenario = sum([a.n for a in self.scenario])
        n_ps_completed = len(self.completed_passengers)
        n_ps_moving = sum([len(ps) for ps in self.moving_passengers])
        n_ps_waiting = len(self.waiting_passengers)
        n_ps_future = len(self.future_arrivals)
        assert(n_ps_future == 0), n_ps_future
        assert(n_ps_scenario == n_ps_completed + n_ps_moving + n_ps_waiting)
        moving_max_time = max([self.sim_time-ps.t1 for ps_list in self.moving_passengers for ps in ps_list]) \
            if n_ps_moving else 0
        waiting_max_time = max([self.sim_time-ps.t0 for ps in self.waiting_passengers]) \
            if n_ps_waiting else 0
        # summarize
        summary = {
            "general": {
                "time": self.sim_time,
                "runtime": int(time.time()-self.real_time.ti+0.5)
            },
            "goals": {
                "service_time": dist([ps.t2-ps.t0 for ps in self.completed_passengers], do_round=1),
                "total_distance": [el.total_distance for el in self.el],
                "density": None # not implemented
            },
            "passengers": {
                "arrived": n_ps_scenario,
                "served": n_ps_completed,
                "on_board": [n_ps_moving, moving_max_time],
                "waiting": [n_ps_waiting, waiting_max_time]
            },
            "sanity": {
                "unassigned_passengers": sum([ps.assigned_el==-1 for ps in self.waiting_passengers]),
                "unnecessary_opens": self.useless_opens,
                "blocked_entrances": self.blocked_entrances,
                "indirect_motions": [
                    sum([ps.indirect_motion>0 for ps in self.completed_passengers]),
                    sum([ps.indirect_motion   for ps in self.completed_passengers])
                ]
            },
            "info": {
                "waiting_time": dist([ps.t1 - ps.t0 for ps in self.completed_passengers], do_round=1),
                "inside_time":  dist([ps.t2 - ps.t1 for ps in self.completed_passengers], do_round=1),
                "total_opens": [el.total_opens for el in self.el],
                "moves_without_open": self.moves_without_open,
                "remaining_missions": [len(el.missions) for el in self.el]
            }
        }

        if verbose:
            if self.debug:
                pprint.pprint(summary)
            self.plot_results(summary)

        if self.logger: self.logger.close()

        return summary

    def plot_results(self, S):
        f, axs = plt.subplots(2, 2)
        # service time
        ax = axs[0,0]
        quants = tuple(range(0,101))
        t_tot = dist([ps.t2 - ps.t0 for ps in self.completed_passengers], quants)
        t_wait = dist([ps.t1 - ps.t0 for ps in self.completed_passengers], quants)
        t_inside = dist([ps.t2 - ps.t1 for ps in self.completed_passengers], quants)
        ax.plot(quants, t_tot[2:], 'k-')
        ax.plot(quants, t_wait[2:], 'm-')
        ax.plot(quants, t_inside[2:], 'y-')
        ax.hlines(y=t_tot[1],    xmin=0, xmax=100, linestyles='dashed', color='k')
        ax.hlines(y=t_wait[1],   xmin=0, xmax=100, linestyles='dashed', color='m')
        ax.hlines(y=t_inside[1], xmin=0, xmax=100, linestyles='dashed', color='y')
        ax.set_xlim((0,100))
        ax.set_ylim((0,None))
        ax.legend(("Total", "Outside", "Inside"))
        ax.set_xlabel('Quantile [%]')
        ax.set_ylabel('Time [s]')
        ax.set_title('Service Time Distribution (for fully-served passengers)')
        # passengers
        ax = axs[0,1]
        s = S['passengers']
        ax.bar(list(range(3)), [s['waiting'][0],s['on_board'][0],s['served']], color='k')
        ax.set_ylabel('Passengers')
        ax.set_title('Eventual Passengers Status')
        ax.set_xticks(list(range(3)))
        ax.set_xticklabels((f"Waiting\n(longest={s['waiting'][1]:.0f}[s])",
                            f"On-board\n(longest={s['on_board'][1]:.0f}[s])",
                            "Served"))
        # bad incidences
        ax = axs[1,0]
        s = S['sanity']
        ax.bar(list(range(4)),
                    [s['unassigned_passengers'],s['indirect_motions'][0],s['unnecessary_opens'],s['blocked_entrances']],
                    color='r'
                    )
        ax.set_ylabel('Occurences')
        ax.set_title('Bad Behavior')
        ax.set_xticks(list(range(4)))
        ax.set_xticklabels(('Unassigned\npassengers', 'Indirect\ntravels', 'Unnecessary\nopens', 'Blocked\nentrances'))
        # text info
        ax = axs[1,1]
        text1 = '\n'.join((
            f"        ADDITIONAL INFO",
            f"Simulation:",
            f"    Time:            {S['general']['time']:.0f}",
            f"    Runtime:         {S['general']['runtime']:.0f}",
            f"Elevators:",
            f"    Total distance:  " + ", ".join([f"{x:.0f}" for x in S['goals']['total_distance']]),
            f"    Total opens:     " + ", ".join([f"{x:.0f}" for x in S['info']['total_opens']]),
            f"    Non-open moves:  {S['info']['moves_without_open']:.0f}",
            f"    Remaining tasks: " + ", ".join([f"{x:.0f}" for x in S['info']['remaining_missions']])
        ))
        text2 = '\n'.join((
            f"        CONFIGURATION",
            f"Manager:         {self.manager_info[0]:s}",
            f"World:",
            f"    Time length: {self.sim_len:.0f}",
            f"    Floors:      {self.n_floors:.0f}",
            f"    Elevators:   {self.n_elevators:.0f}",
            f"    Seed:        {self.seed:f}",
            f"Elevators:",
            f"    Capacity:    {self.el[0].capacity:.0f}",
            f"    Speed:       {self.el[0].speed:.0f}",
            f"    Open time:   {self.el[0].open_time:.0f}",
            f"Passengers:",
            f"    Average time between: {self.arrivals_pace:.0f}",
            f"    Average number:       {self.arrival_size:.0f}",
            f"    Typical delay:        {self.delay:.0f}",
            f"    Going up:             {100*self.p_go_up:.0f}%",
            f"    Going down:           {100*self.p_go_down:.0f}%",
            f"    Going between:        {100*self.p_go_between:.0f}%"
        ))
        ax.text(0.05, 0.95, text1, transform=ax.transAxes, family='monospace',
                fontsize=10, verticalalignment='top')
        ax.text(0.55, 0.95, text2, transform=ax.transAxes, family='monospace',
                fontsize=10, verticalalignment='top')
        ax.set_xticks(())
        ax.set_yticks(())
        # draw
        plt.get_current_fig_manager().window.showMaximized()
        plt.draw()
        plt.pause(1e-17)

    def log(self, event, data, t=None):
        if t is None: t = self.sim_time
        if self.logger is None:
            print('\t'.join([f"[{t:.1f}]", f"{event:s}:", data]))
        else:
            self.logger.write('\t'.join([f"[{t:.1f}]", f"{event:s}:", data])+'\n')

    def update_plot(self, dt):
        if self.sim_plot is not None:
            self.sim_plot.update_plot(dt, self.el,
                                      [wp.xi for wp in self.waiting_passengers],
                                      [len(mp) for mp in self.moving_passengers])


if __name__ == "__main__":
    sim_pace = 100 if 'verbose' in sys.argv else None
    arrival_pace = 1/8
    x = Simulator(sim_pace=sim_pace, arrival_pace=arrival_pace,
                  manager=ElevatorManager.NaiveRoundRobin)
    x.generate_scenario()
    summary = x.run_simulation()
    plt.show()
