
from MyTools import *
import numpy as np
import pickle
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import ElevatorSimulator, ElevatorManager

'''
    ElevatorTester
    This module defines various scenarios, tests the managers of ElevatorManager
    using ElevatorSimulator, and summarizes the results.
    
    Written by Ido Greenberg, 2018
'''

ELEVATOR_TESTS_MANAGERS = (ElevatorManager.NaiveManager,
                           ElevatorManager.NaiveRoundRobin,
                           ElevatorManager.GreedyManager)

ELEVATOR_TESTS_CONFS = [
    {
        'name': 'Residence\n(low)',
        'sim_len': 60*60,
        'n_floors': 6,
        'n_elevators': 1,
        'capacity': 4,
        'speed': 1/3,
        'open_time': 3,
        'arrival_pace': 1/60,
        'p_between': 1/20,
        'p_up': 0.5-1/40,
        'size': 1.5,
        'delay': 3.5
    },
    {
        'name': 'Residence\n(small elevators)',
        'sim_len': 60*60,
        'n_floors': 16,
        'n_elevators': 2,
        'capacity': 4,
        'speed': 1/2,
        'open_time': 3,
        'arrival_pace': 1/24,
        'p_between': 1/20,
        'p_up': 0.5-1/40,
        'size': 1.5,
        'delay': 3.5
    },
    {
        'name': 'Residence\n(high)',
        'sim_len': 60*60,
        'n_floors': 50,
        'n_elevators': 6,
        'capacity': 8,
        'speed': 1,
        'open_time': 2,
        'arrival_pace': 1/8,
        'p_between': 1/20,
        'p_up': 0.5-1/40,
        'size': 1.5,
        'delay': 3.5
    },
    {
        'name': 'Office\n(low, slow open)',
        'sim_len': 60*60,
        'n_floors': 6,
        'n_elevators': 2,
        'capacity': 8,
        'speed': 1/2,
        'open_time': 5,
        'arrival_pace': 1/15,
        'p_between': 0.2,
        'p_up': 0.4,
        'size': 2,
        'delay': 2
    },
    {
        'name': 'Office\n(medium)',
        'sim_len': 60*60,
        'n_floors': 20,
        'n_elevators': 4,
        'capacity': 8,
        'speed': 1,
        'open_time': 2,
        'arrival_pace': 1/8,
        'p_between': 0.2,
        'p_up': 0.4,
        'size': 2.5,
        'delay': 2
    },
    {
        'name': 'Office\n(high)',
        'sim_len': 60*60,
        'n_floors': 40,
        'n_elevators': 8,
        'capacity': 12,
        'speed': 1,
        'open_time': 2,
        'arrival_pace': 1/4,
        'p_between': 0.2,
        'p_up': 0.4,
        'size': 2.5,
        'delay': 2
    }
]

# SIMULATOR POSSIBLE ARGUMENTS:
# def __init__(self, manager=ElevatorManager.NaiveManager, debug_mode=False, verbose=True,
#              sim_len=120, sim_pace=None, time_resolution=np.inf, logfile=None, seed=1,
#              n_floors=3, n_elevators=2, capacity=4, speed=1, open_time=2,
#              arrival_pace=1 / 10, p_between=0.1, p_up=0.45, size=1.5, delay=3)


class ManagerTester:

    def __init__(self, manager=ElevatorManager.NaiveManager, configurations=(),
                 show=False):
        self.manager = manager
        self.configurations = list(configurations)
        self.name = manager.version_info()
        self.results = []
        self.totals = {}
        self.show = show

    def reset(self):
        self.results = []
        self.totals = {}

    def add_configurations(self, configurations):
        self.configurations.extend(configurations)

    def test(self):
        for conf in self.configurations:
            self.results.append(self.single_test(conf))

    def single_test(self, conf):
        if "name" in conf:
            print('Warning: unexpected "name" field in configuration.')
            del (conf["name"])
        x = ElevatorSimulator.Simulator(debug_mode=self.show>=2, verbose=self.show,
                                        sim_pace=10 if self.show else None,
                                        manager=self.manager, **conf)
        x.generate_scenario()
        summary = x.run_simulation()
        return summary

    def summarize(self):
        self.totals['avg time'] = \
            np.mean([S['goals']['service_time'][ 1] for S in self.results])
        self.totals['max time'] = \
            np.mean([S['goals']['service_time'][-1] for S in self.results])
        self.totals['dist per task'] = \
            np.mean([sum(S['goals']['total_distance'])/S['passengers']['served']
                     for S in self.results])
        self.totals['std(dist)'] = \
            np.mean([np.std(S['goals']['total_distance'])/S['passengers']['served']
                     for S in self.results])
        self.totals['unassigned'] = \
            np.mean([S['sanity']['unassigned_passengers'] for S in self.results])
        self.totals['indirect'] = \
            np.mean([S['sanity']['indirect_motions'][0] for S in self.results])
        self.totals['bad opens'] = \
            np.mean([S['sanity']['unnecessary_opens'] for S in self.results])
        self.totals['blocked'] = \
            np.mean([S['sanity']['blocked_entrances'] for S in self.results])
        self.totals['non-served'] = \
            np.mean([S['passengers']['arrived']-S['passengers']['served'] for S in self.results])
        self.totals['non-served [%]'] = \
            np.mean([100*(1-S['passengers']['served']/S['passengers']['arrived'])
                     for S in self.results])
        return self.totals.copy()


class Tester:

    def __init__(self, managers=(ElevatorManager.NaiveManager,),
                 configurations=({},)):
        self.conf_names = [conf['name'] for conf in configurations]
        for conf in configurations: del(conf['name'])
        self.configurations = list(configurations)
        self.managers = managers
        self.names = [m.version_info()[0] for m in self.managers]
        self.scores = [] # service time per scenario
        self.served = [] # completely-served passengers [%]
        self.results = [] # other results averaged over scenarios

    def reset(self):
        self.results = []

    def add_configurations(self, configurations):
        self.configurations.extend(configurations)

    def test(self):
        for m in self.managers:
            x = ManagerTester(m, self.configurations)
            x.test()
            self.scores.append([S['goals']['service_time'][1] for S in x.results])
            s = x.summarize()
            self.served.append(100-s['non-served [%]'])
            del(s['non-served [%]'])
            self.results.append(s)

    def summarize_results(self):
        self.print_results()
        self.plot_results()

    def print_results(self):
        if not self.results:
            print("No available results (did you run test()?).")
            return
        titles = ['Manager'] + list(self.results[0].keys())
        t = PrettyTable(titles)
        for tit,row in zip(self.names,self.results):
            t.add_row([tit]+[int(val+0.5*np.sign(val)) for val in list(row.values())])
        print(t)

    def plot_results(self):
        M = len(self.managers)
        f, axs = plt.subplots(2, 2)
        # service time
        ax = axs[0,0]
        quants = tuple(range(0,101))
        for s in self.scores:
            p = ax.plot(quants, dist(s, quants)[2:])
            ax.hlines(y=np.mean(s), xmin=0, xmax=100, linestyles='dashed', color=p[0].get_color())
        ax.set_xlim((0,100))
        ax.set_ylim((0,None))
        ax.legend(tuple(nm+f' ({served:.0f}%)' for nm,served in zip(self.names,self.served)))
        ax.set_xlabel('Scenario-quantile [%]')
        ax.set_ylabel('Service Time [s]')
        ax.set_title('Service Time Distribution (fully-served passengers only)')
        # service time
        # ax = axs[0,0]
        # ax.bar(tuple(range(M)), [r['avg time'] for r in self.results], color='magenta')
        # ax.bar(tuple(range(M)), [r['max time']-r['avg time'] for r in self.results],
        #        bottom=[r['avg time'] for r in self.results], color='navy')
        # ax.set_ylabel('Service Time [sec / passenger]')
        # ax.set_xlabel('Manager')
        # ax.set_title('Service Time')
        # ax.set_xticks(tuple(range(M)))
        # ax.set_xticklabels(self.names)
        # ax.legend(('Average','Max (averaged over scenarios)'))
        # elevators distance
        ax = axs[0,1]
        ax.bar(tuple(range(M)), [r['dist per task'] for r in self.results],
        yerr=[r['std(dist)'] for r in self.results], color='green', alpha=0.8)
        ax.set_ylabel('Elevators Distance [floors per task]')
        ax.set_xlabel('Manager')
        ax.set_title('Elevators Distance')
        ax.set_xticks(tuple(range(M)))
        ax.set_xticklabels(self.names)
        # service time per scenario
        #self.scores
        ax = axs[1,0]
        C = len(self.configurations)
        bar_width = 1/(M+1)
        for i,s in enumerate(self.scores):
            ax.bar(np.arange(C)+i/(M+1), s, bar_width, alpha=0.8)
        ax.set_ylabel('Average Service Time [s]')
        ax.set_title('Service Time per Scenario')
        ax.set_xticks(np.arange(C)+0.5-1/(M+1))
        ax.set_xticklabels(self.conf_names)
        ax.legend(self.names)
        # bad incidences
        ax = axs[1,1]
        bar_width = 1/(M+1)
        for i,r in enumerate(self.results):
            ax.bar(np.arange(5)+i/(M+1),
                   (r['unassigned'],r['indirect'],r['bad opens'],r['blocked'],r['non-served']),
                   bar_width, alpha=0.8)
        ax.set_ylabel('Occurences')
        ax.set_title('Bad Behavior')
        ax.set_xticks(np.arange(5)+0.5-1/(M+1))
        ax.set_xticklabels(('Unassigned\npassengers', 'Indirect\ntravels', 'Unnecessary\nopens',
                            'Blocked\nentrances', 'Non-served\npassengers'))
        ax.legend(self.names)
        # draw
        plt.get_current_fig_manager().window.showMaximized()
        plt.draw()
        plt.pause(1e-17)

    def save(self, filename="ElevatorTester"):
        if not filename.endswith(".pickle"):
            filename += ".pickle"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

def load_tester(filename="ElevatorTester"):
    if not filename.endswith(".pickle"):
        filename += ".pickle"
    with open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    x = Tester(ELEVATOR_TESTS_MANAGERS, ELEVATOR_TESTS_CONFS)
    x.test()
    x.summarize_results()
    plt.show()
