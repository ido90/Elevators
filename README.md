# Elevators Management: Visual Simulator and Optimization Algorithms

This repo implements a visual simulator of elevators system, along with several simple optimization algorithms and analysis of their results.

A [Reinforcement-Learning manager](#module-reinforcementelevator) is intended to be implemented and tested vs. the classic [DirectManager](#implemented-managers).

## Implemented managers

Note: all the currently-implemented managers are either naive or imcomplete, and were mainly used for testing and demonstration of the simulative infrastructure.

- **NaiveManager**: Use the first elevator to handle passengers arrivals sequentially.
- **NaiveRoundRobin**: Use the elevators in turns to handle passengers arrivals.
- **GreedyManager**: Try to disperse waiting elevators, and assign elevators to passengers greedily.
- **DirectManager**: Go on while there're more passengers in the current motion direction, then turn around (variant of the classic elevator algorithm).

## Class: ElevatorSimulator.Simulator

This class implements a simulation of elevators which serve arriving passengers.

Involved classes:
- Simulator       = simulation manager
- ElevatorManager = decision makers
- Elevator        = represent the elevators in the simulation
- Arrival         = encode a single event of arrival of passengers
- Passenger       = encode and track a single passenger
- SimPlot         = visualize the simulation dynamically via matplotlib

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

| ![](https://idogreenberg.neocities.org/linked_images/elevators.JPG) |
| :--: |
| A screenshot from the visual simulation |

## Module: ElevatorTester

This module defines various scenarios, tests the managers of ElevatorManager using ElevatorSimulator, and summarizes the results.

## Module: ElevatorManager

This module contains the elevator-managers (one class per manager).

A manager handles 3 kinds of events:
1. Initialization of elevators locations.
2. Arrival of passengers.
3. End of tasks of a certain elevator.

A manager can assign tasks in the format {elevator_index : list_of_missions}
where a single task is encoded as a 3D-tuple:
- (n,True,-1)      = go to floor n and open.
- (n,False,-1)     = go to floor n (without opening).
- (n,True/False,k) = get in the middle of another mission - go to n and push it as the k'th task of the elevator.
- (None,False,k)   = remove the current k'th mission of the elevator.

In cases of new arrival, the output dict must also include: {-1 : elevator_assigned_to_arrival}

| ![](https://github.com/ido90/Elevators/blob/master/Demonstrations/tests%20summary.JPG) |
| :--: |
| Summary of the results of the various managers |


## Module: ReinforcementElevator

This module is **NOT IMPLEMENTED**, up to definition of states and a simple count of them (or at least a lower-bound of the count), demonstrating that direct search in state-space (e.g. Value Iteration) is impractical for any interesting configuration.
Instead, some encoding of the states should be used (e.g. like [here](https://papers.nips.cc/paper/1073-improving-elevator-performance-using-reinforcement-learning.pdf)).

Implementation of this module should take care of the following issues:
1. **Sampling resolution**: high-resolution (e.g. sample every time the elevators move one floor) permits simple state-space, but low-resolution (e.g. sample when an elevator reaches its destination, etc.) is better synced with the simulator interface.
2. **State encoding**: compact encoding of the states so that a learner can use an encoded state to make a decision.
3. **Train & test process**: train the decision-maker in various scenarios and test it.
