# Elevators
Elevators simulator intended to test various algorithms for elevators managing.

Implemented managers

    NOTE: ALL CURRENTLY-IMPLEMENTED MANAGERS ARE EITHER NAIVE OR INCOMPLETE, AND ARE MAINLY USED FOR DEMONSTRATION OF THE SIMULATIVE INFRASTRUCTURE.

    NaiveManager: Use the first elevator to handle passengers arrivals sequentially.
    NaiveRoundRobin: Use the elevators in turns to handle passengers arrivals.
    GreedyManager: Try to disperse waiting elevators, and assign elevators to passengers greedily.
    DirectManager: Go on while there're more passengers in the current motion direction, then turn around (variant of the classic elevator algorithm).

Class: ElevatorSimulator.Simulator

    This class implements a simulation of elevators which serve arriving passengers.

    Involved classes:
    Simulator       = simulation manager
    ElevatorManager = decision makers
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


Module: ElevatorTester

    This module defines various scenarios, tests the managers of ElevatorManager
    using ElevatorSimulator, and summarizes the results.


Module: ElevatorManager

    This module contains the elevator managers (one class per manager).
    
    The manager handles 3 kinds of events:
    1. Initialization of elevators locations.
    2. Arrival of passengers.
    3. End of tasks of a certain elevator.
    
    The manager can assign tasks in the following format:
    {elevator_index : list_of_missions}
    where a single task is encoded as a 3D-tuple:
    (n,True,-1)      = go to floor n and open.
    (n,False,-1)     = go to floor n (without opening).
    (n,True/False,k) = get in the middle of another mission -
                       go to n and push it as the k'th task of the elevator.
    (None,False,k)   = remove the current k'th mission of the elevator.
    In cases of new arrival, the output dict must also include:
    {-1 : elevator_assigned_to_arrival}
