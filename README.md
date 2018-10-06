# Elevators
Elevators simulator intended to test various algorithms for elevators managing.

Class: ElevatorSimulator.Simulator
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


Module: ElevatorTester
    This module defines various scenarios, tests the managers of ElevatorManager
    using ElevatorSimulator, and summarizes the results.


Module: ElevatorManager
    This module contains the elevator managers (one class per manager).
    
