#!/usr/bin/env python3
#
"""
This module contains code that solves the Vorto.ai coding challenge.

For this challenge, you will submit a program that solves a version of the Vehicle Routing Problem
(VRP).

The VRP specifies a set of loads to be completed efficiently by an unbounded number of drivers.

Each load has a pickup location and a dropoff location, each specified by a Cartesian point.
A driver completes a load by driving to the pickup location, picking up the load, driving to the
dropoff, and dropping off the load. The time required to drive from one point to another, in
minutes, is the Euclidean distance between them. That is, to drive from (x1, y1) to (x2, y2) takes
sqrt((x2-x1)^2 + (y2-y1)^2) minutes.

As an example, suppose a driver located at (0,0) starts a load that picks up at (50,50) and delivers
at (100,100). This would take 2*sqrt(2*50^2) = ~141.42 minutes of drive time to complete:
sqrt((50-0)^2 + (50-0)^2) minutes to drive to the pickup, and sqrt((100-50)^2 + (100-50)^2) minutes
to the dropoff.

Each driver starts and ends his shift at a depot located at (0,0). A driver may complete multiple
loads on his shift, but may not exceed 12 hours of total drive time. That is, the total Euclidean
distance of completing all his loads, including the return to (0,0), must be less than 12*60.

A VRP solution contains a list of drivers, each of which has an ordered list of loads to be
completed. All loads must be assigned to a driver.

The total cost of a solution is given by the formula:

     total_cost = 500*number_of_drivers + total_number_of_driven_minutes

A good program will produce a solution with a low total cost, but does not take too long to run
(see Evaluation section below).

Program Requirements

You must provide a program that takes a text file path describing a VRP as a command line argument,
and writes a solution to stdout.

We will accept solutions written in Go, Javascript, Typescript, Python, Java, C or C++. Please make
sure your solution is version controlled using Git. All work must be your own. You may research and
implement algorithms you find online (papers, blog posts, etc.), but you must provide references to
all external sources. Please avoid using Google's OR-Tools.

The problem input contains a list of loads. Each load is formatted as an id followed by pickup and
dropoff locations in (x,y) floating point coordinates. An example input with four loads is:

loadNumber pickup dropoff
1 (-50.1,80.0) (90.1,12.2)
2 (-24.5,-19.2) (98.5,1,8)
3 (0.3,8.9) (40.9,55.0)
4 (5.3,-61.1) (77.8,-5.4)

Your program must write a solution to stdout. The solution should list, on separate lines, each
driver's ordered list of loads as a schedule. An example solution to the above problem could be:

[1]
[4,2]
[3]

This solution means one driver does load 1; another driver does load 4 followed by load 2; and a
final driver does load 3. Do not print anything else to stdout in your submission.

For example, if your solution is a python script called "mySubmission.py", then your program should
run with the following command:

python mySubmission.py {path_to_problem}

This should load the problem described in the file {path_to_problem}, run your algorithm, and print
a solution to stdout in the correct format.
"""


import csv
import sys
import time

import numpy as np


# Type hint definitions.
#
Location = tuple[float, float]
LocationSegment = tuple[Location, Location]
IndexSegment = tuple[int, int]

Array2D = np.ndarray[tuple[int, int], float]


# Constants of the problem.
#
BASE_COST = 500.0  # The cost to begin a new route loop.
BIG_NUMBER = 1.0e100
CONVERGENCE = 1.0e-4
GENERATIONS = 1000
MAX_LOOP_COST = 500.0 + 12 * 60.0  # The exclusive upper bound on the cost of a route loop.
MAX_TIME = 60.0
CULLING_SIZE = 100
POPULATION_SIZE = 1000
STOP_COUNT = 5

# Print diagnostics if DEBUG = True.
#
DEBUG = True


def distance(point_a: Location, point_b: Location) -> float:
    """
    Calculate the distance between point A and point B.

    Parameters
    ----------
    point_a : Location
        A tuple describing the x,y location of point A.
    point_b : Location
        A tuple describing the x,y location of point B.

    Returns
    -------
    float
        The Euclidean distance between the points.
    """

    # Calculate and return the distance.
    #
    dist = np.sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)

    return dist


def read_load_file(path: str) -> dict[int, LocationSegment]:
    """
    Read a list of load locations from a file.

    Parameters
    ----------
    path : str
        The path to the file containing a list of load locations.

    Returns
    -------
    dict[int, Segment]
        A dict keyed by load number. Each entry contains the pickup and dropoff point for the load
        as a pair of x,y tuples.
    """

    # Open the file and build a dict of loads from the contents.
    #
    load_dict: dict[int, LocationSegment] = {}

    def parse_loc(loc_str: str) -> Location:
        loc = tuple(float(x) for x in loc_str[1:-1].split(','))

        return loc

    with open(path, newline='', encoding="utf-8") as fi:
        reader = csv.DictReader(fi, delimiter=' ')

        for row in reader:
            load_number = int(row['loadNumber'])
            pick_up = parse_loc(row['pickup'])
            drop_off = parse_loc(row['dropoff'])

            load_dict[load_number] = (pick_up, drop_off)

    return load_dict


def get_location_list(load_dict: dict[int, LocationSegment]) -> list[Location]:
    """
    Build a list of the unique locations in a load dict.

    The home base location (0, 0) will be included and the list will be sorted by distance from home
    base.

    Parameters
    ----------
    load_dict : dict[int, Segment]
        A load dict with pick up and drop off locations.

    Returns
    -------
    list[Location]
        A list of unique locations collected from the load dict, along with the home base location
        as the first element. The elements will be sorted by distance from home base.
    """

    # Create a set of locations taken from the load pick up and drop off locations. Add the home
    # base location to the set.
    #
    location_set: set[Location] = set()

    location_set.add((0, 0))

    for pick_up, drop_off in load_dict.values():
        location_set.add(pick_up)
        location_set.add(drop_off)

    # Convert the set to a list and sort it according to the distance from the home base location.
    #
    location_list = list(location_set)

    def sorter(loc: Location) -> float:
        return np.sqrt(loc[0]**2 + loc[1]**2)

    location_list.sort(key=sorter)

    # Return the list.
    #
    return location_list


def get_loads_by_location_index(load_dict: dict[int, LocationSegment],
                                location_list: list[Location]) -> dict[int, IndexSegment]:
    """
    Build a dict keyed by load number where each element is a pair of ints representing the index in
    location_list of the pick up and drop off locations for the load.

    Parameters
    ----------
    load_dict : dict[int, Segment]
        A dict of x,y pick up and drop off location tuple pairs keyed by load number.
    location_list : list[Location]
        A list of x,y pick up and drop off location tuple pairs.

    Returns
    -------
    dict[int, IndexSegment]
        The pick up and drop off location indexes for each load.
    """

    # Iterate through load_dict and build a new dict where the x,y pick up and drop off locations
    # are replaced by the indexes of the locations in location_list.
    #
    load_index_dict: dict[int, IndexSegment] = {}

    for load_num, (pick_up, drop_off) in load_dict.items():
        pick_up_index = location_list.index(pick_up)
        drop_off_index = location_list.index(drop_off)

        load_index_dict[load_num] = (pick_up_index, drop_off_index)

    # Return the new dict.
    #
    return load_index_dict


def get_distances_array(location_list: list[Location]) -> Array2D:
    """
    Create a 2D array filled with the distances between each pair of locations in the locations
    list.

    Parameters
    ----------
    location_list : list[Location]
        A list of x,y location tuples.

    Returns
    -------
    Array2D
        A square 2D array of shape (locations, locations) where the value a[i, j] is the distance
        between the ith location and the jth location.
    """

    # Build an Nx2 array containing the locations.
    #
    locations = np.array(location_list)

    # Calculate an Nx1 array containing elements (x[i]**2 + y[i]**2)
    #
    sq_lens = (locations * locations).sum(axis=1).reshape(1, -1)

    # Calculate an NxN array containing the cross terms -2 * (x[i] * x[j] + y[i] * y[j]).
    #
    crosses = -2.0 * np.matmul(locations, locations.T)

    # Calculate the squares of the distances. Clip the values at zero.
    #
    distances = sq_lens + crosses
    distances = distances + sq_lens.T
    distances = distances.clip(0.0)

    # Force the diagonal values to zero.
    #
    distances[np.diag_indices_from(distances)] = 0.0

    # Take the square roots and return the distances array.
    #
    distances = np.sqrt(distances)

    return distances


def cost_of_total_route(route: list[IndexSegment], distances: Array2D) -> float:
    """
    Get the total cost for the specified route between load segments.

    Parameters
    ----------
    route : list[IndexSegment]
        A route specifying load segments in the order they are visited.
    distances : Array2D
        A square 2D array of distances between each pair of locations.

    Returns
    -------
    float
        The total cost for the route.
    """

    # Make a function that determines whether or not to continue to build the route.
    #
    def at_max(start: int, segment: IndexSegment, current_cost: float) -> bool:
        # The route is at the maximum allowed length if the cost to move through the next load
        # segment and go home is more than the max cost. Return True if the route is at maximum
        # length.
        #
        current_cost += distances[start, segment[0]] + distances[segment] + distances[segment[1], 0]

        return MAX_LOOP_COST < current_cost

    # Start at home (0) and work through the load segments, adding the base cost each time a route
    # exceeds MAX_LOOP_COST and a new loop is started.
    #
    total_cost = 0
    loop_cost = BASE_COST
    current_loc = 0

    for load in route:
        # If the load segment would exceed the maximum route loop cost, finalize the loop cost, add
        # it to the total cost, and reset the current location and loop cost.
        #
        if at_max(current_loc, load, loop_cost):
            total_cost += loop_cost + distances[current_loc, 0]
            current_loc = 0
            loop_cost = BASE_COST

        # Add the cost to travel from the current location to the end of the load to the loop cost
        # and update the current location to the end of the load.
        #
        loop_cost += distances[current_loc, load[0]] + distances[load]
        current_loc = load[1]

    # Add the last loop cost to the total cost, adding a last trip home, and return the total cost.
    #
    total_cost += loop_cost + distances[current_loc, 0]

    return total_cost


def break_route_into_loops(route: list[IndexSegment],
                           distances: Array2D) -> list[list[IndexSegment]]:
    """
    Break the route up into individual loops.

    Parameters
    ----------
    route : list[IndexSegment]
        A route specifying load segments in the order they are visited.
    distances : Array2D
        A square 2D array of distances between each pair of locations.

    Returns
    -------
    list[list[IndexSegment]]
       The loops for the route.
    """

    # Make a function that determines whether or not to continue to build the route.
    #
    def at_max(start: int, segment: IndexSegment, current_cost: float) -> bool:
        # The route is at the maximum allowed length if the cost to move through the next load
        # segment and go home is more than the max cost. Return True if the route is at maximum
        # length.
        #
        current_cost += distances[start, segment[0]] + distances[segment] + distances[segment[1], 0]

        return MAX_LOOP_COST < current_cost

    # Start at home (0) and work through the load segments, starting a new loop each time a route
    # exceeds MAX_LOOP_COST and a new loop is started.
    #
    loops = []
    loop = []
    loop_cost = BASE_COST
    current_loc = 0

    for load in route:
        # If the load segment would exceed the maximum route loop cost, add the loop to the list of
        # loops, start a new loop, and reset the current location and loop cost.
        #
        if at_max(current_loc, load, loop_cost):
            loops.append(loop)

            loop = []
            current_loc = 0
            loop_cost = BASE_COST

        # Add the load to the loop, add the cost to travel from the current location to the end of
        # the load to the loop cost, and update the current location to the end of the load.
        #
        loop.append(load)
        loop_cost += distances[current_loc, load[0]] + distances[load]
        current_loc = load[1]

    # Add the last loop to the list of loops and return the list.
    #
    loops.append(loop)

    return loops


def create_population(population_size: int, loads: list[IndexSegment]) -> list[list[IndexSegment]]:
    """
    Create a population of randomly selected routes.

    Parameters
    ----------
    population_size : int
        The number of routes to create.
    loads : list[IndexSegment]
        A list of load segments to use in creating the routes.

    Returns
    -------
    list[list[IndexSegment]]
        A list of randomly ordered routes.
    """

    # Make a copy of the loads segments list so the original is unaltered.
    #
    loads = loads.copy()

    # Create the routes by shuffling the load segments array.
    # Make sure the shuffled routes are unique.
    #
    routes: list[list[IndexSegment]] = []

    while population_size > len(routes):
        np.random.shuffle(loads)

        for route in routes:
            if loads == route:
                continue

        # Append a copy of the shuffled loads to the routes list. (Otherwise you get the same list
        # multiple times!)
        #
        routes.append(loads.copy())

    # Return the list of shuffled routes.
    #
    return routes


def mutate_population(population: list[list[IndexSegment]],
                      new_size: int) -> list[list[IndexSegment]]:
    """
    Grow the population of routes with mutant routes created by swapping a random pair of load
    segments in an existing route.

    Only unique mutations are returned.

    Parameters
    ----------
    population : list[list[IndexSegment]]
        A list of routes. Each route is a list of load segments.
    new_size : int
        The size to grow the population to.

    Returns
    -------
    list[list[IndexSegment]]
        The new population with mutated routes.
    """

    # Create mutations and add them to the population.
    #
    old_size = len(population)

    while len(population) < new_size:
        # Pick a random route from the old population.
        #
        route = population[np.random.choice(old_size)]

        # Keep trying until a unique mutation is found.
        #
        while True:
            # Make a copy of the route.
            #
            new_route = route.copy()

            # Pick the load segments to swap.
            #
            swaps = np.random.choice(len(route), 2, replace=False)

            # Swap the load segments to make the mutated route.
            #
            load = new_route[swaps[0]]
            new_route[swaps[0]] = new_route[swaps[1]]
            new_route[swaps[1]] = load

            # Make sure the mutation is unique.
            #
            for test_route in population:
                if new_route == test_route:
                    continue

            for test_route in population:
                if new_route == test_route:
                    continue

            # Add the mutated route to population and move to the next route.
            #
            population.append(new_route)

            break

    # Return the expanded population.
    #
    return population


def score_population(population: list[list[IndexSegment]], distances: Array2D) -> list[float]:
    """
    Score the routes in the population.

    Parameters
    ----------
    population : list[list[IndexSegment]]
        A list of routes.
    distances : Array2D
        A square 2D array of distances between each pair of locations.

    Returns
    -------
    list[float]
        The score for each route in the population.
    """

    # Collect the scores for the routes in the population and return them.
    #
    scores: list[float] = []

    for route in population:
        scores.append(cost_of_total_route(route, distances))

    return scores


def find_best_route(loads: list[IndexSegment], distances: Array2D,
                    generations: int, population_size: int, culling_size: int,
                    convergence: float) -> list[IndexSegment]:
    """
    Find the best route through the loads.

    Parameters
    ----------
    loads : list[IndexSegment]
        A list of load segments describing a route loop.
    distances : Array2D
        A square 2D array of distances between each pair of locations.
    generations : int
        The maximum number of generations to run.
    population_size : int
        The size to grow the population to in each generation.
    culling_size : int
        The size to cull the population to in each generation.
    convergence : float
        The fractional change in the best score that will be considered convergent.

    Returns
    -------
    list[IndexSegment]
        The best route.
    """

    # Get the starting time.
    #
    start_time = time.perf_counter()

    # Create an initial population of routes.
    #
    population = create_population(population_size, loads)

    # Run through the generations keeping only the best population_size routes each time.
    #
    last_score = BIG_NUMBER

    if DEBUG:
        print("Generations =", generations, "Population size =", population_size)

    stop_count = 0

    for generation in range(0, generations):
        # Score the population.
        #
        scores = score_population(population, distances)

        # Keep the best of the population.
        #
        indexes = np.argsort(scores)[:culling_size]

        population = [population[i] for i in indexes]

        # Get the best score and the fractional change.
        #
        best_score = scores[indexes[0]]
        change = np.abs(best_score - last_score) / last_score

        if DEBUG:
            print(f"gen {generation} change {change:.4e}", end="\r", flush=True)

        # Break out if the fractional change has been less than tolerance for STOP_COUNT generations
        # or the execution time has reached MAX_TIME seconds.
        #
        elapsed_time = time.perf_counter() - start_time

        if MAX_TIME < elapsed_time:
            break

        if change < convergence:
            stop_count += 1

        else:
            stop_count = 0

        if stop_count == STOP_COUNT:
            break

        # Update last_score.
        #
        last_score = best_score

        # Grow the population through mutation.
        #
        mutate_population(population, population_size)

    # Print the stopping generation.
    #
    if DEBUG:
        print("")
        print("Stopping generation =", generation)
        print("Elapsed time =", elapsed_time)

    # Return the route from the final population with the best score.
    #
    return population[0]


def main(path: str):
    """
    Find a best route through the loads specified in file at path.

    Parameters
    ----------
    path : str
        The path to the loads file.
    """

    # Get the starting time.
    #
    elapsed = time.perf_counter()

    # Read in the loads from the load file. Get a list of the unique locations and create a square
    # 2D array that contains the distances between each pair of locations. Get the loads by location
    # index.
    #
    load_location_dict = read_load_file(path)
    location_list = get_location_list(load_location_dict)
    distances = get_distances_array(location_list)
    load_index_dict = get_loads_by_location_index(load_location_dict, location_list)

    # Get a list of the load numbers and a list of the load segments.
    #
    load_numbers = list(load_index_dict.keys())
    loads = list(load_index_dict.values())

    # Find the route with the best score.
    #
    best_route = find_best_route(loads, distances, GENERATIONS, POPULATION_SIZE, CULLING_SIZE,
                                 CONVERGENCE)

    # Get the ending time and print the elapsed time.
    #
    elapsed = time.perf_counter() - elapsed

    if DEBUG:
        print(f"Elapsed time (sec) ={elapsed:.3f}")

    # Print out the score.
    #
    if DEBUG:
        print("Cost =", cost_of_total_route(best_route, distances))

    # Break the route into loops.
    #
    best_set = break_route_into_loops(best_route, distances)

    # For each loop in the best route, build a list of the load numbers in the loop and write them
    # to standard out.
    #
    for loop in best_set:
        numbers = [load_numbers[loads.index(load)] for load in loop]

        print(numbers)


if '__main__' == __name__:
    main(sys.argv[1])
