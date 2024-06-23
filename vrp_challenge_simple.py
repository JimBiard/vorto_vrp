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

import numpy as np
import numpy.testing


# Type hint definitions.
#
Location = tuple[float, float]
LocationSegment = tuple[Location, Location]
IndexSegment = tuple[int, int]

Array2D = np.ndarray[tuple[int, int], float]


# Constants of the problem.
#
BIG_NUMBER = 1.0e100
BASE_COST = 500.0  # The cost to begin a new route loop.
MAX_LOOP_COST = 500.0 + 12 * 60.0  # The exclusive upper bound on the cost of a route loop.


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


def cost_of_route_loop(route: list[IndexSegment], distances: Array2D) -> float:
    """
    Calculate the total cost of a route.

    Parameters
    ----------
    route : list[int]
        A list of load segments describing a route loop.
    distances : Array2D
        A square 2D array of distances between each pair of locations.

    Returns
    -------
    float
        The total cost to drive the route loop.
    """

    # Return 0.0 if the route is empty.
    #
    if not route:
        return 0.0

    # Construct a full route loop index list that starts at home (0) and ends at home.
    #
    route_list = [0]

    for load_segment in route:
        route_list.extend(load_segment)

    route_list.append(0)

    # Get arrays of start locations and end locations for each route segment.
    #
    starts = np.array(route_list[:-1], dtype=int)
    ends = np.array(route_list[1:], dtype=int)

    # Collect the total cost to drive the route loop, which is the base cost plus the sum of the
    # distances.
    #
    cost = distances[(starts, ends)].sum()

    cost += BASE_COST

    # Return the route loop cost.
    #
    return cost


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


def build_route_loops(starting_load: IndexSegment, loads: list[IndexSegment],
                      distances: Array2D) -> list[tuple[float, list[IndexSegment]]]:
    """
    Build a set of route loops that visit all load segments, starting with the specified load
    segment.

    Each loop is constrained to have a cost of less than MAX_LOOP_COST.

    Parameters
    ----------
    starting_load: IndexSegment
        The load segment to start with.
    loads : list[IndexSegment]
        A list containing the route segments to use.
    distances : Array2D
        A square 2D array of distances between each pair of locations.

    Returns
    -------
    list[tuple[float, list[IndexSegment]]]
        A list of route loops with their costs.
    """

    # Make a copy of the loads list so the original is not altered.
    # Find the index of the specified load.
    #
    available_loads = loads.copy()
    load_index = loads.index(starting_load)

    # Get a list of the start indexes for the load segments.
    #
    pick_ups = [x[0] for x in loads]

    # Make a function that determines whether or not to continue to build the route.
    #
    def at_max(start: int, segment: IndexSegment, current_cost: float) -> bool:
        # The route is at the maximum allowed length if the cost to move through the next load
        # segment and go home is more than the max cost. Return True if the route is at maximum
        # length.
        #
        current_cost += distances[start, segment[0]] + distances[segment] + distances[segment[1], 0]

        return MAX_LOOP_COST < current_cost

    # Initialize a loop route list, a cost, the initial location (home), and a list for holding
    # costs and loops.
    #
    loop = []
    cost = BASE_COST
    current_loc = 0

    loops_list: list[tuple[float, list[IndexSegment]]] = []

    # Iterate over the remaining load segments and build route loops.
    #
    while available_loads:
        # Pop a new load segment from available loads and delete the load start index from pick_ups.
        #
        load = available_loads.pop(load_index)
        pick_ups.pop(load_index)

        # If the cost of adding the new load segment is too great, finish the loop and start a new
        # one.
        #
        if at_max(current_loc, load, cost):
            # Update the cost to include a home segment.
            #
            cost += distances[current_loc, 0]

            # Store the final cost and loop into the loops_list.
            #
            loops_list.append((cost, loop))

            # Start a new loop based on the current load segment.
            #
            current_loc = 0
            loop = []
            cost = BASE_COST

        # Add the load segment to the loop and update the cost and current
        # location.
        #
        loop.append(load)

        cost += distances[current_loc, load[0]] + distances[load]
        current_loc = load[1]

        # Break out if there are no more loads to process.
        #
        if not available_loads:
            break

        # Find the index to the next load segment to use. It is the one that has the shortest
        # distance from the current location.
        #
        row = distances[current_loc]
        sub_row = row[pick_ups]
        load_index = sub_row.argmin()

    # Finalize the last loop cost and store it and the cost into the loops list.
    #
    cost += distances[current_loc, 0]

    loops_list.append((cost, loop))

    # Verify that all loads were used. Raise an exception if one or more loads are missing or if a
    # load appears more than once.
    #
    test_loads = []

    for entry in loops_list:
        test_loads.extend(entry[1])

    if set(test_loads) != set(loads):
        raise RuntimeError(f"build_route_loops({starting_load}): One or more loads are missing.")

    if len(test_loads) > len(loads):
        raise RuntimeError(f"build_route_loops({starting_load}): One or more loads are repeated.")

    # Verify the costs. Raise an exception if a cost is wrong.
    #
    for entry in loops_list:
        test_cost = cost_of_route_loop(entry[1], distances)

        numpy.testing.assert_allclose(entry[0], test_cost)

    # Return the list of loops.
    #
    return loops_list


def main(path: str):
    """
    Find a best route through the loads specified in file at path.

    Parameters
    ----------
    path : str
        The path to the loads file.
    """

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

    # Starting with each load segment in turn, build a set of routes based on shortest distances
    # between load segments.
    #
    loop_sets = []

    for load in loads:
        loop_sets.append(build_route_loops(load, loads, distances))

    # Find the route with the best score.
    #
    best_cost = BIG_NUMBER
    best_set = []

    for loops_list in loop_sets:
        total_cost = np.sum([v[0] for v in loops_list])

        if total_cost < best_cost:
            best_cost = total_cost
            best_set = loops_list

    # Print the total cost.
    #
    # print("Cost", total_cost)

    # For each loop in the best route, build a list of the load numbers in the loop and write them
    # to standard out.
    #
    for _, loop in best_set:
        numbers = [load_numbers[loads.index(load)] for load in loop]

        print(numbers)


if '__main__' == __name__:
    main(sys.argv[1])
