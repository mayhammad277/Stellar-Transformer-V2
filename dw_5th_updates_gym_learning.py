# -*- coding: utf-8 -*-





import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

from scipy.spatial import distance
import random
import datetime
from datetime import timezone
from datetime import datetime


import numpy as np
import random

import matplotlib.pyplot as plt
import math
import gym
import numpy as np
from gym import spaces
from stable_baselines3 import A2C



"""helper functions outside class

def a_star_routing( source, target, satellite_positions):
      import heapq

      # priority q
      open_set = []
      heapq.heappush(open_set, (0, source))  # (f_cost, current_node)

      #  costs
      g_cost = {node: float('inf') for node in range(len(satellite_positions))}
      g_cost[source] = 0

      #  paths and visited nodes
      came_from = {}
      visited_sats = set()


      def heuristic_cost_estimate(current, goal, satellite_positions, method="delay"):

         if method == "delay":
            distance = np.linalg.norm(np.array(satellite_positions[current]) - np.array(satellite_positions[goal]))
            return distance / 299792458



      while open_set:
        idx, current = heapq.heappop(open_set)
        print("current",current,idx)
        visited_sats.add(current)

        # Check if the target is reached
        if current == target:
            path = reconstruct_path(came_from, current)
            distance = sum(
                edge_cost(path[i], path[i + 1], satellite_positions)
                for i in range(len(path) - 1)
            )
            return distance, list(visited_sats), path
        print("satellite_positions",satellite_positions)
        # Explore neighbors
        for neighbor in get_neighbors(current, satellite_positions):
            tentative_g_cost = g_cost[current] + edge_cost(current, neighbor, satellite_positions)

            if tentative_g_cost < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g_cost
                f_cost = tentative_g_cost + heuristic_cost_estimate(neighbor, target)
                heapq.heappush(open_set, (f_cost, neighbor))
                came_from[neighbor] = current

      return float('inf'), list(visited_sats), []  # No valid path found

def reconstruct_path( came_from, current):
      path = [current]
      while current in came_from:
        current = came_from[current]
        path.append(current)
        path.reverse()
      return path

def edge_cost( node1, node2, satellite_positions):
      print("node1",node1)
      x1, y1, z1 = satellite_positions[node1]
      x2, y2, z2 = satellite_positions[node2]
      return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)  # Euclidean distance

def get_neighbors(node, satellite_positions):
    neighbors = []
    node_position = satellite_positions[node]

    for other_node, other_position in satellite_positions.items():
        if node != other_node:  # Avoid self-loops
            if satellite_coverage_check(node_position, other_position):
                neighbors.append(other_node)

    print(f"Satellite {node} has {len(neighbors)} neighbors.")  # Debugging info
    return neighbors
  ""

def a_star_routing(source, target, satellite_positions):
    import heapq
    import numpy as np

    #  reverse lookup
    position_to_index = {k:tuple(v) for k, v in satellite_positions.items()}
    print("position_to_index",position_to_index)
    # p queue
    open_set = []
    heapq.heappush(open_set, (0, source))  # (f_cost, current_node)
    print("open_set",open_set)
    # cost
    g_cost = {pos: float('inf') for pos in satellite_positions.values()}
    g_cost[source] = 0

    # pths and visited nodes
    came_from = {}
    visited_sats = set()

    def heuristic_cost_estimate(current, goal, method="delay"):
        #
        if method == "delay":
            distance = np.linalg.norm(np.array(current) - np.array(goal))
            return distance / 299792458

    def edge_cost(current, neighbor):

        distance = np.linalg.norm(np.array(current) - np.array(neighbor))
        return distance / 299792458



    def get_neighbors(current, satellite_positions, max_range=1000):
      neighbors = []
      current_pos = np.array(satellite_positions[current])
      for node, pos in satellite_positions.items():
        if node != current and np.linalg.norm(current_pos - pos) <= max_range:
            neighbors.append(node)
      return neighbors

    def reconstruct_path(came_from, current):

        path = [current]
        while current in came_from:
            current = came_from[current]
            path.insert(0, current)
        return path

    while open_set:
        _, current = heapq.heappop(open_set)
        visited_sats.add(tuple(current))

        # chck if the target is reached
        if np.array_equal(current, target):
            path = reconstruct_path(came_from, tuple(current))
            distance = sum(
                edge_cost(path[i], path[i + 1])
                for i in range(len(path) - 1)
            )
            return distance, list(visited_sats), path

        # explore neighbors
        for neighbor in get_neighbors(current, satellite_positions):
            if tuple(neighbor) in visited_sats:
                continue
            print("neighbor",neighbor)
            tentative_g_cost = g_cost[tuple(current)] + edge_cost(current, neighbor)

            if tentative_g_cost < g_cost[tuple(neighbor)]:
                g_cost[tuple(neighbor)] = tentative_g_cost
                f_cost = tentative_g_cost + heuristic_cost_estimate(neighbor, target)
                heapq.heappush(open_set, (f_cost, neighbor))
                came_from[tuple(neighbor)] = tuple(current)

    return float('inf'), list(visited_sats), []  # no  path found.
    "

Helper Functions ----------------------------------------------------------
"""

import numpy as np
from datetime import datetime, timezone
def is_on_earth(position, epsilon=10):
    R_E = 6371  #
    distance_from_center = np.linalg.norm(position)
    return abs(distance_from_center - R_E) <= epsilon
def julian_date(utc_time):
    """Convert UTC time to Julian Date."""
    unix_epoch = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    delta_seconds = (utc_time - unix_epoch).total_seconds()
    julian_date = 2440587.5 + delta_seconds / 86400.0
    return julian_date

def gst_from_julian(julian_date):
    """Compute Greenwich Sidereal Time from Julian Date."""
    T = (julian_date - 2451545.0) / 36525.0
    # Compute GST in degrees, modulo 360 to stay within bounds
    GST = 280.46061837 + 360.98564736629 * (julian_date - 2451545.0) + 0.000387933 * T**2 - (T**3 / 38710000.0)
    GST = GST % 360.0  # Ensure GST is within [0, 360] degrees
    return np.radians(GST)



def geodetic_to_eci(lat_deg, lon_deg, altitude_km=0, utc_time=None):

      if utc_time is None:
        utc_time = datetime.utcnow().replace(tzinfo=timezone.utc)

      lat = np.radians(lat_deg)
      lon = np.radians(lon_deg)

      R_E = 6378.137

    # Compute the Julian Date from the given UTC time
      jd = julian_date(utc_time)

      gst = gst_from_julian(jd)

      lon_eci = lon + gst  # Longitude in the inertial frame (ECI)

      x = (R_E + altitude_km) * np.cos(lat) * np.cos(lon_eci)
      y = (R_E + altitude_km) * np.cos(lat) * np.sin(lon_eci)
      z = (R_E + altitude_km) * np.sin(lat)

      return x, y, z



def is_on_earth(position, epsilon=1e-3):#to check wether the current node is a sallite or a ground point while routing
    R_E = 6371
    distance_from_center = np.linalg.norm(position)
    return abs(distance_from_center - R_E) >= epsilon

import numpy as np
from scipy.spatial import distance

R_E = 6371
P_transmitted = 50
N_noise = 1e-9
altitude=550
def line_of_sight(satellite_pos, ground_pos, R_E=6371):# ground to sattlite

      satellite_pos = np.array(satellite_pos)
      ground_pos = np.array(ground_pos)

      ground_to_sat = satellite_pos - ground_pos

      ground_distance = np.linalg.norm(ground_pos)

      angle = np.arccos(np.dot(ground_to_sat, ground_pos) / (np.linalg.norm(ground_to_sat) * ground_distance))


      if angle < np.pi / 2:
        return True  # sat is in LoS
      else:
        return False  # No LoS
def satellite_coverage_check(r1, r2): #between sattlites in space with input ECI postions
      Re=6371

      r1 = np.array(r1) #  1st sat (x1, y1, z1)
      r2 = np.array(r2) #pos of the 2 sat (x1, y1, z1)

      # get the Euc dist
      d = np.linalg.norm(r1 - r2)

      # get the min dist-> earth & line (connecting the sats)
      d_min = np.linalg.norm(np.cross(r1, r2)) / d

      # chk if the min dist > earth rad
      return d_min > Re



def satellite_coverage_check(r1, r2, Re=6371, threshold=500):
        r1, r2 = np.array(r1), np.array(r2)
        d = np.linalg.norm(r1 - r2)
        d_min = np.linalg.norm(np.cross(r1, r2)) / d
        return d_min > Re - threshold  # relxing the constraint

# find route  (simplified)
def find_route(start_ground_pos, end_ground_pos, satellites):

    d_tot = 0
    num_hops = 0
    current_node = start_ground_pos
    route = [start_ground_pos]  # Keep track of the route
    visited_nodes = set()

    while True:
        # ck if we can directly reac to the dest
        if line_of_sight(current_node, end_ground_pos):
            d_tot += distance.euclidean(current_node, end_ground_pos)
            route.append(end_ground_pos)
            break

        #  sats with los to the current
        los_satellites = [
            s for s in satellites if line_of_sight(current_node, s) and tuple(s) not in visited_nodes
        ]

        if not los_satellites:
            raise ValueError("No route found: No satellites in line of sight.")

        #   sats closest to  dest
        next_node = min(los_satellites, key=lambda s: distance.euclidean(s, end_ground_pos))

        # updt total distance,
        d_tot += distance.euclidean(current_node, next_node)
        num_hops += 1
        route.append(next_node)
        visited_nodes.add(tuple(next_node))

        #
        current_node = next_node

    return d_tot, num_hops, route

def is_on_earth(position, epsilon=1e-3):#to check wether the current node is a sallite or a ground point while routing
    R_E = 6371
    distance_from_center = np.linalg.norm(position)
    return abs(distance_from_center - R_E) >= epsilon

# Coverage score calculation
def calculate_coverage_score(A, satellites, num_test_points=100):

    successful_connections = 0
    test_points = generate_test_points(num_test_points)

    for test_point in test_points:
        #
        B = geodetic_to_eci(*test_point, 550)


        d_tot, num_hops=find_route(A, B, satellites)
        if(num_hops>0):

          successful_connections += 1  # route found, success cnt+1
        else :
          pass

    #  coverage score ->fraction of succs connections
    coverage_score = successful_connections / num_test_points
    return coverage_score

# Gs positions (lat, long)
berlin_lat, berlin_lon = 52.52, 13.405
cape_town_lat, cape_town_lon = -33.9249, 18.4241




#  random sat pos
num_satellites = 3
satellite_positions = [
    geodetic_to_eci(np.random.uniform(-90, 90), np.random.uniform(-180, 180), altitude)
    for _ in range(num_satellites)
]

#
berlin_pos = geodetic_to_eci(berlin_lat, berlin_lon, 0)
cape_town_pos = geodetic_to_eci(cape_town_lat, cape_town_lon, 0)

try:
    # find the route
    total_distance, hops, route = find_route(berlin_pos, cape_town_pos, satellite_positions)
    print(f"Total Distance: {total_distance:.2f} km")
    print(f"Number of Hops: {hops}")
    print("Route:")
    for node in route:
        print(node)
except ValueError as e:
    print(str(e))

"""
import heapq
import numpy as np

def a_star_routing(source, target, satellite_positions):

    # reverse lookup dict
    position_to_index = {tuple(v): k for k, v in satellite_positions.items()}

    if tuple(source) not in position_to_index or tuple(target) not in position_to_index:
        raise ValueError("src or tgt position not  in satellite_positions.")

    open_set = []
    heapq.heappush(open_set, (0, source))  # (f_cost, current_node)

    g_cost = {tuple(pos): float('inf') for pos in satellite_positions.values()}
    g_cost[tuple(source)] = 0

    came_from = {}
    visited_sats = set()


    def heuristic_cost_estimate(current, goal, method="delay"):
        if method == "delay":
            distance = np.linalg.norm(np.array(current) - np.array(goal))
            return distance #/ 299792458

    def edge_cost(current, neighbor):
        distance = np.linalg.norm(np.array(current) - np.array(neighbor))
        return distance #/ 299792458

    def get_neighbors(current, satellite_positions, max_range=2000):
        neighbors = []
        current_pos = np.array(current)
        for pos in satellite_positions.values():
            #if tuple(pos) != tuple(current) and np.linalg.norm(current_pos - np.array(pos)) <= max_range:
            #changed to coverage check ----

            if satellite_coverage_check(current, pos):
                neighbors.append(tuple(pos))

        return neighbors

    def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.insert(0, current)
        return [position_to_index[tuple(p)] for p in path]  # convert positions to indices

    while open_set:
        _, current = heapq.heappop(open_set)
        visited_sats.add(position_to_index[tuple(current)])

        if np.array_equal(current, target):
            path = reconstruct_path(came_from, tuple(current))
            distance = sum(
                edge_cost(path[i], path[i + 1])
                for i in range(len(path) - 1)
            )
            return distance, list(visited_sats), path

        for neighbor in get_neighbors(current, satellite_positions):
            if tuple(neighbor) in visited_sats:
                continue
            tentative_g_cost = g_cost[tuple(current)] + edge_cost(current, neighbor)

            if tentative_g_cost < g_cost[tuple(neighbor)]:
                g_cost[tuple(neighbor)] = tentative_g_cost
                f_cost = tentative_g_cost + heuristic_cost_estimate(neighbor, target)
                heapq.heappush(open_set, (f_cost, neighbor))
                came_from[tuple(neighbor)] = tuple(current)

    return float('inf'), list(visited_sats), []  # no valid path
"""
"""
import heapq
import numpy as np
from math import radians, sin, cos, sqrt, atan2

def great_circle_distance(pos1, pos2, R_E=6371):

     #the greatcircle dist
    lat1 = np.arcsin(pos1[2] / np.linalg.norm(pos1))
    lon1 = np.arctan2(pos1[1], pos1[0])
    lat2 = np.arcsin(pos2[2] / np.linalg.norm(pos2))
    lon2 = np.arctan2(pos2[1], pos2[0])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R_E * c

def a_star_routing(source, target, satellite_positions):

    # reverse lookup
    position_to_index = {tuple(v): k for k, v in satellite_positions.items()}

    if tuple(source) not in position_to_index or tuple(target) not in position_to_index:
        raise ValueError("src or tgt not in sats.")

    open_set = []
    heapq.heappush(open_set, (0, source))  # (f_cost, current_node)

    g_cost = {tuple(pos): float('inf') for pos in satellite_positions.values()}
    g_cost[tuple(source)] = 0

    came_from = {}
    visited_sats = set()

    def edge_cost(current, neighbor):

        return great_circle_distance(np.array(current), np.array(neighbor))

    def heuristic_cost_estimate(current, goal, method="delay"):

        if method == "delay":
            return great_circle_distance(np.array(current), np.array(goal))

    def get_neighbors(current, satellite_positions, max_range=2000):

        neighbors = []
        current_pos = np.array(current)
        for pos in satellite_positions.values():
            if satellite_coverage_check(current, pos):
                neighbors.append(tuple(pos))
        return neighbors

    def reconstruct_path(came_from, current):

        path = [current]
        while current in came_from:
            current = came_from[current]
            path.insert(0, current)
        return [position_to_index[tuple(p)] for p in path], sum(
            edge_cost(path[i], path[i + 1])
            for i in range(len(path) - 1)
        )

    while open_set:
        _, current = heapq.heappop(open_set)
        visited_sats.add(position_to_index[tuple(current)])

        if np.array_equal(current, target):
            path_indices, distance = reconstruct_path(came_from, tuple(current))
            return distance, list(visited_sats), path_indices

        for neighbor in get_neighbors(current, satellite_positions):
            if tuple(neighbor) in visited_sats:
                continue
            tentative_g_cost = g_cost[tuple(current)] + edge_cost(current, neighbor)

            if tentative_g_cost < g_cost[tuple(neighbor)]:
                g_cost[tuple(neighbor)] = tentative_g_cost
                f_cost = tentative_g_cost + heuristic_cost_estimate(neighbor, target)
                heapq.heappush(open_set, (f_cost, neighbor))
                came_from[tuple(neighbor)] = tuple(current)

    return float('inf'), list(visited_sats), []  # no valid path

    """

import heapq
import numpy as np
from math import radians, sin, cos, sqrt, atan2

def great_circle_distance(pos1, pos2, R_E=6371):

     #the greatcircle dist
    lat1 = np.arcsin(pos1[2] / np.linalg.norm(pos1))
    lon1 = np.arctan2(pos1[1], pos1[0])
    lat2 = np.arcsin(pos2[2] / np.linalg.norm(pos2))
    lon2 = np.arctan2(pos2[1], pos2[0])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R_E * c
def a_star_routing(source, target, satellite_positions, R_E=6371):

    # reverse lookup
    position_to_index = {tuple(v): k for k, v in satellite_positions.items()}

    if tuple(source) not in position_to_index or tuple(target) not in position_to_index:
        raise ValueError("src or tgt position not in satellite_positions.")

    open_set = []
    heapq.heappush(open_set, (0, source))  # (f_cost, current_node)

    g_cost = {tuple(pos): float('inf') for pos in satellite_positions.values()}
    g_cost[tuple(source)] = 0

    came_from = {}
    visited_sats = set()

    def edge_cost(current, neighbor):

        return great_circle_distance(np.array(current), np.array(neighbor))

    def heuristic_cost_estimate(current, goal, method="delay"):

        if method == "delay":
            return great_circle_distance(np.array(current), np.array(goal))

    def get_neighbors(current, satellite_positions, max_range=2000):

        neighbors = []
        current_pos = np.array(current)
        for pos in satellite_positions.values():
            if satellite_coverage_check(current, pos):
                neighbors.append(tuple(pos))
        return neighbors
    def reconstruct_path(came_from, current):

        path = [current]
        while current in came_from:
            current = came_from[current]
            path.insert(0, current)

        hop_distances = [edge_cost(path[i], path[i + 1]) for i in range(len(path) - 1)]
        total_distance = sum(hop_distances)

        print("hop dists:", hop_distances)
        return [position_to_index[tuple(p)] for p in path], total_distance

    while open_set:
        _, current = heapq.heappop(open_set)
        visited_sats.add(position_to_index[tuple(current)])

        if np.array_equal(current, target):
            path_indices, distance = reconstruct_path(came_from, tuple(current))
            if distance > R_E:
                print(" total  dist is  > re : the path involves multiple hops.")
            return distance, list(visited_sats), path_indices

        for neighbor in get_neighbors(current, satellite_positions):
            if tuple(neighbor) in visited_sats:
                continue
            tentative_g_cost = g_cost[tuple(current)] + edge_cost(current, neighbor)

            if tentative_g_cost < g_cost[tuple(neighbor)]:
                g_cost[tuple(neighbor)] = tentative_g_cost
                f_cost = tentative_g_cost + heuristic_cost_estimate(neighbor, target)
                heapq.heappush(open_set, (f_cost, neighbor))
                came_from[tuple(neighbor)] = tuple(current)

    return float('inf'), list(visited_sats), []  # no valid path

import numpy as np


Re = 6371
altitude = 550  #

berlin_lat, berlin_lon = 52.52, 13.405
cape_town_lat, cape_town_lon = -33.9249, 18.4241



def line_of_sight(satellite_pos, ground_pos, R_E=6371.8):
    satellite_pos = np.array(satellite_pos)


    ground_pos = np.array(ground_pos)

    ground_to_sat = satellite_pos - ground_pos


    ground_distance = np.linalg.norm(ground_pos)

    angle = np.arccos(np.dot(ground_to_sat, ground_pos) / (np.linalg.norm(ground_to_sat) * ground_distance))

    #print("angle",angle)
    if angle < np.pi / 2:
        return True  # sat is in LoS
    else:
        return False  # No LoS
import numpy as np
def line_of_sight(point1, point2, R_E=6371.8):

    point1 = np.array(point1)
    point2 = np.array(point2)

    point1_to_point2 = point2 - point1

    distance = np.linalg.norm(point1_to_point2)

    point1_distance = np.linalg.norm(point1)
    point2_distance = np.linalg.norm(point2)

    angle = np.arccos(np.dot(point1_to_point2, point1) / (distance * point1_distance))

    if angle < np.pi / 2 and distance > np.sqrt(point1_distance**2 - R_E**2) + np.sqrt(point2_distance**2 - R_E**2):
        return True  # los
    else:
        return False  # Earth blocks the los


berlin_pos = geodetic_to_eci(berlin_lat, berlin_lon, 0)
cape_town_pos = geodetic_to_eci(cape_town_lat, cape_town_lon, 0)
"""

num_satellites = 100
satellites_in_los_berlin = []
satellites_in_los_cape_town = []
satellites_eci = dict()
print(cape_town_pos)

for i in range(num_satellites):
    sat_lat = np.random.uniform(-90, 90)
    sat_lon = np.random.uniform(-180, 180)
    satellite_pos = geodetic_to_eci(sat_lat, sat_lon, altitude)
    satellites_eci[i]=satellite_pos
    if line_of_sight(satellite_pos, berlin_pos):
        satellites_in_los_berlin.append((sat_lat, sat_lon, altitude))

    if line_of_sight(satellite_pos, cape_town_pos):
        satellites_in_los_cape_town.append((sat_lat, sat_lon, altitude))
"""


num_satellites = 50
satellites_in_los_berlin = {}
satellites_in_los_cape_town = {}
satellites_eci = {}
#adding ground pomts to the dict eci q
satellites_eci[0]=berlin_pos
satellites_eci[-1]=cape_town_pos

for i in range(1,num_satellites-1):
    sat_lat = np.random.uniform(-90, 90)
    sat_lon = np.random.uniform(-180, 180)
    satellite_pos = geodetic_to_eci(sat_lat, sat_lon, altitude)
    satellites_eci[i] = satellite_pos  # Add sat to dict

    if line_of_sight(satellite_pos, berlin_pos):
        satellites_in_los_berlin[i]=(sat_lat, sat_lon, altitude)

    if line_of_sight(satellite_pos, cape_town_pos):
        satellites_in_los_cape_town[i]=(sat_lat, sat_lon, altitude)


source = berlin_pos
target = cape_town_pos


distance, visited_sats, path = a_star_routing(source, target, satellites_eci)


print(f"Distance: {distance} km")
print(f"Visited Satellites: {visited_sats}")
print(f"Path indicies: {path}")
for idx in path:

  print(f"Path (ECI coordinates): {satellites_eci[idx]}")

print (path)

print(satellites_eci)

"""helper functions --------------------------

under trial not integrated yet

a trial functions ----------------------------
"""



def get_neighbors(node, satellite_positions):
    neighbors = []
    node_position = satellite_positions[node]  #  accesses by index

    for other_node, other_position in satellite_positions.items():
        if node != other_node:
            if satellite_coverage_check(node_position, other_position):
                neighbors.append(other_node)

    print(f"Satellite {node} has {len(neighbors)} neighbors.")
    return neighbors

dist,visted,path =a_star_routing(berlin_pos, cape_town_pos, satellites_eci)
print(dist)

satellite_positions = {
    0: (10000, 20000, 30000),
    1: (15000, 25000, 35000),
    2: (30000, 40000, 50000),
    3: (50000, 60000, 70000),
}

source = (10000, 20000, 30000)  # ECI  source
target = (50000, 60000, 70000)  # ECI  target

distance, visited_sats, path = a_star_routing(source, target, satellite_positions)
print("Distance:", distance)
print("Visited Satellites:", visited_sats)
print("Path:", path)

"""
#satellite positions
print("Satellite Positions:")
for idx, pos in satellites_eci.items():
    print(f"Satellite {idx}: {pos}")

# adjacency list
adjacency_list = {}
for k,node in satellites_eci.items():
    neighbors = get_neighbors(node, satellites_eci)  # Check `get_neighbors`
    adjacency_list[node] = neighbors
    print(f"Satellite {node} has {len(neighbors)} neighbors.")

#  for isolated satellites
isolated_nodes = [node for node, neighbors in adjacency_list.items() if not neighbors]
if isolated_nodes:
    print(f"Isolated satellites: {len(isolated_nodes)}")
    print(f"Isolated nodes: {isolated_nodes}")
else:
    print("No isolated satellites.")
"""

import numpy as np

def delta_walker_constellation(P, S, altitude, inclination, R_E=6371.8):

    satellites = {}
    index = 0

    inclination_rad = np.radians(inclination)

    semi_major_axis = R_E + altitude
    delta_omega = 360 / P  #
    delta_nu = 360 / S     #

    for plane in range(P):
        omega = plane * delta_omega
        for sat in range(S):
            nu = sat * delta_nu
            x = semi_major_axis * (np.cos(np.radians(omega)) * np.cos(np.radians(nu)) -
                                   np.sin(np.radians(omega)) * np.sin(np.radians(nu)) * np.cos(inclination_rad))
            y = semi_major_axis * (np.sin(np.radians(omega)) * np.cos(np.radians(nu)) +
                                   np.cos(np.radians(omega)) * np.sin(np.radians(nu)) * np.cos(inclination_rad))
            z = semi_major_axis * (np.sin(np.radians(nu)) * np.sin(inclination_rad))
            satellites[index] = np.array([x, y, z])
            index += 1

    return satellites

P = 5  # planes
S = 6  # sats per plane
altitude = 800
inclination = 53
F = 1  # phasing

satellites_eci_DW = delta_walker_constellation(P, S, altitude, inclination)
berlin_pos = geodetic_to_eci(52.52, 13.405, 0)
cape_town_pos = geodetic_to_eci(-33.9249, 18.4241, 0)

satellites_eci_DW[-1] = cape_town_pos  # Cape Town
satellites_eci_DW[0] = berlin_pos      # Berlin

berlin_pos, cape_town_pos

satellites_eci_DW

source = berlin_pos
target = cape_town_pos

distance, visited_sats, path_astar = a_star_routing(source, target, satellites_eci_DW)

print(f"Distance: {distance} km")
print(f"Visited Satellites: {visited_sats}")
print(f"Path indices: {path_astar}")
for idx in path_astar:
    print(f"Path (ECI coordinates): {satellites_eci_DW[idx]}")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_constellation(satellites_eci, R_E=6371.8):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the Earth
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x = R_E * np.cos(u) * np.sin(v)
    y = R_E * np.sin(u) * np.sin(v)
    z = R_E * np.cos(v)
    ax.plot_surface(x, y, z, color='blue', alpha=0.1)

    # Plot the satellites
    for pos in satellites_eci.values():
        ax.scatter(pos[0], pos[1], pos[2], color='red')

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    plt.show()

plot_constellation(satellites_eci_DW)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_constellation(satellites_eci, P, R_E=6371.8):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the Earth
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x = R_E * np.cos(u) * np.sin(v)
    y = R_E * np.sin(u) * np.sin(v)
    z = R_E * np.cos(v)
    ax.plot_surface(x, y, z, color='blue', alpha=0.1)

    # Define a color map for the planes
    colors = plt.cm.tab10(np.linspace(0, 1, P))  # 10 distinct colors

    # Plot the satellites with colors based on their plane
    for idx, pos in satellites_eci.items():
        plane = idx // (len(satellites_eci) // P)  # Determine the plane of the satellite
        ax.scatter(pos[0], pos[1], pos[2], color=colors[plane])

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    plt.legend()
    plt.show()

# Example usage
P = 6  # Number of planes
satellites_eci = delta_walker_constellation(P, 11, 550, 53)  # Generate constellation
plot_constellation(satellites_eci, P)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_constellation_with_route(satellites_eci, path_indices, start,end,R_E=6371):
    pos_berlin=berlin_pos
    pos_capetown=cape_town_pos
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the Earth
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x = R_E * np.cos(u) * np.sin(v)
    y = R_E * np.sin(u) * np.sin(v)
    z = R_E * np.cos(v)
    ax.plot_surface(x, y, z, color='blue', alpha=0.1)

    # Plot all satellites
    for idx, pos in satellites_eci.items():
        if idx in path_indices:
            # Highlight satellites in the optimal route
            ax.scatter(pos[0], pos[1], pos[2], color='red', s=100, label='Optimal point' if idx == path_indices[0] else "")
        else:
            # Plot other satellites
            ax.scatter(pos[0], pos[1], pos[2], color='gray', s=50, alpha=0.5)

    # Plot the optimal route
    route_positions = [satellites_eci[idx] for idx in path_indices]
    route_x = [pos[0] for pos in route_positions]
    route_y = [pos[1] for pos in route_positions]
    route_z = [pos[2] for pos in route_positions]
    ax.plot(route_x, route_y, route_z, color='green', marker='o', markersize=8, label='Optimal Route')
    #------------------------berlin capetown
    ax.scatter(start[0],start[1],start[2], color='yellow',s=50 ,marker='*',  label='start')

    ax.scatter(end[0],end[1],end[2], color='black',s=50, marker='*', label='end ')

    # Add labels and legend
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Satellite Constellation with Optimal Route')
    plt.legend()
    plt.show()

start = berlin_pos
end = cape_town_pos
plot_constellation_with_route(satellites_eci_DW, path_astar,start,end)



P2 = 4  # planes
S2 = 4  # sats per plane
altitude = 900
inclination = 55
F = 1  # phasing

satellites_eci_DW2 = delta_walker_constellation(P2, S2, altitude, inclination)


satellites_eci_DW2[-1] = cape_town_pos  # Cape Town
satellites_eci_DW2[0] = berlin_pos      # Berlin

satellites_eci_DW2

source = berlin_pos
target = cape_town_pos

distance2, visited_sats2, path_astar2 = a_star_routing(source, target, satellites_eci_DW2)

print(f"Distance: {distance} km")
print(f"Visited Satellites: {visited_sats2}")
print(f"Path indices: {path_astar2}")
for idx in path_astar2:
    print(f"Path (ECI coordinates): {satellites_eci_DW2[idx]}")

plot_constellation_with_route(satellites_eci_DW2, path_astar2, start,end,R_E=6371)

"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

R_EARTH = 6371.8
C = 3e5
NOISE_POWER = 1e-9

def delta_walker_constellation(P, S, altitude, inclination, R_E=R_EARTH):
    satellites = {}
    index = 0
    inclination_rad = np.radians(inclination)
    semi_major_axis = R_E + altitude
    delta_omega = 360 / P
    delta_nu = 360 / S

    for plane in range(P):
        omega = plane * delta_omega
        for sat in range(S):
            nu = sat * delta_nu
            x = semi_major_axis * (np.cos(np.radians(omega)) * np.cos(np.radians(nu)) -
                                   np.sin(np.radians(omega)) * np.sin(np.radians(nu)) * np.cos(inclination_rad))
            y = semi_major_axis * (np.sin(np.radians(omega)) * np.cos(np.radians(nu)) +
                                   np.cos(np.radians(omega)) * np.sin(np.radians(nu)) * np.cos(inclination_rad))
            z = semi_major_axis * (np.sin(np.radians(nu)) * np.sin(inclination_rad))
            satellites[index] = np.array([x, y, z])
            index += 1
    return satellites

def gen_walker_graph(P=6, S=10, altitude=1200, inclination=55, max_link_distance=5000, visualize=True):

    def distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    #  satellite positions
    satellites = delta_walker_constellation(P, S, altitude, inclination)
    sat_positions = list(satellites.values())

    #  the Graph
    G = nx.Graph()

    # Add nodes
    for idx, pos in satellites.items():
        G.add_node(idx, pos=pos)

    # Add edges
    for i in range(len(sat_positions)):
        for j in range(i+1, len(sat_positions)):
            d = distance(sat_positions[i], sat_positions[j])
            if d < max_link_distance:
                latency = d / C  # prop  (seconds)
                sinr = 1 / (NOISE_POWER * d)  # simplified SINR model

                G.add_edge(i, j, weight=d, latency=latency, sinr=sinr)

    print(f"Graph created with {len(G.nodes)} satellites and {len(G.edges)} ISLs.")


    if visualize:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        x, y, z = zip(*sat_positions)
        ax.scatter(x, y, z, c='b', marker='o', label="Satellites")

        #
        for u, v in G.edges():
            x_vals = [sat_positions[u][0], sat_positions[v][0]]
            y_vals = [sat_positions[u][1], sat_positions[v][1]]
            z_vals = [sat_positions[u][2], sat_positions[v][2]]
            ax.plot(x_vals, y_vals, z_vals, c='gray', alpha=0.5)

        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        plt.title("Walker-Delta Satellite Network")
        plt.legend()
        plt.show()

    return G
    """

Re=6371
new_york_pos = geodetic_to_eci(40.7128, -74.0060, 0)
tokyo_pos = geodetic_to_eci(35.6762, 139.6503, 0)
sydney_pos = geodetic_to_eci(-33.8688, 151.2093, 0)
moscow_pos = geodetic_to_eci(55.7558, 37.6173, 0)
wellington_pos = geodetic_to_eci(-41.2865, 174.7762, 0)  #

alert_pos = geodetic_to_eci(82.5018, -62.3481, 0)  # Alert, Nunavut
south_pole_pos = geodetic_to_eci(-90, 0, 0)

# Generate the Delta Walker constellation
satellites_eci_DW_a = delta_walker_constellation(P=6, S=11, altitude=550, inclination=53)

# Add ground stations to the dictionary
satellites_eci_DW_a[0] = cape_town_pos  #)
satellites_eci_DW_a[1] = berlin_pos
satellites_eci_DW_a[2] = new_york_pos
satellites_eci_DW_a[3] = tokyo_pos
satellites_eci_DW_a[4] = sydney_pos
satellites_eci_DW_a[5] = moscow_pos
satellites_eci_DW_a[6] = wellington_pos
satellites_eci_DW_a[7] = alert_pos
satellites_eci_DW_a[8] = south_pole_pos


start_end_pairs = [
    (new_york_pos, tokyo_pos),    #
    (sydney_pos, moscow_pos),
    (berlin_pos, new_york_pos),
    (cape_town_pos, sydney_pos),
    (berlin_pos, wellington_pos),
    (alert_pos,south_pole_pos)
]

results = []  # List to store results

for start, end in start_end_pairs:
    print(f"Routing from {start} to {end}")
    distance, visited_sats, path = a_star_routing(start, end, satellites_eci_DW_a)
    if distance == float('inf'):
      print("No los comeback later")
    else:
      print(f"Distance: {distance} km")
      print(f"Path indices: {path}")

    # Store results in a dictionary
    result = {
        "start": start,
        "end": end,
        "distance": distance,
        "visited_sats": visited_sats,
        "path": path
    }
    results.append(result)  # Add the result to the list

    # Print the results
    print(f"Distance: {distance} km")
    print(f"Visited Satellites: {visited_sats}")
    print(f"Path indices: {path}")
    for idx in path:
        print(f"Path (ECI coordinates): {satellites_eci_DW_a[idx]}")
    print("\n")

results[5]["distance"]-Re

plot_constellation_with_route(satellites_eci_DW_a, results[0]["path"], results[1]["start"],results[0]["end"],R_E=6371)

plot_constellation_with_route(satellites_eci_DW_a, results[1]["path"], results[1]["start"],results[1]["end"],R_E=6371)

plot_constellation_with_route(satellites_eci_DW_a, results[2]["path"], results[2]["start"],results[2]["end"],R_E=6371)

P3 = 4  # planes
S3 = 4  # sats per plane
altitude = 700
inclination = 45
F = 1  # phasing

berlin_pos = geodetic_to_eci(52.52, 13.405, 0)  #

plot_constellation_with_route(satellites_eci_DW_a, results[4]["path"], results[4]["start"],results[4]["end"],R_E=6371)



plot_constellation_with_route(satellites_eci_DW_a, results[5]["path"], results[5]["start"],results[5]["end"],R_E=6371)
