
# Satellite Routing Environment
class SatelliteRoutingEnv(gym.Env):
    def __init__(self):
        super(SatelliteRoutingEnv, self).__init__()
        self.R_E = 6371  # Radius of Earth in km
        self.num_satellites = 3
        self.state_space_var=3 #lat ,long ,h

        self.heights = [500, 600, 700, 800, 900, 1000]
        self.lats = [np.random.uniform(-90, 90) for i in range(self.num_satellites)] # Latitude grid
        self.lons = [np.random.uniform(-180, 180)  for i in range(self.num_satellites)]  # Longitude grid
        self.delta_height_range = [-100, 0, 100]
        self.delta_angle_range = [-5, 0, 5]
        self.P_transmitted = 1.0
        self.N_noise = 1.0  # Noise power
        self.c = 299792.458
        # act space: multi-d discrete acopns
        self.action_space = spaces.MultiDiscrete(
            [len(self.delta_height_range)] * self.num_satellites +
            [len(self.delta_angle_range)] * 2 * self.num_satellites  # Latitude, Longitude actions
        )

        # state space: 3 sats, each having lat, lon, and height
        self.observation_space = spaces.Box(low=-180, high=180, shape=(3 * self.num_satellites,), dtype=np.float32)



        self.A = self.geodetic_to_eci(52.52, 13.41)  # Berlin
        self.B = self.geodetic_to_eci(-33.92, 18.42)  # Cape Town

        self.state = None
        self.reset()

    def reset(self):
        self.state = self._generate_random_state()
        return self.state

    def _generate_random_state(self):
        state = []
        for _ in range(self.num_satellites):
            lat = np.random.choice(self.lats)
            lon = np.random.choice(self.lons)
            alt = 550 #np.random.choice(self.heights)
            state.extend([lat, lon, alt])
        return np.array(state)

    def step(self, action):
        new_state = self._apply_action(self.state, action)
        reward = self.compute_reward(self.state, action, new_state)
        self.state = new_state
        done = reward < -10000000   # Terminate episode if reward is too low
        return self.state, reward, done, {}




    def transition_probability(self, state, action):
        #lat1, lon1, h1, lat2, lon2, h2, lat3, lon3, h3 = state

        #actions=[] shouold have num stallites h actions ,num sats lat actions , sum sats long acts


        states=[state[i:i+3] for i in range(0,len(state), self.num_satellites)]
        #--------------------------------------------------------

        #dh1, dh2, dh3 = self.delta_height_range[action[0]], self.delta_height_range[action[1]], self.delta_height_range[action[2]]

        dh=[self.delta_height_range[action[i]] for i in range(self.num_satellites)]# len(dh) equals to num sats
        #---------------------------------------------------------



        #dlat1, dlat2, dlat3 = self.delta_angle_range[action[3]], self.delta_angle_range[action[4]], self.delta_angle_range[action[5]]

        dlat=[self.delta_height_range[action[i]] for i in range(self.num_satellites,self.num_satellites*2)]

        #---------------------------------------------------------------


        #dlon1, dlon2, dlon3 = self.delta_angle_range[action[6]], self.delta_angle_range[action[7]], self.delta_angle_range[action[8]]

        dlong=[self.delta_height_range[action[i]] for i in range(self.num_satellites*2,self.num_satellites*3)]

        h_new = [min(max(state[i][2] + dh[i], 500), 600) for i in range(self.num_satellites)]
        lat_new = [min(max(state[i][0] + dlat[i], -90), 90) for i in range(self.num_satellites)]
        lon_new = [min(max(state[i][1] + dlong[i], -180), 180) for i in range(self.num_satellites)]




        """
        h2_new = min(max(h2 + dh2, 500), 600)
        h3_new = min(max(h3 + dh3, 500), 600)

        lat1_new = min(max(lat1 + dlat1, -90), 90)
        lat2_new = min(max(lat2 + dlat2, -90), 90)
        lat3_new = min(max(lat3 + dlat3, -90), 90)

        lon1_new = (lon1 + dlon1) % 360
        lon2_new = (lon2 + dlon2) % 360
        lon3_new = (lon3 + dlon3) % 360
        """
        #-----------------------------------------------------------------------
        #replaced with a lost
        """
        new_state = (lat1_new, lon1_new, h1_new,
                     lat2_new, lon2_new, h2_new,
                     lat3_new, lon3_new, h3_new)


        """

        new_states=[[lat_new[i],lat_new[i],h_new[i]] for i in range(self.num_satellites)]
        #---------------------------------------------

        new_state_eci = [
            [self.geodetic_to_eci(lat_new[i],lat_new[i],h_new[i])] for i in range(self.num_satellites)]

        return new_states

        """
        for coords in new_state_eci:
            if not self.is_valid_eci(coords):
                new_state = self.find_closest_state(new_state)   i ll remove this condition from the transition function and add it to the reward
        """

    def generate_training_data(self,num_samples):
      training_data = []
      satellite_positions=[]
      for _ in range(num_samples):
        # Random scenario
        source, target =self.A, self.B
        lat=np.random.uniform(-90, 90)
        lon=np.random.uniform(-180, 180)
        alt=550
        satellite_positions .extend([self.geodetic_to_eci(lat,lon,alt)])  # 10
        #satellites with lat, lon, h
      distance, visited_sats, path = self.a_star_routing(source, target, satellite_positions)
      print("distance",distance)
      print("visited sats",visited_sats)
      print("path",path)

        # Convert path into target actions (example: next-step adjustments)
      actions = np.zeros_like(satellite_positions)
      for i in range(len(path) - 1):
            actions[path[i]] = satellite_positions[path[i + 1]] - satellite_positions[path[i]]

        # Flatten positions and actions for training
            training_data.append((satellite_positions, actions))
      return training_data
    def is_valid_eci(self, coords):

        x, y, z = coords
        distance_from_center = np.linalg.norm([x, y, z])
        return (self.R_E + 500) <= distance_from_center <= (self.R_E + 1000)

    #

    def geographic_distance(self,s1, s2):
      # info from  states
      lat1, long1, alt1 = s1
      lat2, long2, alt2 = s2

      #
      lat1_rad, long1_rad = np.radians(lat1), np.radians(long1)
      lat2_rad, long2_rad = np.radians(lat2), np.radians(long2)


      R = 6371.0

      # Haversine formula to calculate the distance on the sphere
      delta_lat = lat2_rad - lat1_rad
      delta_long = long2_rad - long1_rad
      a = np.sin(delta_lat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_long / 2)**2
      c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

      #
      surface_distance = R * c#-----

      #   dist with the alt diff (Euclidean)
      alt_diff = abs(alt2 - alt1)
      total_distance = np.sqrt(surface_distance**2 + alt_diff**2)

      return total_distance


    def is_on_earth(self,position, epsilon=1e-3):#to check wether the current node is a sallite or a ground point while routing
      R_E = 6371
      distance_from_center = np.linalg.norm(position)
      return abs(distance_from_center - R_E) >= epsilon


    def find_closest_state(self, state):
    # min dis
      closest_state = min(self.state_space, key=lambda s: geographic_distance(state, s))
      return closest_state


    def julian_date(self,utc_time):

      unix_epoch = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
      delta_seconds = (utc_time - unix_epoch).total_seconds()
      julian_date = 2440587.5 + delta_seconds / 86400.0
      return julian_date

    def gst_from_julian(self,julian_date):
      T = (julian_date - 2451545.0) / 36525.0
      GST = 280.46061837 + 360.98564736629 * (julian_date - 2451545.0) + 0.000387933 * T**2 - (T**3 / 38710000.0)
      GST = GST % 360.0  # Keep within 0 to 360 degrees
      return np.radians(GST)

    def geodetic_to_eci(self,lat_deg, lon_deg, altitude_km=0, utc_time=None):

      if utc_time is None:
        utc_time = datetime.utcnow().replace(tzinfo=timezone.utc)

      lat = np.radians(lat_deg)
      lon = np.radians(lon_deg)

      R_E = 6378.137

    # Compute the Julian Date from the given UTC time
      jd = self.julian_date(utc_time)

      gst = self.gst_from_julian(jd)

      lon_eci = lon + gst  # Longitude in the inertial frame (ECI)

      x = (R_E + altitude_km) * np.cos(lat) * np.cos(lon_eci)
      y = (R_E + altitude_km) * np.cos(lat) * np.sin(lon_eci)
      z = (R_E + altitude_km) * np.sin(lat)

      return x, y, z





    def los_between_satellites(self,pos1, pos2, earth_radius=6371.0):

      p1 = np.array(pos1)
      p2 = np.array(pos2)

      vector = p2 - p1

      midpoint = (p1 + p2) / 2

      distance_midpoint_to_center = np.linalg.norm(midpoint)

      return distance_midpoint_to_center >= earth_radius



    def line_of_sight(self,satellite_pos, ground_pos, R_E=6371):# ground to sattlite

      satellite_pos = np.array(satellite_pos)
      ground_pos = np.array(ground_pos)

      ground_to_sat = satellite_pos - ground_pos

      ground_distance = np.linalg.norm(ground_pos)

      angle = np.arccos(np.dot(ground_to_sat, ground_pos) / (np.linalg.norm(ground_to_sat) * ground_distance))


      if angle < np.pi / 2:
        return True  # sat is in LoS
      else:
        return False  # No LoS
    def satellite_coverage_check(self,r1, r2): #between sattlites in space with input ECI postions
      Re=6371

      r1 = np.array(r1) #  1st sat (x1, y1, z1)
      r2 = np.array(r2) #pos of the 2 sat (x1, y1, z1)

      # get the Euc dist
      d = np.linalg.norm(r1 - r2)

      # get the min dist-> earth & line (connecting the sats)
      d_min = np.linalg.norm(np.cross(r1, r2)) / d

      # chk if the min dist > earth rad
      return d_min > Re



    def closest_satellite(self, B, satellites):
        return min(satellites, key=lambda s: distance.euclidean(B, s))
    def gst_from_julian(self,julian_date):
      T = (julian_date - 2451545.0) / 36525.0
      GST = 280.46061837 + 360.98564736629 * (julian_date - 2451545.0) + 0.000387933 * T**2 - (T**3 / 38710000.0)
      GST = GST % 360.0  # Keep within 0 to 360 degrees
      return np.radians(GST)

    def geodetic_to_eci(self,lat_deg, lon_deg, altitude_km=0, utc_time=None):

      if utc_time is None:
        utc_time = datetime.utcnow().replace(tzinfo=timezone.utc)

      lat = np.radians(lat_deg)
      lon = np.radians(lon_deg)

      R_E = 6378.137

    # Compute the Julian Date from the given UTC time
      jd = self.julian_date(utc_time)

      gst = self.gst_from_julian(jd)

      lon_eci = lon + gst  # Longitude in the inertial frame (ECI)

      x = (R_E + altitude_km) * np.cos(lat) * np.cos(lon_eci)
      y = (R_E + altitude_km) * np.cos(lat) * np.sin(lon_eci)
      z = (R_E + altitude_km) * np.sin(lat)

      return x, y, z

    def coverage(self, h, N):

      if h <= 0 or N <= 0:
        return 0  # no coverage for +++non-positive h or satellite cnt

      return 1 - np.exp(-N * h / (2 * (self.R_E + h)))


    def path_loss(self, d, L_0, n, d_0):
        return L_0 * (d / d_0) ** -n

    def received_power(self, P_transmitted, path_loss, l, zeta):
        return P_transmitted * l / path_loss

    def SINR(self, P_received, interference, noise):
        return P_received / (interference + noise)

    def propagation_delay(self, distance):

        return distance / self.c
    def average_sinr(self, satellite_positions, ground_positions):
      sinr_values = []
      for ground in ground_positions:
        for sat in satellite_positions:
            if self.line_of_sight(sat, ground):
                power = self.received_power(self.P_transmitted, self.path_loss(sat, ground), 1, 1)
                sinr_values.append(self.SINR(power, 0, self.N_noise))
      return np.mean(sinr_values) if sinr_values else 0

    def penalty_for_invalid_states(self, satellite_positions):
      for pos in satellite_positions:
        if not self.is_valid_eci(pos):
            return -100000  # Large penalty for invalid positions
      return 0

    def find_route(self,A, B, satellites):
      d_tot = 0
      num_hops = 0
      current_node = A
      visited_satellites = set()

      while True :
        # gwt satellites with a line of sight to the current node, exclude  visited satellites
        if self.is_on_earth(current_node):

          los_satellites = [s for s in satellites if self.line_of_sight (current_node, s) and s not in visited_satellites]


        else:
          los_satellites = [s for s in satellites if self.satellite_coverage_check (s, current_node) and s not in
            visited_satellites] #the current node is a sattliete

        if not los_satellites:
            # No forward satellites available, coverage failed
            #raise ValueError("Coverage failed: No forward satellites with line of sight found.")
            return 100000000000000,100000000000000000000000,[] #to penalize no covg

            break



        # Find the closest satellite to B from the available line-of-sight satellites

        s_hat = self.closest_satellite(B, los_satellites)
        if(s_hat in visited_satellites):
          continue  # skip if the closest satellite  visited

        # updt total dist and incr the nr of hops
        d_tot += distance.euclidean(current_node, s_hat)
        num_hops += 1

        # mark the curr sat as visited -- no back hoping
        visited_satellites.add(s_hat)

        # chk if there's a line of sight between the chosen satellite and the destination B
        if(self.is_on_earth(B)):

          if self.line_of_sight(s_hat, B):
            d_totaltitude += distance.euclidean(s_hat, B)
            break
        else :
          if self.satellite_coverage_check(s_hat, B):
            d_tot += distance.euclidean(s_hat, B)
            break
        # Move to the next satellite for the next hop
        current_node = s_hat

      return d_tot, num_hops,visited_satellites

    def haversine_distance(self, pos1, pos2):

        R = 6371


        lat1, lon1 = map(math.radians, pos1)
        lat2, lon2 = map(math.radians, pos2)


        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Haversine formula
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c

        return distance

    def get_neighbors(self, current, satellite_positions):

      neighbors = []

      current_position = current

      print("current_position",current_position,"satellite_position",satellite_positions)

      for i, position in enumerate(satellite_positions):
        print("current_position",current_position,"satellite_position",position)
        if position != current:  # Exclude the current satellite
            if self.satellite_coverage_check(current_position, position):
                neighbors.append(i)
      return neighbors

    def a_star_routing(self, source, target, satellite_positions):
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
         if method == "great_circle":
           return self.haversine_distance(satellite_positions[current], satellite_positions[goal])
         elif method == "delay":
            distance = np.linalg.norm(np.array(satellite_positions[current]) - np.array(satellite_positions[goal]))
            return distance / 299792458



      while open_set:
        idx, current = heapq.heappop(open_set)
        print("current",current,idx)
        visited_sats.add(current)

        # target is reached?
        if current == target:
            path = self.reconstruct_path(came_from, current)
            distance = sum(
                self.edge_cost(path[i], path[i + 1], satellite_positions)
                for i in range(len(path) - 1)
            )
            return distance, list(visited_sats), path
        print("satellite_positions",satellite_positions)
        # Explore neighbors
        for neighbor in self.get_neighbors(current, satellite_positions):
            tentative_g_cost = g_cost[current] + self.edge_cost(current, neighbor, satellite_positions)

            if tentative_g_cost < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g_cost
                f_cost = tentative_g_cost + heuristic_cost_estimate(neighbor, target)
                heapq.heappush(open_set, (f_cost, neighbor))
                came_from[neighbor] = current

      return float('inf'), list(visited_sats), []  # No valid path found

    def reconstruct_path(self, came_from, current):
      path = [current]
      while current in came_from:
        current = came_from[current]
        path.append(current)
        path.reverse()
      return path

    def edge_cost(self, node1, node2, satellite_positions):
      x1, y1, z1 = satellite_positions[node1]
      x2, y2, z2 = satellite_positions[node2]
      return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)  # Euclidean distance


    def compute_reward(self, state, action, new_state):
      """
      lat1, lon1, h1, lat2, lon2, h2, lat3, lon3, h3 = state
      lat1_new, lon1_new, h1_new, lat2_new, lon2_new, h2_new, lat3_new, lon3_new, h3_new = new_state
      """
      #print(" state",state)
      states=[state[i:i+3] for i in range(0,len(state), self.state_space_var)]
      print("states",states)

      new_states=[new_state[i:i+3] for i in range(0,len(new_state), self.state_space_var)]

      print("new states",new_states)

      #print("new states",new_states)
      #states =[(lat,long,h),,,,,,,,,,]
      #------------------------------------------------------------------
      old_sat_positions = [self.geodetic_to_eci(*state[i:i+3]) for i in range(0,len(state), self.state_space_var)]

      new_sat_positions = [self.geodetic_to_eci(*new_state[i:i+3]) for i in range(0,len(state), self.state_space_var)]

      #---------------------------------------------------------------------
      """
      satellite_positions = [
        self.geodetic_to_eci(lat1, lon1, h1),
        self.geodetic_to_eci(lat2, lon2, h2),
        self.geodetic_to_eci(lat3, lon3, h3)]
      satellite_positions_new = [
        self.geodetic_to_eci(lat1_new, lon1_new, h1_new),
        self.geodetic_to_eci(lat2_new, lon2_new, h2_new),
        self.geodetic_to_eci(lat3_new, lon3_new, h3_new)]
      """
      # chk if  new positions are valid ----eci
      #if not all(self.is_valid_eci(pos) for pos in new_sat_positions):
      #  return -10000000000000000000000000  # penalize invalid states heavily
      #  coverage

      penalty = self.penalty_for_invalid_states(new_sat_positions)


      #coverage_old = sum(self.coverage(h, self.num_satellites) for h in [h1, h2, h3])
      #coverage_new = sum(self.coverage(h, self.num_satellites) for h in [h1_new, h2_new, h3_new])

      h_old = [states[i][2] for i in range(self.num_satellites)]
      h_new = [new_states[i][2] for i in range(self.num_satellites)]

      coverage_old = sum([self.coverage(h, self.num_satellites) for h in  h_old])   #states[i,2] =h

      coverage_new = sum([self.coverage(h, self.num_satellites) for h in h_new])
      print("covg old",coverage_old)
      #--------------------------------------------------------------------------
      # p loss and SINR
      #path_loss_old = [self.path_loss(np.sqrt((self.R_E + h) ** 2 - self.R_E ** 2), 1.0, 2.0, 1.0) for h in [h1, h2, h3]]
      #path_loss_new = [self.path_loss(np.sqrt((self.R_E + h_new) ** 2 - self.R_E ** 2), 1.0, 2.0, 1.0) for h_new in [h1_new, h2_new, h3_new]]
      #-----------------------------------------------------------------------
      # p loss and SINR

      path_loss_old = [self.path_loss(np.sqrt((self.R_E +h) ** 2 - self.R_E ** 2), 1.0, 2.0, 1.0) for h in  h_old]

      path_loss_new = [self.path_loss(np.sqrt((self.R_E + h) ** 2 - self.R_E ** 2), 1.0, 2.0, 1.0) for h in h_new]




      #------------------------------------------------------------------------
      P_received_old = [self.received_power(self.P_transmitted, pl, 1, 1) for pl in path_loss_old]
      P_received_new = [self.received_power(self.P_transmitted, pl, 1, 1) for pl in path_loss_new]



      #-------------------------------------------------------------------------






      SINR_old = [self.SINR(P, 0, self.N_noise) for P in P_received_old]
      SINR_new = [self.SINR(P, 0, self.N_noise) for P in P_received_new]


      #-------------------------------------------------------------------------
      """
      # Routing and latency calculations
      d_tot_old, num_hops_old,vis_sats = self.find_route(self.A, self.B, old_sat_positions)
      d_tot_new, num_hops_new,new_vis_sats = self.find_route(self.A, self.B, new_sat_positions)
      """
      latency_old = self.propagation_delay(d_tot_old) + num_hops_old * 1  # Assuming L_processing = 1
      latency_new = self.propagation_delay(d_tot_new) + num_hops_new * 1

      # Reward weights
      alpha = 0.8  # Coverage weight
      beta = 0.2   # Latency weight
      gamma = 1.0  # SINR weight

      d_covg=coverage_new - coverage_old


      if(d_covg==0 and  coverage_old>0  ):
        cov_factor= alpha*coverage_old
      elif(d_covg==0 and  coverage_old==0 ):
        cov_factor =alpha*-1
      else:
        cov_factor= alpha*d_covg


      d_lat=latency_new - latency_old
      if(d_lat==0 and  latency_old>0  ):
        lat_factor= beta*-10
      elif(d_lat==0 and  latency_old==0 ):
        lat_factor =10

      else :
        lat_factor= -beta*d_lat


      # Compute reward
      """
      reward = (
        alpha * (coverage_new - coverage_old)
        - beta * (latency_new - latency_old)
        +penalty)
        #+ gamma * (np.mean(SINR_new) - np.mean(SINR_old)))
      """
      reward = (lat_factor+cov_factor+penalty)

      return reward


    def _apply_action(self, state, action):
        new_state = state.copy()
        #action []
        for i in range(self.num_satellites):
            lat_idx = i * 3
            lon_idx = i * 3 + 1
            alt_idx = i * 3 + 2
            new_state[lat_idx] += (action[i * 3] - 1) * self.delta_angle_range[0]  # lat adj
            new_state[lon_idx] += (action[i * 3 + 1] - 1) * self.delta_angle_range[0]  # long adj
            new_state[alt_idx] += (action[i * 3 + 2] - 1) * self.delta_height_range[0]  # h adjustment
        return np.clip(new_state, -180, 1000)  # Valid bounds




epochs=0
env = SatelliteRoutingEnv()
rewards=[]
# A2C agent
model = A2C("MlpPolicy", env, verbose=1)
samples=20

training_data=env.generate_training_data(samples)
#
print( "training_data",training_data )
"""
model.learn(total_timesteps=500)

#
model.save("a2c_satellite_routing")

obs = env.reset()
done = False
while not epochs>2:
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    rewards.append(reward)
    epochs+=1
    print(f"Action: {action}, Reward: {reward}")


plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("A2C Reward Over Time")
plt.show()
