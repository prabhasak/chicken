import gym
from gym import spaces
import numpy as np
import random


REWARD_TIME_OUT         =  -50.0
REWARD_COLLISION        = -100.0
REWARD_SUCCESS          =   75.0
REWARD_FEAR             =  -20.0

MAXIMUM_EPISODE_TIME    = 250

SEED = 42

class CarState:
    def __init__(self, car_x_loc, car_y_loc, car_vel, car_acc):
        self.car_x_loc = car_x_loc
        self.car_y_loc = car_y_loc
        self.car_acc   = car_acc
        self.car_vel   = car_vel

class PedState:
    def __init__(self, ped_x_loc, ped_y_loc, ped_vel):
        self.ped_x_loc = ped_x_loc
        self.ped_y_loc = ped_y_loc
        self.ped_vel   = ped_vel

class PedestrianEnv(gym.Env):
    """
        <----------------------50----------------------->
     ^  =======88888888888==========================================================
     |         8   Car   8                               Goal
    7.5 #######8###==>###8###############################X##########################
     |         8         8                               X
    4.5 #######88888888888###############################X##########################
     |                                                   X
    1.5 #################################################X##########################
     |                                                   Start
     V  ============================================================================

    8 - safe zone
    """
    def __init__(self,
                 width_of_the_road          = 10.0,
                 length_of_road_considered  = 50.0,
                 car_min_start_vel_limit    = 10.0,
                 car_max_vel_limit          = 15.0,
                 car_acc_limit              =  3.0,
                 ped_velocity_limit         =  2.0,
                 car_len_by_2               =  2.0,
                 car_width_by_2             =  1.0,
                 safe_zone_x                =  4.0,
                 safe_zone_y                =  3.0,
                 stop_at_zebra_crossing     =  0.8,
                 static_state_prob          =  0.1,
                 delta_t                    = 0.25):

        self.width_of_the_road              =  width_of_the_road
        self.length_of_road_considered      =  length_of_road_considered
        self.state                          =  None
        self.car_state                      =  CarState(car_x_loc=0, car_y_loc=0, car_vel=0, car_acc=0)
        self.ped_state                      =  PedState(ped_x_loc=0, ped_y_loc=0, ped_vel=0)
        self.car_next_state                 =  CarState(car_x_loc=0, car_y_loc=0, car_vel=0, car_acc=0)
        self.ped_next_state                 =  PedState(ped_x_loc=0, ped_y_loc=0, ped_vel=0)
        self.car_min_start_vel_limit        =  car_min_start_vel_limit
        self.car_max_vel_limit              =  car_max_vel_limit
        self.car_acc_limit                  =  car_acc_limit
        self.ped_velocity_limit             =  ped_velocity_limit
        self.car_len_by_2                   =  car_len_by_2
        self.car_width_by_2                 =  car_width_by_2
        self.safe_zone_x                    =  safe_zone_x
        self.safe_zone_y                    =  safe_zone_y
        self.stop_at_zebra_crossing         =  stop_at_zebra_crossing
        self.static_state_prob              =  static_state_prob
        self.delta_t                        =  delta_t
        self.time                           =  None
        self.static_state                   =  0
        # Used for exploration from in epsilon greedy approach
        self.action_space                   =  spaces.Box(np.array([-1 * self.ped_velocity_limit]),
                                                          np.array([-1 * self.ped_velocity_limit]),
                                                           dtype=np.float32)

    def _collision(self):
        """
        Return True if pedestrian is killed
        :return: Bool
        """
        if (self.car_state.car_y_loc - self.car_len_by_2
                < self.ped_state.ped_y_loc <
            self.car_state.car_y_loc + self.car_len_by_2) and \
            (self.car_state.car_x_loc - self.car_width_by_2
                < self.ped_state.ped_x_loc <
             self.car_state.car_x_loc + self.car_width_by_2):
            return True
        else:
            return False

    def _success(self):
        """
        Return True if the pedestrian crosses the street
        :return: Bool
        """
        if self.ped_state.ped_y_loc >= self.width_of_the_road:
            return True
        else:
            return False

    def _car_in_zebra_crossing(self):

        car_almost_crashes = (self.car_state.car_y_loc - self.safe_zone_y
                                    < self.ped_state.ped_y_loc <
                                self.car_state.car_y_loc + self.safe_zone_y) and \
                                (self.length_of_road_considered - self.safe_zone_x
                                    < self.car_state.car_x_loc <
                                 self.length_of_road_considered - self.car_len_by_2)

        prob = random.uniform(0, 1) < self.stop_at_zebra_crossing

        car_in_danger_zone = self.length_of_road_considered - 20 < self.car_state.car_x_loc < self.length_of_road_considered - 10 and \
                              self.ped_state.ped_y_loc < self.car_state.car_y_loc + self.car_width_by_2 and prob

        return car_almost_crashes or car_in_danger_zone

    def _car_in_zebra_crossing_And_pedestrian_did_no_cross(self):
        if self.ped_state.ped_y_loc < self.car_state.car_y_loc + self.car_width_by_2 and \
            self.length_of_road_considered - 10.0 < self.car_state.car_x_loc < self.length_of_road_considered:
            return True

    def _fear_reward(self, reward_fear_factor):
        """
        Return True if pedestrian is killed
        :return: Reward by Fear
        """

        reward_travel = -0.1*(self.width_of_the_road + 2 - self.ped_state.ped_y_loc)**2

        if (self.length_of_road_considered - 25
            < self.car_state.car_x_loc <
            self.length_of_road_considered + self.car_width_by_2)   and \
            self.car_state.car_y_loc + self.car_width_by_2 > self.ped_state.ped_y_loc:

            vel_fraction = self.car_state.car_vel / (self.car_max_vel_limit + 0.1)
            distance = ((self.ped_state.ped_y_loc - self.car_state.car_y_loc)**2 +
                          (self.ped_state.ped_x_loc - self.car_state.car_x_loc)**2) ** 0.5

            reward_fear = reward_fear_factor * (vel_fraction ** distance)

            return reward_fear + reward_travel
        else:

            return reward_travel

    def reset(self, random_seed=1):

        # Starting time
        self.time = 0

        # Selecting if the car is stopped or not for the episode
        self.static_state = random.uniform(0,1) < self.static_state_prob

        # Initializing Pedestrian State, Here x is fixed and y varies
        self.ped_state.ped_x_loc = self.length_of_road_considered
        self.ped_state.ped_y_loc = 0.0
        self.ped_state.ped_vel   = 0.0
        self.ped_next_state.ped_x_loc = self.length_of_road_considered
        self.ped_next_state.ped_y_loc = 0.0
        self.ped_next_state.ped_vel   = 0.0

        # Initializing Car State
        if not self.static_state:

            if random_seed == 1:  # lane 1
                self.car_state.car_y_loc = random.uniform(2.5, 5.0)
            elif random_seed == 2:  # lane 2
                self.car_state.car_y_loc = random.uniform(5.0, 7.0)
            else:  # lane 3
                self.car_state.car_y_loc = random.uniform(7.0, self.width_of_the_road - 1.0)

            self.car_state.car_x_loc = random.uniform(0.0, self.length_of_road_considered + 10.0)
            self.car_state.car_vel   = random.uniform(self.car_min_start_vel_limit, self.car_max_vel_limit)
            self.car_state.car_acc   = random.uniform(0.0, self.car_acc_limit)

        else:
            # Static Case
            self.car_state.car_x_loc = random.uniform(0.0, self.length_of_road_considered - self.car_len_by_2)
            self.car_state.car_y_loc = random.uniform(2.5, self.width_of_the_road - 1.0)
            self.car_state.car_vel   = 0.0
            self.car_state.car_acc   = 0.0


        state = [self.ped_state.ped_y_loc,
                 self.car_state.car_x_loc,
                 self.car_state.car_y_loc,
                 self.car_state.car_vel,
                 self.car_state.car_acc]

        return state, self.static_state

    def step_car(self):

        if self.static_state or self._car_in_zebra_crossing():
            self.car_next_state.car_acc = 0.0
            self.car_next_state.car_vel = 0.0
        else:
            # Car in zebra crossing location and pedestrian did not cross, so slow down
            # Decreasing the maximum velocity
            if self._car_in_zebra_crossing_And_pedestrian_did_no_cross():
                self.car_next_state.car_acc = random.uniform(0, self.car_acc_limit)
                self.car_next_state.car_vel = min(
                                                max(0.0, self.car_state.car_vel + self.car_next_state.car_acc*self.delta_t),
                                                self.car_max_vel_limit / 2
                                                )
            else:
                self.car_next_state.car_acc = random.uniform(-self.car_acc_limit, self.car_acc_limit)
                self.car_next_state.car_vel = min(
                                    max(0.0, self.car_state.car_vel + self.car_next_state.car_acc*self.delta_t),
                                    self.car_max_vel_limit
                                    )
        self.car_next_state.car_y_loc = self.car_state.car_y_loc
        self.car_next_state.car_x_loc = max(
                                            self.car_state.car_x_loc,
                                            self.car_state.car_x_loc + self.car_state.car_vel * self.delta_t +
                                            self.car_next_state.car_acc*self.delta_t**2)

        return

    def step_ped(self, action):

        self.ped_next_state.ped_y_loc = min(
                                            self.width_of_the_road,
                                            max(0, self.ped_state.ped_y_loc + action*self.delta_t)
                                            )
        if self.time > MAXIMUM_EPISODE_TIME:
            reward = REWARD_TIME_OUT
            done   = True
        elif self._collision():
            reward = REWARD_COLLISION
            done   = True
        elif self._success():
            reward = REWARD_SUCCESS
            done   = True
        else:
            reward = self._fear_reward(REWARD_FEAR)
            done   = False

        return reward, done

    def step(self, action):
        action = action[0]
        self.time += 1

        self.step_car()
        reward, done = self.step_ped(action)

        # Transitioning from current state to next state
        self.ped_state  = self.ped_next_state
        self.car_state  = self.car_next_state

        state = [self.ped_state.ped_y_loc,
                 self.car_state.car_x_loc,
                 self.car_state.car_y_loc,
                 self.car_state.car_vel,
                 self.car_state.car_acc]

        return state, reward, done



###Junk Code

# self.ped_state < (self.car_y_loc + self.car_width_by_2) and
#    (self.length_of_road_considered - 10) < self.car_x_loc < self.length_of_road_considered


## ToDo in Modularity
#  1. c_y_position restriction to modularity
#  2. change every value in initial loop
#  3. change the parameters from the init function to the paramters above
#  4. Change fear reward condition
#  5. Action for Car
#  6.  0.1

## Todo change the Behaviour
#  > Change the maximum velocity thing
#  > Chnage the streets thing to one
#  > Change 0.1 to something small
#  Reward Fear is (vel_fraction)**distance_between_both