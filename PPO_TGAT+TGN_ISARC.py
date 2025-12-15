import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import itertools
import gym
from gym import spaces
from torch.distributions import Categorical
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv, GATv2Conv # Graph attention layers
import math
from collections import defaultdict
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

# To see if the virtual environment is being used.
import sys
import time
start_time = time.time()
print(f"Python interpreter used: {sys.executable}")

import os

# Define the directory to save models
model_save_dir = "Add the path to directory here"

# Need to make changes in Graph structure. 
# Hereby, I am implementing it using the TGAT based TGN. 
# Create the directory if it doesn't exist
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

# Define the environment
class PrecastSchedulingEnv(gym.Env): #This class is modeling the environment in which the agent interacts
    metadata = {'render.modes' : ['human']}

    def __init__(self, n_elements, daily_crane_cost, daily_labor_cost, daily_indirect_cost, crane_capacity, labor_capacity, weather_forecast, 
                 material_schedule, weights, support_relations, module_area, adjacency_matrix, interference_matrix, resource_requirements, duration_distribution, 
                 weather_impact_distribution, initial_material_availability, element_types, structure_support_relationships, action_space, weight, area):
        super(PrecastSchedulingEnv, self).__init__() 
    

        self.n_elements = n_elements #Number of elements to be scheduled
        self.daily_crane_cost = daily_crane_cost #Daily crane cost
        self.daily_labor_cost = daily_labor_cost #Daily labor cost
        self.daily_indirect_cost = daily_indirect_cost #Daily inditect cost of the resources

        self.crane_capacity = crane_capacity #The available number of cranes
        self.labor_capacity = labor_capacity #The available number of labor. You can create this in form of a matrix for different resources like below
        self.crane_available = crane_capacity # Initialize them with the maximum availability
        self.labor_availabe = labor_capacity # Initialize them with the maximum availability, which will then be updated with the number of steps, I think. 

        self.material_availability_schedule = material_schedule # To check the constraints against the material availabiliy. This will be in form of which modules are delivered on which day. 
        self.initial_material_availability = initial_material_availability 
        self.material_availability = initial_material_availability # This is again a vector, showin which elements are available.

        self.resource_requirements = resource_requirements

        self.weather_forecast = weather_forecast  # Weather forecast data. This will be a daily probability distribution, based on which the agent has to make the decision. 
       
        self.weather_conditions = None # To store the sampled weather condition for the current day
        self.weather_impact_distribution = weather_impact_distribution # Need to have a distribution of impact of weather on the activity durations. 

        self.min_rain, self.max_rain, self.min_temp, self.max_temp, self.min_wind, self.max_wind = self.get_minmaxweather(weather_forecast)

        self.weights = weights #To check the weight constraints of the modules
        self.support_relations = support_relations # To check the structural support relations of the modules
        self.module_area = module_area # To check the area of the modules

        self.adjacency_matrix = adjacency_matrix # Storing the adjacency relationships of the elements
        self.interference_matrix = interference_matrix # Storing the number of directions in which the interference is stopped due to a particular element being installed before the other element

        self.total_time_steps = 0 # Total timesteps
        self.current_day = 0  # Current day (increments every 8 hours)
        self.current_time = 0  # Hour within the current day - May be you can update it every 8 hour. 

        
        self.element_types = element_types  
        
        self.newly_completed_elements_indices = torch.zeros(self.n_elements, dtype=torch.bool)  # All elements start as incomplete
        self.newly_completed_elements = 0 # Tracking the count of newly_completed_elements
        
        self.duration_params = duration_distribution
        
        self.duration_normalization_param = self.get_min_duration(duration_distribution)

        # Initialize node features and edge indices for the GCN
        self.node_features = torch.zeros((self.n_elements, 8))  # [status, type, crane_needed, labor_needed, actual_duration, remaining_duration, weight, area]
        self.edge_index = torch.empty((2, 0), dtype=torch.long) # Edge indices for the graph (empty at initialization)

        self.action_space = torch.from_numpy(np.array(action_space)) # Generating the action space here. 

        self.structural_support_relationships = structure_support_relationships # Setting the structural support relationships

        # So I will maintain three arrays here.
        self.in_progress_elements = np.zeros(n_elements) 
        self.completed_elements = np.zeros(n_elements)
        self.not_yet_started_elements = np.ones(n_elements)

        self.timestamp = self.current_time # Variable to track the current time.

        self.weights = weight # The numpy array of weight of elements
        self.area = area # The numpy array of area of elements

        self.observation_space = spaces.Dict({ 
            'node_features': spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_elements, 8), dtype=np.float32), 
            'edge_index': spaces.Box(low=0, high=1, shape=(self.n_elements, self.n_elements), dtype=np.float32), 
            'remaining_elements': spaces.MultiBinary(self.n_elements), 
            'resource_availability': spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32), 
            'material_availability': spaces.MultiBinary(self.n_elements), 
            'weather_conditions': spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.weather_forecast),3), dtype=np.float32),  
            'action_mask': spaces.MultiBinary(len(self.action_space)),  
            'structure_support_relationships': spaces.Box(low=0, high=1, shape=(self.n_elements, self.n_elements), dtype=np.float32),  
            'timestamp' : spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_elements,1), dtype=np.float32)  
        })

        self.reset()

    def get_max_duration(self, duration_distribution):
        max_value = float('-inf')
        for values in duration_distribution.values():
            max_value = max(max(values), max_value)

        return max_value
    
    def get_min_duration(self, duration_distribution):
        min_value = float('inf')
        for values in duration_distribution.values():
            min_value = min(min(values), min_value)

        return min_value

    def getActionSpace(self, n_elements, crane_capacity):
        """
        Generate all valid actions where the nmber of selected elements is less than crane_capacity
        
        Returns : list of binary arrays representing valid actions"""
        valid_actions = []
        for k in range(0, crane_capacity + 1):
            for subset in itertools.combinations(range(n_elements), k):
                action = np.zeros(n_elements, dtype=int)
                action[list(subset)] = 1
                action_tensor = torch.tensor(action)
                valid_actions.append(action_tensor) 
        return valid_actions 
    
    def get_minmaxweather(self, weather_forecast):
        min_rain = float('inf')
        min_wind = float('inf')
        min_temp = float('inf')
        max_rain = float('-inf')
        max_wind = float('-inf')
        max_temp = float('-inf')

        for month_data in weather_forecast.values():
            for key, value_list in month_data.items():
                values = [v[0] for v in value_list]
                if key == 'rain':
                    min_rain = min(min_rain, min(values))
                    max_rain = max(max_rain, max(values))
                elif key == 'wind':
                    min_wind = min(min_wind, min(values))
                    max_wind = max(max_wind, max(values))
                elif key == 'temp':
                    min_temp = min(min_temp, min(values))
                    max_temp = max(max_temp, max(values))

        return min_rain, max_rain, min_temp, max_temp, min_wind, max_wind

    def reset(self):
        # Reset the environment to the initial state
        # This will be used at the starting of each new episode.
        self.total_time_steps = 0
        self.current_day = 1  # Current day (increments every 8 hours)
        self.current_time = 0  # Hour within the current day - May be you can update it every 8 hour.

        # Initialize available resources at the start of the simulation
        self.crane_available = self.crane_capacity
        self.labor_available = self.labor_capacity
        self.material_availability = torch.tensor(self.initial_material_availability) 

        self.newly_completed_elements_indices = torch.zeros(self.n_elements, dtype=torch.bool)  
        self.newly_completed_elements = 0 
        
        
        self.weather_conditions = self.get_weather_conditions_for_episode(self.weather_impact_distribution) 

       
        self.node_features = torch.zeros((self.n_elements, 8), dtype=torch.float32)  

        
        self.edge_index = torch.empty((2,0), dtype=torch.long) 
        self.edge_adjacency_matrix = torch.zeros((self.n_elements, self.n_elements), dtype=torch.float32) 

        # So I will maintain three arrays here.
        self.in_progress_elements = np.zeros(self.n_elements) 
        self.completed_elements = np.zeros(self.n_elements)
        self.not_yet_started_elements = np.ones(self.n_elements)

        self.timestamp = self.current_time

        min_weight = np.min(self.weights)
        max_weight = np.max(self.weights)
        min_area = np.min(self.area)
        max_area = np.max(self.area)

        for i in range(self.n_elements):
            element_type = self.element_types[i] 
            resource_vector = self.resource_requirements[element_type] 

            # Sample a duration for each element from the triangular distribution
            min_dur, most_likely_dur, max_dur = self.duration_params[element_type]
            sampled_duration = np.random.triangular(left=min_dur, mode=most_likely_dur, right=max_dur)

            weight = (self.weights[i] - min_weight)/(max_weight - min_weight)
            area = (self.area[i] - min_area)/(max_area - min_area)

            self.node_features[i] = torch.tensor([0, element_type, resource_vector[0],  resource_vector[1], 0, sampled_duration, weight, area], dtype = torch.float32) 

        # Prepare and return the initial observation
        initial_state = self.get_state() 
        return initial_state
    
    def get_weather_conditions_for_episode(self, weather_distribution):
        # Takes the weather distribution for the upcoming days and returns the two dimensional array and convert it to tensor. 
        weather = torch.zeros(len(weather_distribution),3)

        
        for key in weather_distribution.keys():
            sampled_weather = self.sample_weather(key) 

            # Updating the values in the weather array. 
            weather[key-1][0] = sampled_weather['rain'] 
            weather[key-1][1] = sampled_weather['wind']
            weather[key-1][2] = sampled_weather['temp']

        return weather # This will be the tensor of the required shape

    def get_state(self):
        
        node_features = self.node_features.clone().detach().cpu()

        # Normalize the node features
        r1 = node_features[:, 2] # Getting the values of resource1
        r2 = node_features[:, 3] # Getting the values of resource2

        min_r1 = r1.min()
        max_r1 = r1.max()

        min_r2 = r2.min()
        max_r2 = r2.max()

        if (min_r1 != max_r1):
            normalized_r1 = (r1 - min_r1)/(max_r1 - min_r1)

        elif (min_r1 == max_r1):
            normalized_r1 = r1/max_r1

        if (min_r2 != max_r2):
            normalized_r2 = (r2 - min_r2)/(max_r2 - min_r2)

        elif (min_r2 == max_r2):
            normalized_r2 = r2/max_r2

        node_features[:, 2] = normalized_r1
        node_features[:, 3] = normalized_r2

        node_features[:, 4] = node_features[:, 4]/(node_features[:, 4] + self.duration_normalization_param)
        node_features[:, 5] = node_features[:, 5]/(node_features[:, 5] + self.duration_normalization_param)
        
        edge_adjacency_marix = self.edge_adjacency_matrix.clone().detach().cpu()
        
        # Elements remaining to be assembled: Elements with status < 1 are considered "remaining"
        remaining_elements = (self.node_features[:, 0] == 0).float().cpu() # Binary mask: 1 if remaining, else 0
        
        
        resource_availability = torch.tensor([self.crane_available/self.crane_capacity, self.labor_available/self.labor_capacity], dtype=torch.float32).cpu()

        # Material availability: Materials delivered up to the current time step - make sure that initial_material_availability is a pytorch tensor.
        material_availability = self.material_availability.clone().detach().cpu()

        support_relationships = torch.tensor(self.structural_support_relationships).clone().detach().cpu() # So it was an array before. No I converted it into tensor. 
        
        # So you pass the complete weather for the whole episode to the state this time.
        weather_conditions = self.weather_conditions.clone().detach().cpu()

        # Need to do the normalization of weather conditions.
        # So, I am normalizing them as well using the min-max normalization. The minimum and maximum values here are based on  
        weather_conditions[:,0] = (weather_conditions[:,0] - self.min_rain)/(self.max_rain - self.min_rain)
        weather_conditions[:,1] = (weather_conditions[:,1] - self.min_wind)/(self.max_wind - self.min_wind)
        weather_conditions[:,2] = (weather_conditions[:,2] - self.min_temp)/(self.max_temp - self.min_temp)
        
        weather_conditions = weather_conditions.reshape(1,15*3)

        valid_actions = torch.tensor(self.get_valid_action(), dtype = torch.float32).clone().detach().cpu() 

        timestamp = torch.full((self.n_elements,1), self.current_time).clone().detach().cpu() 
        
        # Concatenate everything into the state representation
        state = {
            "node_features": node_features,
            "edge_index": edge_adjacency_marix, 
            "remaining_elements": remaining_elements,
            "resource_availability": resource_availability,
            "material_availability": material_availability,
            "weather_conditions": weather_conditions, # Weather for the current day
            "action_mask": valid_actions, 
            'structure_support_relationships': support_relationships,
            "timestamp" : timestamp # The current timestamp.
        }
    
        return state
    
    def update_status_based_on_step(self, action):
        """
        The function to identify which element is going to finish next
        """

        if np.any(self.in_progress_elements!=0): 
            current_weather_factor = np.ones(self.n_elements)

            
            weather = self.weather_conditions[self.current_day-1] 
            
            remaining_times = []

            for i, a in enumerate(self.in_progress_elements):
                if a == 1:
                    current_weather_factor[i] = self.get_weather_effect_factor(weather, self.node_features[i,5], self.node_features[i,1]) 
                    self.node_features[i,5] = self.node_features[i,5] / current_weather_factor[i] 
                    remaining_times.append(self.node_features[i,5].item()) 
                else:
                    current_weather_factor[i] = 1

            min_remaining_time = min(remaining_times) 

            # If the selected element finishes in the current day itself. 
            if ((self.current_time + min_remaining_time)//8 + 1) <= self.current_day: 
                
                # Advance the current time and current day
                self.current_time = self.current_time + min_remaining_time 
                self.current_day =  int((self.current_time // 8) + 1)

                progress_elements = self.in_progress_elements.copy()

                for i, a in enumerate(progress_elements):
                    if a == 1 and self.node_features[i, 5] == min_remaining_time: # The inprogress element which has the remaining time equal to the minimum remaining time
                        # print("completed")
                        # (i)
                        # Complete the element
                        self.completed_elements[i] = 1
                        self.node_features[i, 0] = 1
                        # Remove it from the in_progress elements
                        self.in_progress_elements[i] = 0
                        self.node_features[i, 5] = 0
                        # Update the duration
                        self.node_features[i, 4] = self.node_features[i, 4] + min_remaining_time
                        self.newly_completed_elements = self.newly_completed_elements + 1
                        self.newly_completed_elements_indices[i] = True

                        self.crane_available = self.crane_available + self.node_features[i][2] # Release resources
                        self.labor_available = self.labor_available + self.node_features[i][3] # Release resources

                        self.labor_assigned = self.labor_assigned + self.node_features[i][3]
                        self.crane_assigned = self.crane_assigned + self.node_features[i][2]
                    elif a == 1 and self.node_features[i, 5] != min_remaining_time:
                        
                        self.node_features[i, 5] = self.node_features[i, 5] - min_remaining_time 
                        self.node_features[i, 4] = self.node_features[i, 4] + min_remaining_time
                        
                        self.node_features[i, 5] = self.node_features[i, 5] * current_weather_factor[i]
                        self.labor_assigned = self.labor_assigned + self.node_features[i][3]
                        self.crane_assigned = self.crane_assigned + self.node_features[i][2]

            else: # So the case when the activity is not finishing in the same day.
                # previous_day = self.current_day
                remaining_time = min_remaining_time 
            
                while (remaining_time!=0):
                        
                    remaining_hours_in_the_day = 8 - (self.current_time % 8) # So you get the remaining hours in the current day here. 
                    self.current_time = self.current_time + remaining_hours_in_the_day 
                    self.current_day = int((self.current_time // 8) + 1)  # Here, you reach the next day
                    
                    weather = self.weather_conditions[self.current_day-1] # You are accessing the value from the array, so keep this in mind to reduce 1 from this.
                    new_weather_factor = np.ones(self.n_elements)
                    remaining_times = [] # You have to re-initialize it here.

                    for i, a in enumerate(self.in_progress_elements):
                        if a == 1:
                            new_weather_factor[i] = self.get_weather_effect_factor(weather, self.node_features[i,5], self.node_features[i,1])  

                            # Updating the duration parameters in the node features
                            self.node_features[i, 5] = self.node_features[i, 5] - remaining_hours_in_the_day  
                            self.node_features[i, 4] = self.node_features[i, 4] + remaining_hours_in_the_day

                            self.node_features[i, 5] = (self.node_features[i, 5] * current_weather_factor[i])/new_weather_factor[i] 

                           
                            remaining_times.append(self.node_features[i,5].item()) # You also get the remaining time here itself
                            self.labor_assigned = self.labor_assigned + self.node_features[i][3]
                            self.crane_assigned = self.crane_assigned + self.node_features[i][2]
                        else:
                            new_weather_factor[i] = 1
 
                    remaining_time = min(remaining_times) # The gives the minimum remaining time.

                    if ((self.current_time + remaining_time)//8 + 1) <= self.current_day: 
                        # Now you check again that whether the activity finishes in the next day.
                        
                        self.current_time = self.current_time + remaining_time 
                        self.current_day =  int((self.current_time // 8) + 1) # It will essentially remain to be the same.
                        
                        for i, a in enumerate(self.in_progress_elements):
                            if a == 1 and self.node_features[i, 5] == remaining_time: 
                                # Complete the element
                                self.completed_elements[i] = 1
                                self.node_features[i, 0] = 1

                                # Remove it from the in_progress elements
                                self.in_progress_elements[i] = 0
                                self.node_features[i, 5] = 0

                                # Update the duration
                                self.node_features[i, 4] = self.node_features[i, 4] + min_remaining_time
                                self.newly_completed_elements = self.newly_completed_elements + 1
                                self.newly_completed_elements_indices[i] = True

                                self.crane_available = self.crane_available + self.node_features[i][2] 
                                self.labor_available = self.labor_available + self.node_features[i][3]

                                self.labor_assigned = self.labor_assigned + self.node_features[i][3]
                                self.crane_assigned = self.crane_assigned + self.node_features[i][2]

                            elif a == 1 and self.node_features[i, 5] != remaining_time:
                                
                                self.node_features[i, 5] = self.node_features[i, 5] - remaining_time 
                                self.node_features[i, 4] = self.node_features[i, 4] + remaining_time  

                                
                                self.node_features[i, 5] = self.node_features[i, 5] * new_weather_factor[i]
                                self.labor_assigned = self.labor_assigned + self.node_features[i][3]
                                self.crane_assigned = self.crane_assigned + self.node_features[i][2] 

                        remaining_time = 0 # This is the termination condition. This has to be reached.

                    else:
                        current_weather_factor = new_weather_factor # This will be the weather factor for the present day.
                    
    def step(self, action, timestep):
        """
        Processes the agent's action, updates the environment's state, and returns the next observation, reward, and done flag.

        Args:
            action (array or tensor): The action taken by the agent.

        Returns:
            next_state (dict): The next observation.
            reward (float): The reward obtained from the action.
            done (bool): Whether the episode has ended.
            info (dict): Additional information.
        """
        prev_labor_assigned = self.labor_available
        prev_crane_assigned = self.crane_available
        self.cost = 0
        prev_time = self.current_time
        self.labor_assigned = 0
        self.crane_assigned = 0
        if isinstance(action, torch.Tensor):
            action = action.numpy() # Convert action to numpy array

        self.newly_completed_elements = 0  
        self.newly_completed_elements_indices = torch.zeros(self.n_elements, dtype=torch.bool) 

        
        if np.any(action != 0): # Check if all the elements in the selected action are not zero. If only this is true, rest of the steps should be followed, right?
            if self.newly_completed_elements != 0:  
                completed_element_indices = self.newly_completed_elements_indices.numpy().copy() # Creating a numpy array here because the tensor might not behave accurately with the enumerate function. 
                for i, a in enumerate (completed_element_indices):
                    if a==True: 
                        for j, b in enumerate (action):
                            if b==1:
                                self.update_edges(i,j)  

            # Basically starting the elements in the action and setting the resource requirements
            for i, a in enumerate(action):
                if a == 1: 
                    self.node_features[i][0] = 0.5 # Start that element
                    # print(f"element {i} is started")  
                    self.crane_available = max(self.crane_available - self.node_features[i][2], 0) # Reduce the crane availability. Limiting it to zero minimum, it can not be negative. 
                    self.labor_available = max(self.labor_available - self.node_features[i][3], 0)# Reduce the labor availability. Limiting it to zero minimum, it can not be negative.

                    # So when we are starting the element, we make the in_progress_elements = 1
                    self.in_progress_elements[i] = 1
                    # For the not_yet_Started_elements, which were all 1 before, make the value for that element to be zero.
                    self.not_yet_started_elements[i] = 0
        
        
        self.update_status_based_on_step(action) #Update the status based on the step taken

        self.update_materials(action)

        done = self.check_termination()
        count = 0

        if timestep!=0 and torch.all(self.material_availability == 0) and done!=True:
            while np.any(self.in_progress_elements != 0.0):
                self.update_status_based_on_step(action)
                self.update_materials(action)
                done = self.check_termination()
                if torch.all(self.material_availability == 0) and done!=True:
                    continue
                else:
                    count = 1
                    break
            if count == 0:
                self.current_day = self.current_day + 1
                self.current_time = 8 - (self.current_time % 8) + self.current_time
                self.update_materials(action)

        reward, constraint_violations, cost = self.calculate_reward(action, self.current_time-prev_time)

        # Now you need to find the elements which are going to finish next
        # find_next_finishing_element()
        
        return self.get_state(), reward, done, self.current_time, constraint_violations, cost

    ######You may need to update it based on the action actually taken.
    def update_materials(self, action):
        delivered_elements = []
        for day in range(1, self.current_day + 1):
            if day in self.material_availability_schedule:
                delivered_elements = self.material_availability_schedule[day]
                self.material_availability[delivered_elements] = 1  # Set material availability to 1 for delivered elements
        
        # Subtract elements that are already started (progress > 0 means the element has started)
        started_elements = self.node_features[:, 0] > 0
        self.material_availability[started_elements] = 0  # Mark elements that are already started as unavailable
    
    def update_edges(self, source, target):
        self.edge_adjacency_matrix[source][target] = 1


    def sample_weather(self, day):
        # So here, from the weather distribution, it is basically sampling the weather data for each day from the weather distribution.
        """
        This function samples the weather for a given day based on the probability distributions.
        """
        weather = {}
        
        # Sample the rain condition based on the probability distribution for rain
        rain_probs = self.weather_forecast[day]['rain']
        rain = random.choices([r[0] for r in rain_probs], [r[1] for r in rain_probs])[0]
        
        # Sample the wind condition based on the probability distribution for wind
        wind_probs = self.weather_forecast[day]['wind']
        wind = random.choices([w[0] for w in wind_probs], [w[1] for w in wind_probs])[0]
        
        # Sample the temperature condition based on the probability distribution for temperature
        temp_probs = self.weather_forecast[day]['temp']
        temp = random.choices([t[0] for t in temp_probs], [t[1] for t in temp_probs])[0]
        
        weather['rain'] = rain
        weather['wind'] = wind
        weather['temp'] = temp
        
        return weather

    # Need to improve this. For now its just simple assumption.
    def apply_weather_effects(self, weather, duration, activity_type): 
        sampled_weather = weather 
        temp, wind, rain = sampled_weather['temp'], sampled_weather['wind'], sampled_weather['rain']

        # Calculate weather factor based on sampled weather conditions
        # Need a distribution of temperature, wind, and rain. For now, let's keep it like this. 
        temp_factor = max(0.8, 1.0 - (abs(temp - 25) / 50))  # Ideal temp is 25°C
        wind_factor = max(0.7, 1.0 - (wind / 100))  # High wind slows progress
        rain_factor = max(0.6, 1.0 - (rain / 50))  # Heavy rain slows progress

        # Final weather factor is the product of all individual factors
        weather_factor = min(temp_factor, wind_factor, rain_factor)

        # Adjust remaining time based on weather factor
        return duration/weather_factor # So basically you need to increase the remaining duration based on the weather condition.
    
    # Need to improve this. 
    def get_weather_effect_factor(self, weather, duration, activity_type):
        sampled_weather = weather 
        temp, wind, rain = sampled_weather[2].item(), sampled_weather[1].item(), sampled_weather[0].item()

        temp_factor = 1
        if wind == 9.8:
            wind_factor = 0.5
        else:
            wind_factor = 1
        rain_factor = 1

        # Final weather factor is the product of all individual factors
        weather_factor = min(temp_factor, wind_factor, rain_factor)

        # Adjust remaining time based on weather factor
        return weather_factor # So basically you need to increase the remaining duration based on the weather condition.  

    def get_valid_action(self):
        all_possible_actions = self.action_space # This gives all the possible actions at a particular step.

        valid_action_mask = np.ones(len(all_possible_actions), dtype = int)

        # Initialize a list to store valid actions
        valid_actions = []
        invalid_actions = set()

        for j, action in enumerate(all_possible_actions):
            # First, check if any element in the action is already completed
            for i, a in enumerate(action):
                if a == 1:
                    # Check if the element is already completed
                    if self.node_features[i, 0] > 0:  # Progress > 0 means it's already started
                        invalid_actions.add(j) # Append the index of the action in j.
                        break

        # Second check the resource availability criteria
        for j, action in enumerate(all_possible_actions):
            if j in invalid_actions:
                continue
            cranes_needed = 0
            labor_needed = 0
            material_needed = []
            for i, a in enumerate(action):
                if a == 1 :  # If the action proposes to start element i
                    cranes_needed = cranes_needed + self.node_features[i, 2]
                    labor_needed = labor_needed + self.node_features[i, 3]
                    material_needed.append(i)

            if self.crane_available < cranes_needed or self.labor_available < labor_needed: # Check if the available values is being updated properly.
                invalid_actions.add(j)
                continue

            # For the selected elements in material_needed, ensure they have been delivered by the current day
            for element in material_needed: # So that will essentially fetch the index.
                if self.material_availability[element] == 0: # Using self.material_availability here directly.
                    # If the element has not been delivered by the current time step, the action is invalid
                    invalid_actions.add(j)
                    break

        # Third, you need to check the structural support relationships.
        for j, action in enumerate(all_possible_actions):
            if j in invalid_actions:
                continue
            if torch.all(action == 0):
                invalid_actions.add(j)
                continue
            for i, a in enumerate(action):
                if a == 1:  # If the action proposes to start element i
                    for k in range(self.n_elements):
                        if self.structural_support_relationships[k][i] == 1: # This will mean element i is supported by element k i.e. element k should complete first.
                            if self.node_features[k, 0] != 1:  # If element k is not finished yet
                                invalid_actions.add(j)
                                break
                if j in invalid_actions:
                    break

        # Returning the indices of the valid actions only.                            
        for j in invalid_actions:
            valid_action_mask[j] = 0

        # Now you need to check that if all the actions are masked, you can select only that action which starts no elements
        if np.all(valid_action_mask == 0):
            # Find the action which has all zeros.
            for i, action in enumerate(all_possible_actions):
                if torch.all(action == 0): # So this is the action where no element can be started.
                    valid_action_mask[i] = 1

        return valid_action_mask 

    
    def calculate_reward(self, action, deltat): # This will stay the same as the action in this case is also a vector.
        
        constraint_violations_numbers = 0
        crane_cost = self.crane_assigned*deltat*(self.daily_crane_cost/8)
        labor_cost = self.labor_assigned*deltat*(self.daily_labor_cost/8)
        direct_cost = crane_cost + labor_cost

        indirect_cost = (self.daily_indirect_cost/8) * deltat # Again, taking the cost on per time basis. 
        total_cost = direct_cost + indirect_cost
        
        # Time and progress-based reward
        time_penalty = -5000 #Again, give this penalty based on the hours.

        # Need to add a positive reward for completion of each individual component
        action_finishing_reward = 0

        # Add reward for elements that have just finished during this time step
        for i in range(self.n_elements):
            if self.newly_completed_elements_indices[i]:  # If the element was completed in this step
                action_finishing_reward += 1000  # Reward for completing an individual element

        # Need to add a positive reward for starting a new component
        action_starting_reward = 0
        for i, a in enumerate(action):
            if a == 1:  # If element is selected for assembly
                action_starting_reward = action_starting_reward + 1000 # This should be fine because you are looking at the starting of the elements, which can come directly from the action you took. 

        # Need to add a large positive reward if all the activities are completed. 
        all_completed_reward = 0
        if torch.all(self.node_features[:, 0] == 1):  # Check if all elements are fully completed
            all_completed_reward = 20000  # Larger reward for completing all activities

        started_elements = (self.node_features[:, 0] > 0).nonzero(as_tuple=True)[0]  # Get indices of started elements

            # Identify elements that are started in the current action (action == 1)
        current_action_started_elements = (torch.tensor(action) == 1).nonzero(as_tuple=True)[0]  # Get indices of elements started in the current action
        remaining_started_elements = started_elements[~torch.isin(started_elements, current_action_started_elements)] # This will find out indices of the elements that were started before current action

        previously_selected_weights = [] # Initializing the previously selected weights as an empty array
        for i in range(self.n_elements):
            if i in remaining_started_elements:
                previously_selected_weights.append(self.weights[i])
 
        weight_penalty = 0
        for i, a in enumerate(action):
            if a == 1:  # If the element is selected for assembly
                # Compare the weight of the newly selected element with the previously selected elements
                if previously_selected_weights:  # Ensure there are previously selected elements to compare
                    min_previous_weight = min(previously_selected_weights)  # Get the minimum weight of previously selected elements
                    if self.weights[i] > min_previous_weight:  # If the newly selected element is heavier
                        weight_penalty -= 2000  # Penalty for selecting a heavier element
                        constraint_violations_numbers = constraint_violations_numbers + 1

        # Negative reward due to interference
        interference_penalty = 0
        for i, a in enumerate(action):
            if a == 1:  # If the element is selected for assembly
                
                adjacent_elements = [j for j in remaining_started_elements if self.adjacency_matrix[i][j] == 1] # So the adjacent elements here will be the previously installed adjacent elements to the current selected element.

                # Find interference vectors for the current element and adjacent elements
                interference_vectors = []
                for j in adjacent_elements:
                    interference_vectors.append(self.interference_matrix[i][j])  # Get interference directions as a set...SO you sttore the individual interference vectors here.

                # Compute the intersection of all interference vectors
                if interference_vectors:  # Ensure there are interference vectors to process
                    intersection = np.maximum.reduce(interference_vectors)  # Find the common directions - Should be union I think.
                    interference_penalty -= len(intersection) * 2000  # Apply penalty based on the number of constrained directions
                    constraint_violations_numbers = constraint_violations_numbers + len(intersection)

        previously_selected_areas = [] # Initializing the previously selected weights as an empty array
        for i in range(self.n_elements):
            if i in remaining_started_elements:
                previously_selected_areas.append(self.module_area[i]) # Getting the previously selected areas here. 

        area_penalty = 0
        for i, a in enumerate(action):
            if a == 1:  # If the element is selected for assembly
                if previously_selected_areas:  # Ensure there are previously selected elements to compare
                    min_previous_area = min(previously_selected_areas)  # Get the minimum area of previously selected elements
                    if self.module_area[i] > min_previous_area:  # If the newly selected element has a larger area
                        area_penalty -= 2000  # Penalty for selecting an element with larger area
                        constraint_violations_numbers = constraint_violations_numbers + 1

        # Combine the reward
        reward = -1*total_cost + time_penalty + action_starting_reward + action_finishing_reward + all_completed_reward + weight_penalty + area_penalty + interference_penalty
        return reward, constraint_violations_numbers, total_cost
 
    def render(self, mode='human'):
        # Function - Prints the current day and time, Displays available resources, Shows current weather conditions, Lists each element's status and associated attributes
        """
        Renders the current state of the environment.

        Args:
            mode (str): The mode in which to render the environment. Defaults to 'human'.
        """
        print("\n===== Precast Scheduling Environment =====")
        print(f"Current Day: {self.current_day}, Current Time: {self.current_time:.2f} hours")
        print(f"Available Cranes: {self.crane_available}/{self.crane_capacity}")
        print(f"Available Laborers: {self.labor_available}/{self.labor_capacity}")
        print(f"Weather Conditions: {self.weather_conditions}")
        print("\nElement Status:")
        print("Index | Status       | Type | Crane Needed | Labor Needed | Actual Dur | Remaining Dur")
        status_dict = {0.0: 'Not Started', 0.5: 'In Progress', 1.0: 'Completed'}
        for i in range(self.n_elements):
            status_value = self.node_features[i, 0].item()
            status = status_dict.get(status_value, 'Unknown')
            element_type = int(self.node_features[i, 1].item())
            crane_needed = int(self.node_features[i, 2].item())
            labor_needed = int(self.node_features[i, 3].item())
            actual_dur = self.node_features[i, 4].item()
            remaining_dur = self.node_features[i, 5].item()
            print(f"{i:5d} | {status:12} | {element_type:4d} | {crane_needed:12d} | {labor_needed:12d} | {actual_dur:10.2f} | {remaining_dur:13.2f}")
        print("==========================================\n")


    def seed(self, seed=None): 
        """
        Sets the seed for the environment's random number generators.

        Args:
            seed (int, optional): The seed value to use. If None, a random seed is chosen.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return [seed]

    def check_termination(self):
        if np.all(self.completed_elements) == 1:
            return True
        else:
            return False
        
# This is the time encoding module to encode the time step. 
class TimeEncode(nn.Module):
    def __init__(self, time_embedding_dim):
        super(TimeEncode, self).__init__()
        self.time_embedding_dim = time_embedding_dim # The output dimension of the time embedding
        self.freq = nn.Parameter(torch.linspace(0.1, 1.0, time_embedding_dim//2)) # A learnable parameter vector of length that defines a set of frequencies. It is initialized with a linearly spaced set of values b/w 0.1 and 1.
        # self.phase = nn.Parameter(torch.zeros(time_dim)) # 
        self.sqrt_dim = math.sqrt(time_embedding_dim)

    def forward(self, timestamps):
        # [N, 1] -> [N, time_dim]
        timestamp = timestamps * self.freq # So basically you multiply the timestamp with the frequency, which can be a learnable parameter as well. 

        # Computes cosines and sines
        time_features = torch.cat([torch.cos(timestamp), torch.sin(timestamp)], dim = -1)

        # Normalize embeddings
        return time_features/self.sqrt_dim
        
        # This returns the time features of dimensions number of nodes * time_embedding_dim - Creates a time embedding for each node.

class TemporalGraphNetwork(nn.Module):
    def __init__(
        self,
        node_feat_dim,
        time_feat_dim,
        memory_dim,
        embedding_dim,
        n_heads,
        mlp_hidden_dim,
        dropout=0,
    ):
        super(TemporalGraphNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Time embedding generation
        self.time_encoder = TimeEncode(time_feat_dim)

        # Memory initialization
        self.memory_dim = memory_dim
        self.memory = torch.zeros(0, memory_dim) # I think this is basically a useless line

        # GRU for memory updates
        self.memory_updater = nn.GRUCell(input_size=embedding_dim, hidden_size=memory_dim)

        self.linear_transformation = nn.Sequential(nn.Linear(node_feat_dim, 2 * node_feat_dim),
                                                   nn.ReLU()) # So this is being used for converting the node features from 8 to 16. 

        
        # First layer: Multi-head attention with MLP
        self.attention_layer1 = GATConv(
            in_channels = memory_dim + 2 * node_feat_dim + time_feat_dim,
            out_channels = 2 * embedding_dim // n_heads,
            heads = n_heads,
            dropout = dropout,
            concat = True,
            edge_dim = 0  # No edge features in this case
        )

        self.mlp1 = nn.Sequential(
            nn.Linear(2 * embedding_dim, 2 * mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * mlp_hidden_dim, 2 * embedding_dim)
        )

        # Second layer: Multi-head attention with MLP
        self.attention_layer2 = GATConv(
            in_channels= 2 * embedding_dim + time_feat_dim, # Need to add time embedding as well based on the original TGAT architecture.
            out_channels= 2 * embedding_dim // n_heads,
            heads=n_heads,
            dropout=dropout,
            concat = True,
            edge_dim=0
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(2* embedding_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, embedding_dim)
        )

    def initialize_memory(self, num_nodes):
        """ Initialize memory for all nodes. """
        self.memory = torch.zeros(num_nodes, self.memory_dim, device = 'cuda')


    def update_memory(self, src, dst, timestamps, new_node_embeddings):
        """ Update memory for nodes after processing a timestep using GRU. """
        # Update source nodes
        src_memory = self.memory[src]
        src_new_embeddings = new_node_embeddings[src]
        self.memory[src] = self.memory_updater(src_new_embeddings, src_memory)

        # Update destination nodes
        dst_memory = self.memory[dst]
        dst_new_embeddings = new_node_embeddings[dst]
        self.memory[dst] = self.memory_updater(dst_new_embeddings, dst_memory)

    def forward(self, node_features, edge_index, timestamps):
        """
        Args:
            node_features: Tensor of shape [num_nodes, node_feat_dim]
            edge_index: Tensor of shape [2, num_edges] representing the graph connectivity
            timestamps: Tensor of shape [num_nodes, 1] representing the time encoding for each node

        Returns:
            Tensor of shape [num_nodes, embedding_dim] representing the node embeddings
        """
        # Encode timestamps
        time_features = self.time_encoder(timestamps)

        x = self.linear_transformation(node_features)

        # Concatenate node features, time features, and memory
        x = torch.cat([x, time_features, self.memory], dim=1)

        # First layer of attention + MLP
        x = self.attention_layer1(x, edge_index)
        x = self.mlp1(x)

        x = torch.cat([x, time_features], dim = 1)
        # Second layer of attention + MLP
        x = self.attention_layer2(x, edge_index)
        x = self.mlp2(x)

        return x

class ActorCritic(nn.Module):
    def __init__(self, 
                 n_elements, 
                 node_feature_dim, 
                 gcn_hidden_dim,
                 gcn_output_dim,
                 material_availability_feature_size,
                 resource_availability_feature_size,
                 remaining_elements_feature_size,
                 structural_support_feature_size,
                 weather_feature_size,
                 hidden_dim, 
                 actor_hidden_dim,
                 critic_hidden_dim,
                 num_heads,
                 time_embedding_dim,
                 num_actions,
                 dropout_rate,  
                 device = None): # Not giving any default value for now. I'll update it later on. 
        super(ActorCritic, self).__init__() # This basically calls the __init()__ method of the superclass.
        self.n_elements = n_elements

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate TGN
        self.gcn = TemporalGraphNetwork(node_feat_dim = node_feature_dim, 
                                        time_feat_dim = time_embedding_dim,
                                        memory_dim = 8,
                                        embedding_dim = gcn_output_dim,
                                        n_heads = num_heads,
                                        mlp_hidden_dim = gcn_hidden_dim,
                                        dropout = dropout_rate).to(device)

        # Other features
        self.additional_feature_size = material_availability_feature_size + resource_availability_feature_size + remaining_elements_feature_size + weather_feature_size

        # Separate Embeddings with Linear + ReLU
        self.material_embedding = nn.Sequential(
            nn.Linear(n_elements, material_availability_feature_size),
            nn.ReLU()
        )

        self.resource_embedding = nn.Sequential(
            nn.Linear(2, resource_availability_feature_size),  # Assuming 2 resources
            nn.ReLU()
        )

        self.remaining_embedding = nn.Sequential(
            nn.Linear(n_elements, remaining_elements_feature_size),
            nn.ReLU()
        )

        self.structural_support_embedding = nn.Sequential(
            nn.Linear(n_elements * n_elements, structural_support_feature_size),
            nn.ReLU()
        )

        # Need to add weather embedding as well to make the agent know the weather of the current day on which the decision was taken. 
        self.weather_embedding = nn.Sequential(
            nn.Linear(15 * 3, weather_feature_size),
            nn.ReLU()
        )

        combined_embedding = gcn_output_dim + material_availability_feature_size + resource_availability_feature_size + remaining_elements_feature_size + structural_support_feature_size + weather_feature_size


        self.shared_fc1 = nn.Linear(combined_embedding, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Actor network - Assuming two Linear layers for now
        self.actor_fc1 = nn.Linear(hidden_dim, actor_hidden_dim)
        self.actor_fc2 = nn.Linear(actor_hidden_dim, num_actions)
        
        # Critic Network - Assuming two Linear layers for now
        self.critic_fc1 = nn.Linear(hidden_dim, critic_hidden_dim)
        self.critic_fc2 = nn.Linear(critic_hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        

    def forward(self, data, timestep, mem_state, is_training):
        """
        Forward pass of the actor critic network.
        Args: observation (dict): Observation from the environment.
        
        Returns: action_probs: Probabilities of each action.
        state_value : Estimated value of the current state.clear
        
        """

        # Extract node features and edge_index
        node_features = data.x                  # [total_nodes_in_batch, node_feature_dim]
        edge_index = data.edge_index            # [2, total_edges_in_batch]
        batch = data.batch                      # [total_nodes_in_batch]
        timestamps = data.timestamps            # [this should be total_nodes_in_batch * 1 - one time for each node.]

        if is_training == True:
            if timestep == 0:
                self.gcn.initialize_memory(node_features.size(0)) # The size of initial memory should be equal to the number of nodes. 
            else:
                self.gcn.memory = self.updated_memory.clone()

        if is_training == False:
            # During the update time, use the collected memory states instead of updated memory states. 
            self.gcn.memory = mem_state

        graph_memory_state = self.gcn.memory.clone().detach() # This gets the graph memory state before taking that particular step.

        node_embedding = self.gcn(node_features, edge_index, timestamps) # This outputs the node level embedding of size [n_elements, output_dimension]
        
        if is_training == True:
            # While collecting trajectories, Update the memory of the nodes so that it can be used next time.
            self.gcn.update_memory(src = edge_index[0], dst = edge_index[1], timestamps=timestamps, new_node_embeddings=node_embedding)
            self.updated_memory = self.gcn.memory.clone().detach() # This will store the updated memory after each rollout step

        graph_embedding = global_mean_pool(node_embedding, batch)  # [gcn_output_dim] - Global mean pooling to get the graph level embedding -  Do you really need this?

        # Separate Embeddings
        material_embedded = self.material_embedding(data.material_availability)  # [batch_size, material_availability_feature_size]
        resource_embedded = self.resource_embedding(data.resource_availability)  # [batch_size, resource_availability_feature_size]
        remaining_embedded = self.remaining_embedding(data.remaining_elements)  # [batch_size, remaining_elements_feature_size]
        weather_embedded = self.weather_embedding(data.weather_conditions)      # [batch_size, remaining_element_feature_size]
        structural_support_embedded = self.structural_support_embedding(data.structural_support_relationships)# [batch_size, 400]
        action_mask = data.action_mask

        # Combine all features
        combined_embedding = torch.cat([
            graph_embedding, #GCN embedding
            material_embedded, # material_availability_embedding
            resource_embedded, # resource_availability_embedding
            remaining_embedded, # remaining_element_feature_size
            structural_support_embedded,
            weather_embedded # Structura_support_feature_size
        ], dim=1) # We want to add them along the dimension 0.

        # Passing through shared layers 
        x = F.relu(self.shared_fc1(combined_embedding))
        x = F.relu(self.shared_fc2(x))

        # Actor network forward pass
        actor_x = F.relu(self.actor_fc1(x))
        action_logits = self.actor_fc2(actor_x)
        large_negative = -1e12
        masked_logits = action_logits + (1-action_mask)*large_negative
        action_probs = F.softmax(masked_logits, dim=-1) # Predicting the probabilities of the actions

        # Critic network forward pass
        critic_1 = F.relu(self.critic_fc1(x))
        state_value = self.critic_fc2(critic_1) # Predicting the value function

        return action_probs, state_value, graph_memory_state
    
    def adjacency_to_edge_index(self, edge_adjacency_matrix):
        "To convert the adjacency matrix of the edges to the edge representation needed for the GCN."

        # You need to get the indices where adjacency_matrix is 1
        # Returns: edge_index: Edge indices [2, num_edges]
        if edge_adjacency_matrix.dim() != 2:
            raise ValueError(f"Expected a 2D adjacency matrix, but got {edge_adjacency_matrix.dim()}D.")

        edge_index = edge_adjacency_matrix.nonzero(as_tuple = False).t().contiguous()
        return edge_index
    
class Agent: 
    def __init__(self, 
                 actor_critic_model, 
                 action_space, # The whole action space should be passed. Why would you pass only valid actions here.
                 lr, 
                 gamma,
                 entropy_coef,
                 clip_epsilon, 
                 gae_lambda, # Need to see how to decide this value.
                 K_epochs,
                 minibatch_size,
                 batch_size,
                 device = None):
        """
        Initializing the Agent with the ActorCritic model and optimizer.
        
        actor_critic_model (nn.Module) : The ActorCritic model.
        device : Device to run the model on
        entropy_coef : Coefficient for entropy regularization"""

        # Automatically select device if not provided
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model = actor_critic_model.to(self.device) # When the parent module is moved to a device, all the submodules should automatically be moved. 

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4) # Adding the weight decay here automatically adds the L2 regularization. 

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.mini_batch_size = minibatch_size
        self.clip_epsilon = clip_epsilon # This is basically the epsilon value that is used for clipping the updates

        self.action_space = action_space

        # Memory to store experiences
        self.reset_memory()

    def reset_memory(self):
        """
        Resets the memory buffers.
        """
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.entropies = []
        self.states = [] # Optional, storing stats
        self.actions = [] # Optional, storing actions
        self.input_mem_state = [] # Storing the input memory state for that particular state.

    def create_data_object(self, state):
        
        node_features = state['node_features']
        edge_index = self.adjacency_to_edge_index(state['edge_index'])   
        
        remaining_elements = state['remaining_elements'].unsqueeze(0)
        
        resource_availability = state['resource_availability'].unsqueeze(0)
        
        material_availability = state['material_availability'].unsqueeze(0)
        
        weather_conditions = state['weather_conditions']
        
        action_mask = state['action_mask'].unsqueeze(0)
        
        structural_support_relationships = state['structure_support_relationships'].flatten(start_dim=0).unsqueeze(0)

        timestamp = state['timestamp']
        
        data = Data(
            x=node_features,
            edge_index=edge_index,
            remaining_elements=remaining_elements,
            resource_availability=resource_availability,
            material_availability=material_availability,
            weather_conditions=weather_conditions,
            action_mask=action_mask,
            structural_support_relationships=structural_support_relationships,
            timestamps = timestamp
        )
        
        return data
    
    def create_batched_data(self, states):
        data_list = [self.create_data_object(state) for state in states]
        batched_data = Batch.from_data_list(data_list)
        batched_data = batched_data.to(self.device)  # Move to GPU or CPU as needed
        return batched_data


    def adjacency_to_edge_index(self, edge_adjacency_matrix):
        "To convert the adjacency matrix of the edges to the edge representation needed for the GCN."

        # Returns: edge_index: Edge indices [2, num_edges]
        if edge_adjacency_matrix.dim() != 2:
            raise ValueError(f"Expected a 2D adjacency matrix, but got {edge_adjacency_matrix.dim()}D.")

        edge_index = edge_adjacency_matrix.nonzero(as_tuple = False).t().contiguous()
        return edge_index

    def select_action(self, observation, timestep):
        """
        Selects an action based on the current observation. 
        
        observation (dict) : Current state observation from the environment.
        
        Returns : action (numpy.ndarray) - Selected action. """
        self.model.eval()
        # selected_actions = torch.zeros(n_elements, dtype=torch.float32).to(self.device)

        with torch.no_grad():

            # Convert single state to Data object
            data = self.create_data_object(observation).to(self.device)

            # Batch the single Data object
            batched_data = Batch.from_data_list([data])
            mem_state = [] 
            
            action_probs, state_value, graph_mem_state = self.model(batched_data, timestep, mem_state, is_training = True) 

            dist = Categorical(action_probs) # Creating a categorical distribution over the actions probabilities

            # Sampling an action
            action = dist.sample() # action here will be the index of the selected array. 

            # Get log probability and entropy
            log_prob = dist.log_prob(action) # Gets the log_probability of the selected action. 
            entropy = dist.entropy() # gets the entropy of the distribution.

            selected_action = self.action_space[action.item()] 

            # Storing in the memory

            self.log_probs.append(log_prob)
            self.entropies.append(entropy)
            self.values.append(state_value)
            self.actions.append(action) 
            self.states.append(observation)
            self.input_mem_state.append(graph_mem_state) 

        return selected_action 
    
    def store_transition(self, reward, done):
        # As the agent interacts with the environment, at each time step, it stores the rewards received, and whether that particular state was terminal state or not. 
        """
        Stores the reward and done flag for the current timestep.
        
        Args:
            reward (float): Reward received after taking the action.
            done (bool): Flag indicating if the episode has ended.
        """
        self.rewards.append(torch.tensor([reward], dtype=torch.float32).to(self.device))
        self.dones.append(torch.tensor([done], dtype=torch.float32).to(self.device))

    def compute_returns_and_advantages(self, last_value, first_state_index, last_state_index, done):

        """
        Computes discounted returns and advantages.

        Args:
            last_value (torch.Tensor): The value of the last state.
            done (bool): Flag indicating if the episode has ended.

        Returns:
            returns (torch.Tensor): Tensor of discounted returns.
            advantages (torch.Tensor): Tensor of advantages.
        """
        rewards = self.rewards[first_state_index:last_state_index + 1]
        dones = self.dones[first_state_index:last_state_index + 1]
        # values = self.values[first_state_index:last_state_index] + [last_value.view(1,1)]
        values = [v.squeeze().view(-1) if v.ndim > 1 else v for v in self.values[first_state_index:last_state_index+1]]
        last_value = last_value.view(-1)
        values.append(last_value)

        returns = []
        advantages = []
        gae = 0

        for step in reversed(range(len(rewards))): 
            delta = rewards[step] + self.gamma * values[step+1] * (1 - dones[step]) - values[step] 
            gae = delta + self.gamma * self.gae_lambda * gae * (1-dones[step]) 
            advantages.append(gae) 
            returns.append(gae + values[step])

        # Reverse the lists to maintain the original order
        returns = torch.cat(returns[::-1]).detach().view(-1, 1)
        advantages = torch.cat(advantages[::-1]).detach().view(-1, 1)
        # advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)
    
        return returns, advantages
    
    def update(self):
        "Updating the policy and value network based on collected experiences"
        "Use the PPO clipped surrogate objectives with minibatch updates"

        # Set the model to training mode
        self.model.train()

        # Check if there are enough episodes
        if len(self.rewards) == 0:
            print("No experiences to update")
            return
        
        total_timesteps = len(self.rewards)

        # Initialize lists for first and last states
        first_states = []
        last_states = []

        # Iterate through the batch of dones
        for i in range(len(self.dones)):
            # Always add the first state
            if i == 0:
                first_states.append(i)
            
            # Add to last states if `dones[i]` is True
            if self.dones[i] == True:
                last_states.append(i)
                
                # Add the next state to first_states if it exists
                if i + 1 < len(self.dones):
                    first_states.append(i + 1)
            
            # Ensure the last state is added to `last_states`
            if i == len(self.dones) - 1 and i not in last_states:
                last_states.append(i)

        # Initialize returns and advantages as empty tensors
        returns = torch.tensor([], device=self.device)
        advantages = torch.tensor([], device=self.device)

        # Computing returns segmentwise
        for i in range(0, len(last_states)):
            with torch.no_grad():
                if self.dones[last_states[i]] == True:
                    last_value = torch.tensor(0.0, device='cuda', dtype=torch.float32).squeeze()
                else:

                    last_state_data_list = self.create_data_object(self.states[last_states[i]]) # Getting the last state of the segment
                    last_state_batched_data = Batch.from_data_list([last_state_data_list]).to(self.device)
                    last_state_memory_state = self.input_mem_state[last_states[i]] # That should give the memory state of last state in the segment
                    last_value_action_probs, last_value, last_mem_state = self.model(last_state_batched_data, 1000, last_state_memory_state, is_training = False)
                    last_value = last_value.squeeze()
            segmental_returns, segmental_advantages = self.compute_returns_and_advantages(last_value, first_states[i], last_states[i], done = self.dones[last_states[i]])

            # Append results to overall tensors
            returns = torch.cat([returns, segmental_returns])
            advantages = torch.cat([advantages, segmental_advantages])

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert lists to tensors
        log_probs = torch.stack(self.log_probs).to(self.device)  # [N]
        entropies = torch.stack(self.entropies).to(self.device)  # [N]
        values = torch.stack(self.values).squeeze().to(self.device)  # [N]
        actions = torch.stack(self.actions).to(self.device)  # [N]
        memory_states = torch.stack(self.input_mem_state).to(self.device) # The input memory states of TGN.

        indices = torch.randperm(total_timesteps).to(self.device)

        # Determine the number of mini-batching
        num_mini_batches = total_timesteps // self.mini_batch_size
        if num_mini_batches == 0:
            num_mini_batches = 1 # Atleast one batch

        for epoch in range(self.K_epochs): # So you run it for each epoch
            for i in range(num_mini_batches):
                start = i*self.mini_batch_size
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]

                # Selecting mini_batch data
                # So this will get the data corresponding to the indices selected in the mini batches
                mini_log_probs = log_probs[batch_indices]
                mini_entropies = entropies[batch_indices]
                mini_values = values[batch_indices]
                mini_returns = returns[batch_indices]
                mini_advantages = advantages[batch_indices]
                mini_actions = actions[batch_indices]
                mini_mem_states = memory_states[batch_indices].reshape(-1, 8)

                # Getting the data list
                minibatch = [self.states[j] for j in batch_indices]
                data_list = [self.create_data_object(state) for state in minibatch]

                # Creating the batch of the Data objects
                batched_data = Batch.from_data_list(data_list).to(self.device)

                # Forward pass through the model
                action_probs, state_values, memory_state = self.model(batched_data, 1000, mini_mem_states, is_training = False)  

                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(mini_actions.squeeze())  
                entropy = dist.entropy()

                # Compute the ratio (r_t)
                ratios = torch.exp(new_log_probs.unsqueeze(1) - mini_log_probs.detach())

                # Compute the surrogate losses
                surr1 = ratios * mini_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mini_advantages

                # Compute the actor loss
                actor_loss = -torch.min(surr1, surr2).mean()
                # print(actor_loss)

                # Compute the critic loss
                critic_loss = F.mse_loss(state_values.squeeze(), mini_returns.squeeze())
                # print(critic_loss)

                # Compute the entropy loss
                entropy_loss = -self.entropy_coef * entropy.mean()

                # Total loss
                loss = actor_loss + 0.5 * critic_loss + entropy_loss

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)  # Gradient clipping for stability
                self.optimizer.step()
        print(f"Actor loss at epoch {epoch} is {actor_loss}")
        print(f"Critic loss at epoch {epoch} is {critic_loss}")
        print(f"Total loss at epoch {epoch} is {loss}")

        # Clear memory after updating
        self.reset_memory()

def getActionSpaceLength(n_elements, crane_capacity):
    """
    Generate all valid actions where the nmber of selected elements is less than crane_capacity
    
    Returns : list of binary arrays representing valid actions"""
    valid_actions = []
    for k in range(0, crane_capacity + 1):
        for subset in itertools.combinations(range(n_elements), k):
            action = np.zeros(n_elements, dtype=int)
            action[list(subset)] = 1
            valid_actions.append(action)
    return valid_actions, len(valid_actions) 

def plot_rewards_rewards(episode_rewards, num_episodes, save_path, computed_duration, average_reward, average_durations):
    plt.figure(figsize=(24,12))
    episodes = np.arange(1, num_episodes+1)

    # First axis for episode rewards
    ax1 = plt.gca()
    line1 = ax1.plot(episodes, episode_rewards, label = 'Episode Reward', color = 'blue', zorder = 1)
    ax1.set_xlabel ('Episode')
    ax1.set_ylabel ('Total Reward', color = 'blue')
    ax1.tick_params(axis = 'y', labelcolor = 'blue')

    avg_episodes = np.arange(100, num_episodes+1, 100) # Since the average episodes are sampled at every 10 episodes.
    line3 = ax1.plot(avg_episodes, average_reward, label='Average Reward (100-episode avg)', color='red', linestyle='--', zorder = 2)

    lines = line1 + line3
    labels = [line.get_label() for line in lines]
    plt.title('Reward per Episode')
    plt.legend(lines, labels, loc='best')

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_rewards_durations(episode_rewards, num_episodes, save_path, computed_duration, average_reward, average_durations):
    plt.figure(figsize=(24,12))
    episodes = np.arange(1, num_episodes+1)

    # Second axis for computed durations
    ax1 = plt.gca()
    line2 = ax1.plot(episodes, computed_duration, label='Computed Duration', color='red', zorder = 1)
    ax1.set_ylabel('Computed Duration', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    avg_episodes = np.arange(100, num_episodes+1, 100) # Since the average episodes are sampled at every 10 episodes.
    line4 = ax1.plot(avg_episodes, average_durations, label='Average Duration (100-episode avg)', color='blue', linestyle='--', zorder = 2)

    lines = line2 + line4
    labels = [line.get_label() for line in lines]
    plt.title('Computed Duration')
    plt.legend(lines, labels, loc='best')

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_rewards_violations(episode_rewards, num_episodes, save_path, episode_constraint_violations, average_reward, avg_constraint_violations):
    plt.figure(figsize=(24,12))
    episodes = np.arange(1, num_episodes+1)

    # Second axis for computed durations
    ax1 = plt.gca()
    line2 = ax1.plot(episodes, episode_constraint_violations, label='Computed Duration', color='red', zorder = 1)
    ax1.set_ylabel('Computed Violations', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    avg_episodes = np.arange(100, num_episodes+1, 100) # Since the average episodes are sampled at every 10 episodes.
    line4 = ax1.plot(avg_episodes, avg_constraint_violations, label='Average Constraint Violations (100-episode avg)', color='blue', linestyle='--', zorder = 2)

    lines = line2 + line4
    labels = [line.get_label() for line in lines]
    plt.title('Constraint Violations')
    plt.legend(lines, labels, loc='best')

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_rewards_cost(episode_rewards, num_episodes, save_path_cost, episode_cost, avg_rewards, avg_episode_cost):
    plt.figure(figsize=(24,12))
    episodes = np.arange(1, num_episodes+1)

    # Second axis for computed durations
    ax1 = plt.gca()
    line2 = ax1.plot(episodes, episode_cost, label='Computed Cost', color='red', zorder = 1)
    ax1.set_ylabel('Computed Cost', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    avg_episodes = np.arange(100, num_episodes+1, 100) # Since the average episodes are sampled at every 10 episodes.
    line4 = ax1.plot(avg_episodes, avg_episode_cost, label='Average Cost (100-episode avg)', color='blue', linestyle='--', zorder = 2)

    lines = line2 + line4
    labels = [line.get_label() for line in lines]
    plt.title('Constraint Cost')
    plt.legend(lines, labels, loc='best')

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path_cost)
    plt.show()
    plt.close()

def main():

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    num_elements = 34
    n_elements = num_elements # Initialize with your number of elements

    crane_capacity = 2 # Initialize with your crane capacity
    action_space, num_actions = getActionSpaceLength(n_elements, crane_capacity) # This will get the number of valid actions

    # Defining hyperparameters
    node_features_dim = 8
    gcn_hidden_dim = 64
    gcn_output_dim = 64
    material_availability_feature_size = 32
    resource_availability_feature_size = 32
    remaining_elements_feature_size = 32
    weather_feature_size = 64
    structural_support_feature_size = 64
    hidden_dim = 256 
    actor_hidden_dim = 128
    critic_hidden_dim = 128
    num_gcn_layers = 2
    num_heads_TGAT = 4
    time_embedding_dim = 8
    dropout_rate = 0
    lr = 0.0003
    gamma = 0.99
    entropy_coef = 0.08
    clip_epsilon = 0.25 
    gae_lambda = 0.95 
    K_epochs = 3
    minibatch_size = 256
    batch_size = 1024
    num_episodes = 5000
    max_timesteps = 1000 # Maximum timesteps per episode
    print_every = 100 # Print every n episodes
    save_every = 100 # Save the model every n episodes
    
    
    daily_crane_cost = 960 # Based on NZ data
    daily_labor_cost = 280 # based on NZ data
    daily_indirect_cost = 500 # Assumed
    labor_capacity = 15 # Assumed
    

    weather_impact_distribution =  {
                1: {'rain': [(0, 0.1), (10, 0.7), (50, 0.2)],'wind': [(9.8, 0), (7.5, 0.38), (5, 0.62)],'temp': [(20, 0.5), (30, 0.4), (40, 0.1)]},
                2: {'rain': [(0, 0.1), (10, 0.7), (50, 0.2)],'wind': [(9.8, 0), (7.5, 0.5), (5, 0.5)],'temp': [(20, 0.5), (30, 0.4), (40, 0.1)]},
                3: {'rain': [(0, 0.1), (10, 0.7), (50, 0.2)],'wind': [(9.8, 0.5), (7.5, 0.5), (5, 0)],'temp': [(20, 0.5), (30, 0.4), (40, 0.1)]},
                4: {'rain': [(0, 0.1), (10, 0.7), (50, 0.2)],'wind': [(9.8, 0), (7.5, 1), (5, 0)],'temp': [(20, 0.5), (30, 0.4), (40, 0.1)]},
                5: {'rain': [(0, 0.1), (10, 0.7), (50, 0.2)],'wind': [(9.8, 0), (7.5, 0.88), (5, 0.12)],'temp': [(20, 0.5), (30, 0.4), (40, 0.1)]},
                6: {'rain': [(0, 0.1), (10, 0.7), (50, 0.2)],'wind': [(9.8, 0.5), (7.5, 0.5), (5, 0)],'temp': [(20, 0.5), (30, 0.4), (40, 0.1)]},
                7: {'rain': [(0, 0.1), (10, 0.7), (50, 0.2)],'wind': [(9.8, 0), (7.5, 1), (5, 0)],'temp': [(20, 0.5), (30, 0.4), (40, 0.1)]},
                8: {'rain': [(0, 0.1), (10, 0.7), (50, 0.2)],'wind': [(9.8, 0), (7.5, 1), (5, 0)],'temp': [(20, 0.5), (30, 0.4), (40, 0.1)]},
                9: {'rain': [(0, 0.1), (10, 0.7), (50, 0.2)],'wind': [(9.8, 0), (7.5, 1), (5, 0)],'temp': [(20, 0.5), (30, 0.4), (40, 0.1)]},
                10: {'rain': [(0, 0.1), (10, 0.7), (50, 0.2)],'wind': [(9.8, 0), (7.5, 0.5), (5, 0.5)],'temp': [(20, 0.5), (30, 0.4), (40, 0.1)]},
                11: {'rain': [(0, 0.1), (10, 0.7), (50, 0.2)],'wind': [(9.8, 0), (7.5, 0.5), (5, 0.5)],'temp': [(20, 0.5), (30, 0.4), (40, 0.1)]},
                12: {'rain': [(0, 0.1), (10, 0.7), (50, 0.2)],'wind': [(9.8, 0), (7.5, 0.5), (5, 0.5)],'temp': [(20, 0.5), (30, 0.4), (40, 0.1)]},
                13: {'rain': [(0, 0.1), (10, 0.7), (50, 0.2)],'wind': [(9.8, 0), (7.5, 0.5), (5, 0.5)],'temp': [(20, 0.5), (30, 0.4), (40, 0.1)]},
                14: {'rain': [(0, 0.1), (10, 0.7), (50, 0.2)],'wind': [(9.8, 0), (7.5, 0.5), (5, 0.5)],'temp': [(20, 0.5), (30, 0.4), (40, 0.1)]},
                15: {'rain': [(0, 0.1), (10, 0.7), (50, 0.2)],'wind': [(9.8, 0), (7.5, 0.5), (5, 0.5)],'temp': [(20, 0.5), (30, 0.4), (40, 0.1)]}
                } # This is actually weather value distribution

    material_schedule ={
                1: [2, 17, 24, 9, 28, 5, 13, 29, 0, 21, 32, 7, 18, 12, 25, 31, 4],
                2: [16, 10, 8, 27, 20, 3, 26, 15, 30, 14, 33, 11, 23, 1, 19, 6, 22],
                3: [],
                4: [],
                5: [],
                6: [],
                7: [],
                8: [],
                9: [],
                10: [],
                11: [],
                12: [],
                13: [],
                14: [],
                15: [],
                16: [],
                17: [],
                18: [],
                19: [],
                20: []
            } # Which material elements are going to be delivered which day.
    
    weights =  np.array([250, 300, 300, 175, 75, 50, 50, 150, 75, 300, 275, 150, 225, 300, 75, 175, 250, 150, 150, 175, 175, 50, 50, 100, 100, 100, 175, 100, 150, 150, 175, 150, 50, 275], dtype = np.float32)

    # Support relations showing in a marix which element is supported on which element. Current considering everything to be zero just for the walls 
    support_relations = np.zeros((34, 34), dtype=np.float32)
    
    module_area =  np.array([3.0, 2.5, 2.5, 3.0, 1.25, 0.75, 0.75, 2.0, 0.75, 2.5, 2.75, 2.0, 2.5, 2.5, 0.75, 2.5, 3.0, 2.75, 2.75, 2.5, 3.0, 0.5, 0.5, 1.25, 0.5, 0.5, 2.0, 0.75, 1.25, 1.25, 1.25, 1.25, 0.5, 2.75]) #The area of modules

    adjacency_matrix = np.zeros((34, 34), dtype=np.float32)
     # Put the values as 1 wherever required. 
    adjacency = {1:[2,3],
                 2:[4,19],
                 3:[4],
                 4:[5,19],
                 5:[6],
                 6:[7],
                 7:[8],
                 8:[27],
                 9:[10],
                 10:[11],
                 11:[30,12],
                 12:[30,31],
                 13:[31,14],
                 14:[34],
                 15:[22,16,21],
                 16:[15,17,21],
                 17:[18,16,20],
                 18:[25,20,17],
                 19:[18,4,2],
                 20:[18,17,21],
                 21:[20,24,23,16,15],
                 22:[23,15],
                 23:[21],
                 24:[34,21],
                 25:[18],
                 26:[],
                 27:[28],
                 28:[29,27],
                 29:[28,30],
                 30:[11,12,29],
                 31:[12,32,13],
                 32:[31,33],
                 33:[32],
                 34:[14,24,15]}
    
    for key in adjacency.keys():
        for i in adjacency[key]:
            adjacency_matrix[i-1][key-1] = 1
            adjacency_matrix[key-1][i-1] = 1
    
    # Interference matrix 
    interference_matrix =  np.array([[[0,0,0,0],[0,1,1,1],[1,0,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[1,0,1,1],[0,0,0,0],[0,0,0,0],[1,0,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,1,1,1],[0,0,0,0],[0,0,0,0],[0,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,1,1,1],[1,0,1,1],[0,0,0,0],[1,1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,0],[0,0,0,0],[1,0,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,1,1,1],[0,0,0,0],[1,0,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,1,1,1],[0,0,0,0],[1,1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,0],[0,0,0,0],[0,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,1,1],[0,0,0,0],[0,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,1,1],[0,0,0,0],[0,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,0],[1,1,1,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,0,1],[0,0,0,0],[1,1,1,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,0,1],[0,0,0,0],[1,0,1,1],[0,0,0,0],[1,1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,1,1,1],[0,0,0,0],[1,0,1,1],[1,1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,1,1],[0,0,0,0],[0,0,0,0],[1,1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,0],[0,0,0,0],[0,0,0,0],[1,1,0,1],[1,1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,0],[0,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,0,1]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,0],[0,0,0,0],[1,1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,0,1],[1,1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,0,1],[1,1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,1,1,1],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,1,1],[0,0,0,0],[1,1,0,1],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,0],[0,0,0,0],[0,0,0,0]],
                                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]])
    # Resource requirements
    resource_requirements =  {
                0: [1, 4],  # Small Wall requires 1 crane, 4 labor
                1: [1, 6],  # Big Walls requires 1 cranes, 6 labor
                2: [1, 2],  # Slab requires 1 crane, 2 labor
                3: [1, 2]   # Stair requires 1 crane, 1 labor
            }
    
    # Triangular distribution of durations
    duration_distribution =  { # Remember these durations are in hours
            0: (0.47, 0.625, 0.78),  # For element 0, min=3 days, most likely=5 days, max=7 days
            1: (0.47, 0.625, 0.78),  # For element 1
            2: (0.47, 0.625, 0.78),  # For element 2
            3: (0.47, 0.625, 0.78),  # For element 3
            4: (0.47, 0.625, 0.78),  # For element 4
            5: (0.47, 0.625, 0.78),  # For element 5
            6: (0.47, 0.625, 0.78),  # For element 6
            7: (0.47, 0.625, 0.78),  # For element 7
            8: (0.47, 0.625, 0.78),  # For element 8
            9: (0.47, 0.625, 0.78),  # For element 9
            10: (0.47, 0.625, 0.78),  # For element 10
            11: (0.47, 0.625, 0.78),  # For element 11
            12: (0.47, 0.625, 0.78),  # For element 12
            13: (0.47, 0.625, 0.78),  # For element 13
            14: (0.47, 0.625, 0.78),  # For element 14
            15: (0.47, 0.625, 0.78),  # For element 15
            16: (0.47, 0.625, 0.78),  # For element 16
            17: (0.47, 0.625, 0.78),  # For element 17
            18: (0.47, 0.625, 0.78),  # For element 18
            19: (0.47, 0.625, 0.78),  # For element 19
            20: (0.47, 0.625, 0.78),  # For element 20
            21: (0.47, 0.625, 0.78),  # For element 21
            22: (0.47, 0.625, 0.78),  # For element 22
            23: (0.47, 0.625, 0.78),  # For element 23
            24: (0.47, 0.625, 0.78),  # For element 24
            25: (0.47, 0.625, 0.78),  # For element 25
            26: (0.47, 0.625, 0.78),  # For element 26
            27: (0.47, 0.625, 0.78),  # For element 27
            28: (0.47, 0.625, 0.78),  # For element 28
            29: (0.47, 0.625, 0.78),  # For element 29
            30: (0.47, 0.625, 0.78),  # For element 30
            31: (0.47, 0.625, 0.78),  # For element 31
            32: (0.47, 0.625, 0.78),  # For element 32
            33: (0.47, 0.625, 0.78),  # For element 33
            }
    
    
    # Initialize material availability
    initial_material_availability = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0], dtype = np.float32) # The binary array showing the initial material availability

    # Initialize the element types
    element_types = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    element_type_big = [1,2,3,4,10,11,13,14,16,17,18,19,20,21,34]

    for i in element_type_big:
        element_types[i-1] = 1
    # Initialize the structural support relationships

    structure_support_relationships = np.zeros((34, 34), dtype=np.float32)
    
    reward_count = 0
    running_reward_mean = 0.0
    running_reward_var = 0.0
    eps = 1e-8
    import math

    def update_reward_stats(new_reward, reward_count, running_reward_mean, running_reward_var):
        """ Update running mean and variance with a new reward using Welford's algorithm. """
        reward_count += 1
        old_mean = running_reward_mean
        running_reward_mean += (new_reward - running_reward_mean) / reward_count
        running_reward_var += (new_reward - old_mean) * (new_reward - running_reward_mean)
        return reward_count, running_reward_mean, running_reward_var
    
    def normalize_reward(reward, reward_count, running_reward_mean, running_reward_var):
        """ Normalize the reward using the running mean and variance. """
        if reward_count > 1:
            reward_std = math.sqrt(running_reward_var / (reward_count - 1))
        else:
            reward_std = 1.0  # If we have less than 2 samples, just use 1.0 as std
        normalized = (reward - running_reward_mean) / (reward_std + eps)
        return normalized

    
    # Initialize environment and agent
    # Initializing actor_critic model
    actor_critic = ActorCritic(n_elements = num_elements, 
                    node_feature_dim = node_features_dim, 
                    gcn_hidden_dim = gcn_hidden_dim,
                    gcn_output_dim = gcn_output_dim,
                    material_availability_feature_size = material_availability_feature_size,
                    resource_availability_feature_size = resource_availability_feature_size,
                    remaining_elements_feature_size = remaining_elements_feature_size,
                    structural_support_feature_size = structural_support_feature_size, # Need to try with another graph neural nework
                    weather_feature_size = weather_feature_size, 
                    hidden_dim = hidden_dim, 
                    actor_hidden_dim = actor_hidden_dim,
                    critic_hidden_dim = critic_hidden_dim,
                    num_heads = num_heads_TGAT,
                    time_embedding_dim = time_embedding_dim,
                    num_actions = num_actions,
                    dropout_rate = dropout_rate,
                    device = None)
    
    agent = Agent(actor_critic_model = actor_critic, 
                action_space = action_space, # The whole action space should be passed. Why would you pass only valid actions here.
                lr = lr, 
                gamma = gamma,
                entropy_coef = entropy_coef,
                clip_epsilon = clip_epsilon, 
                gae_lambda = gae_lambda, # Need to see how to decide this value.
                K_epochs = K_epochs,
                minibatch_size = minibatch_size,
                batch_size = minibatch_size,
                device = None)
    
    env = PrecastSchedulingEnv(n_elements = num_elements, 
                daily_crane_cost = daily_crane_cost, 
                daily_labor_cost = daily_labor_cost, 
                daily_indirect_cost = daily_indirect_cost, 
                crane_capacity = crane_capacity, 
                labor_capacity = labor_capacity, 
                weather_forecast = weather_impact_distribution, 
                material_schedule = material_schedule, 
                weights = weights, 
                support_relations = support_relations, 
                module_area = module_area, 
                adjacency_matrix = adjacency_matrix, 
                interference_matrix = interference_matrix, 
                resource_requirements = resource_requirements, 
                duration_distribution = duration_distribution, 
                weather_impact_distribution = weather_impact_distribution, 
                initial_material_availability = initial_material_availability, 
                element_types = element_types, 
                structure_support_relationships = structure_support_relationships,
                action_space = action_space,
                weight = weights,
                area = module_area)

    # Tracking the variables
    episode_rewards = []
    scheduled_durations = []
    avg_rewards = []
    avg_durations = []
    episode_constraint_violations = []
    avg_constraint_violations = []
    episode_cost = []
    avg_episode_cost = []

    for episode in tqdm(range(1, num_episodes+1), desc = "Training"):
        state = env.reset() # Reset the environment
        done = False
        episode_reward = 0
        timestep = 0
        # constraint_violations = 0
        total_constraint_violations = 0
        cost_episode = 0

        while not done and timestep<max_timesteps:
            # Select the action
            action = agent.select_action(state, timestep)
            # print(state['action_mask'])
            # Take action in the environment 
            next_state, reward, done, total_duration, constraint_violations, cost_step = env.step(action, timestep) # This will essentially increase the step.
            total_constraint_violations = total_constraint_violations + constraint_violations
            cost_episode = cost_episode + cost_step

            reward_count, running_reward_mean, running_reward_var = update_reward_stats(reward, reward_count, running_reward_mean, running_reward_var)

            normalized_reward = normalize_reward(reward, reward_count, running_reward_mean, running_reward_var)

            agent.store_transition(normalized_reward, done)

            # Update episode reward
            episode_reward = episode_reward + reward

            # Update state
            state = next_state

            timestep = timestep + 1

            # Check if batch_size is reached
            if len(agent.rewards) >= batch_size:
                agent.update()
                # Reset memory is handled inside agent.update()

        episode_rewards.append(episode_reward)
        scheduled_durations.append(total_duration)
        episode_constraint_violations.append(total_constraint_violations)
        episode_cost.append(cost_episode)

        # Calculated average reward
        if episode % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            avg_rewards.append(avg_reward)
            # print(f"Episode {episode} \t Average Reward: {avg_reward:.2f}")

        # Calculated average reward
        if episode % print_every == 0:
            avg_duration = np.mean(scheduled_durations[-print_every:])
            avg_durations.append(avg_duration)
            # print(f"Episode {episode} \t Average Reward: {avg_reward:.2f}")

        # Calculated average reward
        if episode % print_every == 0:
            avg_constraint_violation = np.mean(episode_constraint_violations[-print_every:])
            avg_constraint_violations.append(avg_constraint_violation)
            # print(f"Episode {episode} \t Average Reward: {avg_reward:.2f}")

        # Calculated average cost
        if episode % print_every == 0:
            avg_cost = np.mean(episode_cost[-print_every:])
            avg_episode_cost.append(avg_cost)
            # print(f"Episode {episode} \t Average Reward: {avg_reward:.2f}")

        # Save the model periodically
        if episode % save_every == 0:
            model_path = os.path.join(model_save_dir, f"PPO_Precast_Scheduling_Model_Episode_{episode}.pth")
            
            # Create a dictionary with model state and metrics
            checkpoint = {
                'model_state_dict': agent.model.state_dict(),  # Save the model parameters
                'avg_rewards': avg_rewards,  # Save the list of average rewards
                'avg_durations': avg_durations,  # Save the list of average durations
                'episode': episode,  # Optionally, save the current episode number
                'average_violations': avg_constraint_violations, # # Saves the list of average number of constraint violations for the episode
                'average_cost': avg_episode_cost
            }

            torch.save(checkpoint, model_path)
            print(f"Model saved at Episode {episode} to {model_path}")
            print(f"Model saved at Episode {episode} to {model_path}")
        
    # After training, save the final model
    torch.save(agent.model.state_dict(), f"PPO_Precast_Scheduling_Model_Final.pth")
    print("Training completed and final model saved.")
    save_path_durations = f"D:/Ajay/Precast_Assembly_Scheduling/Training_models/Improved2_Singleweather_RewardNorm_CostRewardDeltat/Validation/PPO_TGAT_TGN/Plot_Durations.png"
    save_path_rewards = f"D:/Ajay/Precast_Assembly_Scheduling/Training_models/Improved2_Singleweather_RewardNorm_CostRewardDeltat/Validation/PPO_TGAT_TGN/Plot_Rewards.png"
    save_path_violations = f"D:/Ajay/Precast_Assembly_Scheduling/Training_models/Improved2_Singleweather_RewardNorm_CostRewardDeltat/Validation/PPO_TGAT_TGN/Plot_Violations.png"
    save_path_cost = f"D:/Ajay/Precast_Assembly_Scheduling/Training_models/Improved2_Singleweather_RewardNorm_CostRewardDeltat/Validation/PPO_TGAT_TGN/Plot_Cost.png"
    plot_rewards_durations(episode_rewards, num_episodes, save_path_durations, scheduled_durations, avg_rewards, avg_durations)
    plot_rewards_rewards(episode_rewards, num_episodes, save_path_rewards, scheduled_durations, avg_rewards, avg_durations)
    plot_rewards_violations(episode_rewards, num_episodes, save_path_violations, episode_constraint_violations, avg_rewards, avg_constraint_violations)
    plot_rewards_cost(episode_rewards, num_episodes, save_path_cost, episode_cost, avg_rewards, avg_episode_cost)
    end_time = time.time()
    # Calculate and print the execution time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    # Plot average rewards
    """plt.figure(figsize=(12,8))
    plt.plot(range(len(avg_rewards)), avg_rewards)
    plt.xlabel(f'Episode (Every {print_every} Episodes)')
    plt.ylabel('Average Reward')
    plt.title('PPO Training Performance on Precast Scheduling Environment')
    plt.show()"""

if __name__ == "__main__":
    main()