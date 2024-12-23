import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tqdm import tqdm  # For progress bar

# Check TensorFlow version and GPU availability
print("TensorFlow Version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# Load training data
training_data = pd.read_csv('training_data.csv')  # Update the path as necessary

# Preprocess data
def preprocess_data(data):
    # Normalize numerical columns
    numerical_cols = ['Load', 'PV', 'Tou Tariff', 'Hour', 'Day']
    data[numerical_cols] = data[numerical_cols] / data[numerical_cols].max()
    return data

training_data = preprocess_data(training_data)

# Precompute maximum values for denormalization
load_max = training_data['Load'].max()
pv_max = training_data['PV'].max()
tou_tariff_max = training_data['Tou Tariff'].max()
day_max = training_data['Day'].max()
hour_max = 23  # Assuming Hour ranges from 0 to 23

# Define the environment for DQN
class EnergyEnv:
    def __init__(self, data):
        self.data = data.reset_index(drop=True)
        self.max_steps = len(data)
        self.current_step = 0

        # Battery specifications
        self.E_BB = 5.0  # Battery capacity in kWh
        self.SoC_min = 0.2 * self.E_BB  # 20% SoC (1 kWh)
        self.SoC_max = 0.8 * self.E_BB  # 80% SoC (4 kWh)
        self.SoC = self.SoC_min  # Initialize SoC at minimum

        self.eta_ch = 0.95  # Charging efficiency
        self.eta_dis = 0.95  # Discharging efficiency

        self.done = False

        # Initialize variables for charging from grid
        self.charged_from_grid_today = False
        self.previous_day = None

        # Precompute maximum values for denormalization
        self.load_max = load_max
        self.pv_max = pv_max
        self.tou_tariff_max = tou_tariff_max
        self.day_max = day_max
        self.hour_max = hour_max

    def reset(self):
        self.current_step = 0
        self.SoC = self.SoC_min  # Reset SoC to minimum at the start
        self.done = False
        self.charged_from_grid_today = False
        self.previous_day = None
        return self._get_state()

    def _get_state(self):
        row = self.data.iloc[self.current_step]
        state = np.array([
            self.SoC / self.E_BB,  # Normalized SoC
            row['Load'],
            row['PV'],
            row['Tou Tariff'],
            row['Hour'],  # Already normalized in preprocessing
            row['Day']    # Already normalized in preprocessing
        ])
        return state

    def step(self, action):
        if self.done:
            return np.zeros(self.state_size()), 0, self.done, {}

        row = self.data.iloc[self.current_step]
        P_load = row['Load'] * self.load_max  # Denormalize
        P_PV = row['PV'] * self.pv_max        # Denormalize
        lambda_grid = row['Tou Tariff'] * self.tou_tariff_max  # Denormalize

        day = int(row['Day'] * self.day_max)  # Denormalize
        hour = int(row['Hour'] * self.hour_max)  # Denormalize

        # Check if day has changed
        if self.previous_day != day:
            self.charged_from_grid_today = False
            self.previous_day = day

        # Define weight factor w_t (e.g., higher during peak hours)
        if 17 <= hour <= 20:
            w_t = 1.5  # Peak hours
        else:
            w_t = 1.0  # Off-peak hours

        # Action mapping
        # 0: Do nothing
        # 1: Charge battery
        # 2: Discharge battery
        P_BB_ch = 0.0
        P_BB_dis = 0.0

        if action == 1:
            # Charge battery
            # Remove charging limit restriction for PV
            P_BB_ch_max = (self.SoC_max - self.SoC) * self.E_BB / self.eta_ch
            PV_excess = max(0.0, P_PV - P_load)
            P_BB_ch_from_PV = min(PV_excess, P_BB_ch_max)
            P_BB_ch_from_grid = 0.0

            # Check if battery is not fully charged and hasn't charged from grid today
            if P_BB_ch_from_PV < P_BB_ch_max:
                P_BB_ch_remaining = P_BB_ch_max - P_BB_ch_from_PV
                if not self.charged_from_grid_today:
                    P_BB_ch_from_grid = P_BB_ch_remaining
                    self.charged_from_grid_today = True
                else:
                    P_BB_ch_from_grid = 0.0

            P_BB_ch = P_BB_ch_from_PV + P_BB_ch_from_grid
            P_BB_dis = 0.0
        elif action == 2:
            # Discharge battery
            P_BB_dis = min((self.SoC - self.SoC_min) * self.E_BB * self.eta_dis, P_load - P_PV)
            P_BB_ch = 0.0
        else:
            # Do nothing
            P_BB_ch = 0.0
            P_BB_dis = 0.0

        # Update SoC
        delta_SoC = (self.eta_ch * P_BB_ch - P_BB_dis / self.eta_dis) / self.E_BB
        self.SoC += delta_SoC
        self.SoC = max(self.SoC_min, min(self.SoC, self.SoC_max))  # Enforce SoC limits

        # Power balance
        P_grid = P_load - P_PV - P_BB_dis + P_BB_ch

        # Export excess PV to grid after battery is fully charged
        if P_grid < 0:
            P_export = -P_grid
            P_grid = 0.0
        else:
            P_export = 0.0

        # Calculate cost (positive for import, negative for export)
        cost = P_grid * lambda_grid - P_export * lambda_grid  # Subtract export earnings

        # Immediate reward
        reward = -w_t * cost

        # Move to next time step
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True

        next_state = self._get_state() if not self.done else np.zeros(self.state_size())

        # Info dictionary for debugging
        info = {
            'SoC': self.SoC,
            'P_BB_ch': P_BB_ch,
            'P_BB_dis': P_BB_dis,
            'P_grid': P_grid,
            'P_export': P_export,
            'Cost': cost,
            'w_t': w_t,
            'Charged from grid today': self.charged_from_grid_today
        }

        return next_state, reward, self.done, info

    def state_size(self):
        return 6  # [SoC, Load, PV, Tou Tariff, Hour, Day]

    def action_size(self):
        return 3  # Actions: 0 - Do nothing, 1 - Charge, 2 - Discharge

# Define DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size, initial_epsilon=1.0, epsilon_min=0.05, episodes=100):
        self.state_size = state_size  # Number of features in state
        self.action_size = action_size  # Number of possible actions
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95    # Discount rate
        self.epsilon = initial_epsilon   # Exploration rate
        self.epsilon_min = epsilon_min
        self.episodes = episodes
        self.epsilon_decay = self.calculate_epsilon_decay()
        self.learning_rate = 0.001
        # Handle device placement
        self.device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        with tf.device(self.device):
            self.model = self._build_model()

    def calculate_epsilon_decay(self):
        # Calculate epsilon decay rate
        return (self.epsilon_min / self.epsilon) ** (1.0 / self.episodes)

    def _build_model(self):
        # Simplified Neural Net for Deep-Q learning Model
        model = models.Sequential()
        model.add(layers.Dense(32, activation='relu', input_shape=(self.state_size,)))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss=self.weighted_mse,
                      optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def weighted_mse(self, y_true, y_pred):
        # Custom loss function with weight factor w_t
        w = y_true[:, -1]
        y_true = y_true[:, :-1]
        # Expand w to shape [batch_size, 1] for broadcasting
        w = tf.expand_dims(w, axis=-1)
        return tf.reduce_mean(w * tf.square(y_true - y_pred))

    def remember(self, state, action, reward, next_state, done, w_t):
        self.memory.append((state, action, reward, next_state, done, w_t))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with tf.device(self.device):
            state = np.expand_dims(state, axis=0)
            act_values = self.model(state, training=False).numpy()
        return np.argmax(act_values[0])  # Returns action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([item[0] for item in minibatch])
        actions = np.array([item[1] for item in minibatch])
        rewards = np.array([item[2] for item in minibatch])
        next_states = np.array([item[3] for item in minibatch])
        dones = np.array([item[4] for item in minibatch])
        weights = np.array([item[5] for item in minibatch])

        with tf.device(self.device):
            target = self.model.predict(states, verbose=0)
            target_next = self.model.predict(next_states, verbose=0)

        Q_future = np.max(target_next, axis=1)
        target[range(batch_size), actions] = rewards + (1 - dones) * self.gamma * Q_future

        # Append weights to targets for custom loss function
        target = np.hstack((target, weights.reshape(-1, 1)))

        with tf.device(self.device):
            self.model.fit(states, target, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# Initialize environment and agent
state_size = 6  # [SoC, Load, PV, Tou Tariff, Hour, Day]
action_size = 3  # Actions: 0 - Do nothing, 1 - Charge, 2 - Discharge
env = EnergyEnv(training_data)
initial_epsilon = 1.0
final_epsilon = 0.95
episodes = 20  # Number of episodes
agent = DQNAgent(state_size, action_size, initial_epsilon=initial_epsilon, epsilon_min=final_epsilon, episodes=episodes)
batch_size = 64  # Increased batch size for better GPU utilization
rewards = []

# Initialize tqdm progress bar for episodes
with tqdm(total=episodes, desc="Training", unit="episode") as pbar_episodes:
    for e in range(episodes):
        state = env.reset()
        total_reward = 0

        # Initialize tqdm progress bar for steps within the episode
        with tqdm(total=env.max_steps, leave=False, desc=f"Episode {e+1}", unit="step") as pbar_steps:
            while not env.done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                w_t = info['w_t']
                agent.remember(state, action, reward, next_state, done, w_t)
                state = next_state
                total_reward += reward

                if len(agent.memory) >= batch_size:
                    agent.replay(batch_size)

                # Update steps progress bar
                pbar_steps.update(1)

        # Epsilon decay after each episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            agent.epsilon = max(agent.epsilon_min, agent.epsilon)

        rewards.append(total_reward)

        # Update episodes progress bar
        pbar_episodes.set_postfix({'Epsilon': f"{agent.epsilon:.2f}", 'Total Reward': f"{total_reward:.2f}"})
        pbar_episodes.update(1)

# Plot training rewards
plt.figure(figsize=(10, 6))
plt.plot(rewards, marker='o')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Rewards over Episodes')
plt.grid(True)
plt.savefig('training_rewards.png')
plt.close()

# Save the trained model
agent.model.save('dqn_energy_model.h5')
print("Training completed and model saved.")
