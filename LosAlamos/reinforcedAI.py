from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from threading import Thread
from Datacollector import DataCollector
from MLmodel import train_model
from collections import deque
import random

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CustomEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, model) -> None:
        super(CustomEnv, self).__init__()
        self.data_collector = DataCollector()  # Initialize DataCollector within __init__

        # Fetch initial data and setup observation space
        self.main_df = self.data_collector.get_prev_data()
        num_features = len(self.main_df.columns)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float64)

        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = Discrete(3)

        # Initialize state variables
        self.position_status = False
        self.position = None
        self.enter_price = None
        self.current_step = 0

        # Load the trained model
        self.model = model
        self.features = None

        # Initialize experience replay buffer and other DQN parameters
        self.replay_buffer = deque(maxlen=2000)
        self.batch_size = 64
        self.gamma = 0.99
        self.q_network = QNetwork(input_dim=num_features, output_dim=self.action_space.n)
        self.target_q_network = QNetwork(input_dim=num_features, output_dim=self.action_space.n)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def step(self, action):
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        if self.current_step >= len(self.main_df) - 1:
            return self.main_df.iloc[self.current_step].values, 0, True, {}

        current_price = self.main_df.iloc[self.current_step]['Close']
        done = False
        reward = 0

        # Execute the action
        if action == 1:  # Buy
            if not self.position_status:  # If no open position, buy
                self.position_status = True
                self.position = 'long'
                self.enter_price = current_price

        elif action == 2:  # Sell
            if self.position_status and self.position == 'long':  # If in a long position, sell
                self.position_status = False
                profit = current_price - self.enter_price  # Calculate profit
                reward = profit  # Reward is the profit
                print(f"Profit/Loss: {profit}")
                self.enter_price = None

        # Advance to the next step
        self.current_step += 1
        if self.current_step >= len(self.main_df):
            done = True

        # Check end of episode condition
        if self.current_step >= 500:  # Arbitrary end condition
            done = True

        # Return new state, reward, done, and additional info
        next_state = self.main_df.iloc[self.current_step].values if not done else self.main_df.iloc[-1].values
        return next_state, reward, done, {}

    def reset(self):
        # Re-fetch initial market data and reset state variables
        self.main_df = self.data_collector.get_prev_data()
        self.current_step = 0
        self.position_status = False
        self.position = None
        self.enter_price = None

        # Prepare the features
        dataset = self.data_collector.prepare_dataset()
        self.features = dataset.drop('target', axis=1)

        # Return the initial state
        initial_state = self.main_df.iloc[self.current_step].values
        return initial_state

    def render(self, mode='human'):
        pass  # Only print profit and loss

    def close(self):
        print("Environment closed and resources cleaned up")

    def decide_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return self.action_space.sample()  # Explore
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return np.argmax(q_values.numpy())

    def replay_experience(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            reward = torch.FloatTensor([reward])
            action = torch.LongTensor([action])

            # Compute Q targets
            if done:
                q_target = reward
            else:
                q_target = reward + self.gamma * torch.max(self.target_q_network(next_state))

            q_value = self.q_network(state)[action]
            loss = self.criterion(q_value, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

if __name__ == "__main__":
    data_collector = DataCollector()
    websocket_thread_vol = Thread(target=data_collector.websocket_thread_vol)
    websocket_thread_vol.start()
    
    websocket_thread_kline = Thread(target=data_collector.websocket_thread_kline)
    websocket_thread_kline.start()

    # Train and load the model
    model = train_model(data_collector)
    # If you already have a trained model, you can load it directly
    # model = joblib.load('trading_model.pkl')

    env = CustomEnv(model)
    state = env.reset()

    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.995
    epsilon_min = 0.01
    episodes = 100

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = env.decide_action(state, epsilon)  # Use the epsilon-greedy policy
            next_state, reward, done, _ = env.step(action)

            # Store experience in replay buffer
            env.replay_buffer.append((state, action, reward, next_state, done))

            # Learn from experience
            env.replay_experience()

            state = next_state
            total_reward += reward

        # Update the target network
        env.update_target_network()

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

    env.close()
