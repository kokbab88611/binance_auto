from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
from chatgpttestprint import DataCollector
import ta  # Technical Analysis library for financial indicators

class CustomEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self) -> None:
        super(CustomEnv, self).__init__()
        self.data_collector = DataCollector()  # Initialize DataCollector within __init__
        
        # Set up action and observation spaces
        self.action_space = Discrete(3)  # 0: hold, 1: go long, 2: go short
        
        # Observation space depends on the number of features (columns) in the DataFrame
        num_features = len(self.data_collector.main_df.columns)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float64)

    def step(self, action):
        # Execute one time step within the environment
        current_price = self.data_collector.main_df.iloc[-1]['Close']
        done = False
        info = {}

        if action == 1:  # Buy
            self.data_collector.open_position(current_price)
        elif action == 2:  # Sell
            self.data_collector.close_position(current_price)

        # Calculate reward (this is a placeholder, you'll need your own logic)
        reward = self.calculate_reward()

        # Get the new state
        new_obs = self.data_collector.main_df.iloc[-1].values

        return new_obs, reward, done, info

    def reset(self):
        state = None
        return state

    def render(self, mode='human'):
        # Render the environment state (optional)
        print(f"Render in mode: {mode}")

    def close(self):
        # Perform any cleanup on environment close
        pass

env = CustomEnv()
state = env.reset()

class TradingEnv(Env):
    """A simple trading environment for reinforcement learning."""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TradingEnv, self).__init__()

        # Load historical data
        self.data = self.get_prev_data()
        self.current_step = 0
        self.done = False
        self.profits = 0

        # Define action and observation space
        self.action_space = Discrete(3)  # [Hold, Buy, Sell]
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)  # Adjust based on features

    def reset(self):
        self.current_step = 0
        self.done = False
        self.profits = 0
        return self._next_observation()

    def _next_observation(self):
        # Return state as a numpy array of features
        frame = self.data.iloc[self.current_step]
        return np.array([frame['Open'], frame['High'], frame['Low'], frame['Close'], frame['Volume'], self.profits])

    def step(self, action):
        self.current_step += 1

        if self.current_step > len(self.data) - 1:
            self.done = True

        current_data = self.data.iloc[self.current_step]
        reward = 0

        # Implement your trading logic here
        if action == 1:  # Buy
            self.enter_price = current_data['Close']  # Buy at Close price
        elif action == 2:  # Sell
            if self.enter_price is not None:
                reward = current_data['Close'] - self.enter_price  # Profit is realized here
                self.profits += reward
                self.enter_price = None

        next_state = self._next_observation()
        return next_state, reward, self.done, {}

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Price: {self.data.iloc[self.current_step]["Close"]}')

    def get_prev_data(self):
        # Assume this function returns a DataFrame with your price data
        url = 'https://fapi.binance.com/fapi/v1/klines'
        params = {'symbol': 'BTCUSDT', 'interval': '3m', 'limit': 100}
        response = requests.get(url, params=params).json()
        df = pd.DataFrame(response, columns=['openTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'closeTime', 'assetVolume', 'tradeNum', 'TBBAV', 'TBQAV', 'ignore'])
        df = df.drop(df.columns[[6, 7, 8, 9, 10, 11]], axis=1)
        df = df.astype({'Open': 'float', 'High': 'float', 'Low': 'float', 'Close': 'float', 'Volume': 'float'})
        return df
