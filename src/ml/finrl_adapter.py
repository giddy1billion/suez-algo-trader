"""
FinRL Integration — Deep Reinforcement Learning for trading.
Wraps FinRL's DRLAgent for training and inference with Alpaca data.

NOTE: FinRL and its dependencies (stable-baselines3, gymnasium) are heavy.
      Install separately: pip install finrl stable-baselines3 gymnasium
"""

import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Technical indicators used by FinRL
INDICATORS = [
    'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
    'close_30_sma', 'close_60_sma', 'vix', 'turbulence',
]


def prepare_finrl_data(data: dict[str, pd.DataFrame], indicators: list[str] = None) -> pd.DataFrame:
    """
    Prepare data in FinRL's expected format.
    FinRL expects: date, tic, open, high, low, close, volume + indicator columns.

    Args:
        data: Dict of symbol -> OHLCV DataFrame
        indicators: Technical indicators to add

    Returns:
        Combined DataFrame in FinRL format
    """
    indicators = indicators or INDICATORS
    all_frames = []

    for symbol, df in data.items():
        df = df.copy()
        df['tic'] = symbol

        # Ensure we have a date column
        if isinstance(df.index, pd.DatetimeIndex):
            df['date'] = df.index
            df = df.reset_index(drop=True)

        # Add basic technical indicators
        df = _add_indicators(df, indicators)
        all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(['date', 'tic']).reset_index(drop=True)
    return combined


def _add_indicators(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    """Add technical indicators to dataframe."""
    # MACD
    if 'macd' in indicators:
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26

    # Bollinger Bands
    if 'boll_ub' in indicators or 'boll_lb' in indicators:
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['boll_ub'] = sma20 + 2 * std20
        df['boll_lb'] = sma20 - 2 * std20

    # RSI
    if 'rsi_30' in indicators:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(30).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(30).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_30'] = 100 - (100 / (1 + rs))

    # CCI
    if 'cci_30' in indicators:
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(30).mean()
        mad = tp.rolling(30).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci_30'] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

    # DX (Directional Index)
    if 'dx_30' in indicators:
        df['dx_30'] = 0  # Simplified placeholder

    # SMAs
    if 'close_30_sma' in indicators:
        df['close_30_sma'] = df['close'].rolling(30).mean()
    if 'close_60_sma' in indicators:
        df['close_60_sma'] = df['close'].rolling(60).mean()

    # Placeholders for market-wide indicators
    if 'vix' in indicators:
        df['vix'] = df['close'].rolling(20).std() * np.sqrt(252) * 100 / df['close']
    if 'turbulence' in indicators:
        df['turbulence'] = 0  # Would need multi-asset covariance

    return df.fillna(0)


class FinRLTrader:
    """
    FinRL-based deep reinforcement learning trader.

    Supports multiple RL algorithms:
    - A2C (Advantage Actor-Critic)
    - PPO (Proximal Policy Optimization)
    - DDPG (Deep Deterministic Policy Gradient)
    - SAC (Soft Actor-Critic)
    - TD3 (Twin Delayed DDPG)
    """

    def __init__(
        self,
        symbols: list[str],
        model_name: str = "ppo",
        total_timesteps: int = 100_000,
        model_dir: str = "models/finrl",
    ):
        self.symbols = symbols
        self.model_name = model_name.lower()
        self.total_timesteps = total_timesteps
        self.model_dir = model_dir
        self.model = None
        self.env = None

        os.makedirs(model_dir, exist_ok=True)

    def train(self, training_data: pd.DataFrame):
        """
        Train a DRL agent on historical data.

        Args:
            training_data: DataFrame in FinRL format (from prepare_finrl_data)
        """
        try:
            from finrl.agents.stablebaselines3.models import DRLAgent
            from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
        except ImportError:
            logger.error("finrl.not_installed",
                        msg="Install: pip install finrl stable-baselines3 gymnasium")
            return

        # Environment setup
        stock_dimension = len(self.symbols)
        state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
        action_space = stock_dimension

        env_kwargs = {
            "hmax": 100,
            "initial_amount": 100000,
            "num_stock_shares": [0] * stock_dimension,
            "buy_cost_pct": [0.001] * stock_dimension,
            "sell_cost_pct": [0.001] * stock_dimension,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": INDICATORS,
            "action_space": action_space,
            "reward_scaling": 1e-4,
        }

        env = StockTradingEnv(df=training_data, **env_kwargs)
        agent = DRLAgent(env=env)

        # Select algorithm
        model_kwargs = self._get_model_kwargs()
        model = agent.get_model(self.model_name, model_kwargs=model_kwargs)

        # Train
        logger.info("finrl.training", algorithm=self.model_name, timesteps=self.total_timesteps)
        trained_model = agent.train_model(
            model=model,
            tb_log_name=self.model_name,
            total_timesteps=self.total_timesteps,
        )

        self.model = trained_model
        self._save_model()
        logger.info("finrl.trained", model_path=self._model_path())

    def predict(self, current_state: pd.DataFrame) -> dict[str, float]:
        """
        Get trading actions from the trained model.

        Args:
            current_state: Current market state in FinRL format

        Returns:
            Dict of symbol -> action weight (-1 to 1, negative = sell)
        """
        if self.model is None:
            self._load_model()
            if self.model is None:
                return {}

        try:
            action, _ = self.model.predict(current_state)
            # Map actions to symbols
            actions = {}
            for i, symbol in enumerate(self.symbols):
                if i < len(action):
                    actions[symbol] = float(action[i])
            return actions
        except Exception as e:
            logger.error("finrl.predict_error", error=str(e))
            return {}

    def _get_model_kwargs(self) -> dict:
        """Get algorithm-specific hyperparameters."""
        configs = {
            "a2c": {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007},
            "ppo": {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 64},
            "ddpg": {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001},
            "sac": {"batch_size": 64, "buffer_size": 100000, "learning_rate": 0.0001, "ent_coef": "auto"},
            "td3": {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001},
        }
        return configs.get(self.model_name, {})

    def _model_path(self) -> str:
        return os.path.join(self.model_dir, f"{self.model_name}_trader")

    def _save_model(self):
        if self.model:
            self.model.save(self._model_path())

    def _load_model(self):
        path = self._model_path()
        if os.path.exists(path + ".zip"):
            try:
                from stable_baselines3 import A2C, PPO, DDPG, SAC, TD3
                model_classes = {"a2c": A2C, "ppo": PPO, "ddpg": DDPG, "sac": SAC, "td3": TD3}
                cls = model_classes.get(self.model_name)
                if cls:
                    self.model = cls.load(path)
                    logger.info("finrl.model_loaded", path=path)
            except ImportError:
                logger.error("finrl.sb3_not_installed")
