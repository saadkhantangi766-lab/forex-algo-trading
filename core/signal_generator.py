"""Signal generation system using technical indicators and ML models."""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime


class TechnicalIndicators:
    """Technical analysis indicators."""
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_moving_average(data: pd.Series, period: int) -> pd.Series:
        """Calculate simple moving average."""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr


class SignalGenerator:
    """Generate trading signals based on technical analysis and ML models."""
    
    def __init__(self, ml_predictor=None, config: Dict = None):
        """
        Initialize signal generator.
        
        Args:
            ml_predictor: Machine learning predictor object
            config: Configuration dictionary
        """
        self.ml_predictor = ml_predictor
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.indicators = TechnicalIndicators()
    
    def generate_signal(self, ohlc_data: pd.DataFrame, instrument: str) -> Optional[Dict]:
        """
        Generate trading signal from OHLC data.
        
        Args:
            ohlc_data: DataFrame with columns [open, high, low, close, volume]
            instrument: Trading instrument (e.g., 'EURUSD')
        
        Returns:
            Signal dictionary or None if no signal
        """
        try:
            # Calculate technical indicators
            close_prices = ohlc_data['close']
            high_prices = ohlc_data['high']
            low_prices = ohlc_data['low']
            
            # MACD Signal
            macd_signal = self._check_macd_signal(close_prices)
            
            # RSI Signal
            rsi_signal = self._check_rsi_signal(close_prices)
            
            # Moving Average Signal
            ma_signal = self._check_ma_signal(close_prices)
            
            # Bollinger Bands Signal
            bb_signal = self._check_bollinger_signal(close_prices)
            
            # Combine technical signals
            tech_signals = [s for s in [macd_signal, rsi_signal, ma_signal, bb_signal] if s]
            
            if not tech_signals:
                return None
            
            # Get signal direction (BUY or SELL)
            buy_signals = [s for s in tech_signals if s['type'] == 'BUY']
            sell_signals = [s for s in tech_signals if s['type'] == 'SELL']
            
            if len(buy_signals) > len(sell_signals):
                signal_type = 'BUY'
                signal_strength = len(buy_signals) / len(tech_signals)
            elif len(sell_signals) > len(buy_signals):
                signal_type = 'SELL'
                signal_strength = len(sell_signals) / len(tech_signals)
            else:
                return None  # Neutral
            
            # Get current price for entry
            current_price = close_prices.iloc[-1]
            atr = self.indicators.calculate_atr(high_prices, low_prices, close_prices).iloc[-1]
            
            # Calculate stop loss and take profit
            if signal_type == 'BUY':
                stop_loss = current_price - (atr * 1.0)  # 1 ATR below entry
                take_profit = current_price + (atr * 2.0)  # 2 ATR above entry
            else:  # SELL
                stop_loss = current_price + (atr * 1.0)
                take_profit = current_price - (atr * 2.0)
            
            # Get ML prediction if available
            ml_confidence = 0.5  # Default confidence
            if self.ml_predictor:
                ml_pred = self.ml_predictor.predict(ohlc_data)
                ml_confidence = ml_pred.get('confidence', 0.5)
            
            # Combine confidences
            final_confidence = (signal_strength * 0.6) + (ml_confidence * 0.4)
            
            # Only return signal if confidence is above threshold
            if final_confidence < self.config.get('confidence_threshold', 0.65):
                return None
            
            signal = {
                'instrument': instrument,
                'type': signal_type,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': final_confidence,
                'timestamp': datetime.utcnow(),
                'indicators': {
                    'macd': macd_signal,
                    'rsi': rsi_signal,
                    'ma': ma_signal,
                    'bb': bb_signal,
                    'atr': atr
                }
            }
            
            return signal
        
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            return None
    
    def _check_macd_signal(self, close_prices: pd.Series) -> Optional[Dict]:
        """Check MACD crossover signal."""
        try:
            macd, signal = self.indicators.calculate_macd(close_prices)
            
            # Need at least 2 bars to detect crossover
            if len(macd) < 2:
                return None
            
            current_macd = macd.iloc[-1]
            previous_macd = macd.iloc[-2]
            current_signal = signal.iloc[-1]
            previous_signal = signal.iloc[-2]
            
            # Bullish crossover
            if previous_macd <= previous_signal and current_macd > current_signal:
                return {'type': 'BUY', 'indicator': 'MACD', 'strength': 0.8}
            
            # Bearish crossover
            elif previous_macd >= previous_signal and current_macd < current_signal:
                return {'type': 'SELL', 'indicator': 'MACD', 'strength': 0.8}
            
            return None
        except:
            return None
    
    def _check_rsi_signal(self, close_prices: pd.Series) -> Optional[Dict]:
        """Check RSI extremes signal."""
        try:
            rsi = self.indicators.calculate_rsi(close_prices)
            current_rsi = rsi.iloc[-1]
            
            # RSI oversold (< 30)
            if current_rsi < 30:
                return {'type': 'BUY', 'indicator': 'RSI', 'strength': 0.7, 'value': current_rsi}
            
            # RSI overbought (> 70)
            elif current_rsi > 70:
                return {'type': 'SELL', 'indicator': 'RSI', 'strength': 0.7, 'value': current_rsi}
            
            return None
        except:
            return None
    
    def _check_ma_signal(self, close_prices: pd.Series) -> Optional[Dict]:
        """Check moving average crossover signal."""
        try:
            ma_fast = self.indicators.calculate_moving_average(close_prices, 20)
            ma_slow = self.indicators.calculate_moving_average(close_prices, 50)
            
            if len(ma_fast) < 2:
                return None
            
            # Bullish crossover (fast > slow)
            if ma_fast.iloc[-2] <= ma_slow.iloc[-2] and ma_fast.iloc[-1] > ma_slow.iloc[-1]:
                return {'type': 'BUY', 'indicator': 'MA', 'strength': 0.75}
            
            # Bearish crossover (fast < slow)
            elif ma_fast.iloc[-2] >= ma_slow.iloc[-2] and ma_fast.iloc[-1] < ma_slow.iloc[-1]:
                return {'type': 'SELL', 'indicator': 'MA', 'strength': 0.75}
            
            return None
        except:
            return None
    
    def _check_bollinger_signal(self, close_prices: pd.Series) -> Optional[Dict]:
        """Check Bollinger Bands breakout signal."""
        try:
            upper, middle, lower = self.indicators.calculate_bollinger_bands(close_prices)
            
            current_price = close_prices.iloc[-1]
            current_upper = upper.iloc[-1]
            current_lower = lower.iloc[-1]
            
            # Price touches lower band (potential bounce up)
            if current_price <= current_lower:
                return {'type': 'BUY', 'indicator': 'BB', 'strength': 0.65}
            
            # Price touches upper band (potential reversal down)
            elif current_price >= current_upper:
                return {'type': 'SELL', 'indicator': 'BB', 'strength': 0.65}
            
            return None
        except:
            return None
