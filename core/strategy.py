"""Strategy implementation and management."""

import logging
from typing import Dict, List, Optional
from enum import Enum
import pandas as pd
from datetime import datetime, timedelta


class StrategyState(Enum):
    """Strategy states."""
    INITIALIZED = "INITIALIZED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class Strategy:
    """Base strategy class."""
    
    def __init__(self, name: str, config: Dict):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        self.name = name
        self.config = config
        self.state = StrategyState.INITIALIZED
        self.logger = logging.getLogger(__name__)
        self.trades_opened_today = 0
        self.last_signal_time = None
        self.signal_cooldown = config.get('signal_cooldown', 300)  # 5 minutes
    
    def initialize(self) -> bool:
        """Initialize strategy. Override in subclass."""
        self.state = StrategyState.RUNNING
        self.logger.info(f"Strategy '{self.name}' initialized")
        return True
    
    def on_bar_close(self, instrument: str, ohlc_data: pd.DataFrame) -> Optional[Dict]:
        """
        Called when a candle closes. Override in subclass.
        
        Args:
            instrument: Trading instrument
            ohlc_data: OHLC data for the instrument
        
        Returns:
            Trading signal or None
        """
        return None
    
    def on_trade_open(self, trade_id: str, trade_info: Dict) -> None:
        """
        Called when a trade is opened.
        
        Args:
            trade_id: Trade identifier
            trade_info: Trade information
        """
        self.trades_opened_today += 1
    
    def on_trade_close(self, trade_id: str, trade_info: Dict) -> None:
        """
        Called when a trade is closed.
        
        Args:
            trade_id: Trade identifier
            trade_info: Trade information
        """
        pass
    
    def can_trade(self) -> bool:
        """Check if strategy can execute trades."""
        if self.state != StrategyState.RUNNING:
            return False
        
        # Check signal cooldown
        if self.last_signal_time:
            time_since_last_signal = (datetime.utcnow() - self.last_signal_time).total_seconds()
            if time_since_last_signal < self.signal_cooldown:
                return False
        
        return True
    
    def pause(self) -> None:
        """Pause strategy."""
        self.state = StrategyState.PAUSED
        self.logger.info(f"Strategy '{self.name}' paused")
    
    def resume(self) -> None:
        """Resume strategy."""
        self.state = StrategyState.RUNNING
        self.logger.info(f"Strategy '{self.name}' resumed")
    
    def stop(self) -> None:
        """Stop strategy."""
        self.state = StrategyState.STOPPED
        self.logger.info(f"Strategy '{self.name}' stopped")
    
    def get_state(self) -> Dict:
        """Get strategy state."""
        return {
            'name': self.name,
            'state': self.state.value,
            'trades_today': self.trades_opened_today,
            'last_signal': self.last_signal_time.isoformat() if self.last_signal_time else None
        }


class MultiLevelAlgorithm(Strategy):
    """
    Multi-level algorithmic trading strategy combining:
    - Technical analysis (Level 1)
    - Volume analysis (Level 2)
    - Machine learning (Level 3)
    - Risk filtering (Level 4)
    """
    
    def __init__(self, config: Dict, signal_generator, risk_manager):
        """
        Initialize Multi-Level Algorithm.
        
        Args:
            config: Strategy configuration
            signal_generator: Signal generation system
            risk_manager: Risk management system
        """
        super().__init__("MultiLevelAlgo", config)
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.last_signals = {}  # Cache recent signals
        self.signal_history = []  # Track signal history for analysis
    
    def on_bar_close(self, instrument: str, ohlc_data: pd.DataFrame) -> Optional[Dict]:
        """
        Generate trading signal when candle closes.
        
        Args:
            instrument: Trading instrument (e.g., 'EURUSD')
            ohlc_data: OHLC data
        
        Returns:
            Trading signal or None
        """
        try:
            # Check if we can trade
            if not self.can_trade():
                return None
            
            # Generate signal from technical and ML analysis
            signal = self.signal_generator.generate_signal(ohlc_data, instrument)
            
            if signal is None:
                return None
            
            # Log signal
            self.last_signal_time = datetime.utcnow()
            self.last_signals[instrument] = signal
            self.signal_history.append({
                'timestamp': signal['timestamp'],
                'instrument': instrument,
                'type': signal['type'],
                'confidence': signal['confidence']
            })
            
            self.logger.info(
                f"Signal generated for {instrument}: {signal['type']} "
                f"(Confidence: {signal['confidence']:.2%})"
            )
            
            return signal
        
        except Exception as e:
            self.logger.error(f"Error in on_bar_close: {str(e)}")
            return None
    
    def get_performance(self) -> Dict:
        """Get strategy performance metrics."""
        if not self.signal_history:
            return {}
        
        recent_signals = self.signal_history[-100:]  # Last 100 signals
        buy_signals = [s for s in recent_signals if s['type'] == 'BUY']
        sell_signals = [s for s in recent_signals if s['type'] == 'SELL']
        
        return {
            'total_signals': len(recent_signals),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'avg_confidence': sum(s['confidence'] for s in recent_signals) / len(recent_signals),
            'instruments_traded': len(set(s['instrument'] for s in recent_signals))
        }
