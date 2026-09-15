"""Main trading engine for executing trades and managing positions."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from enum import Enum


class OrderType(Enum):
    """Order types."""
    BUY = "BUY"
    SELL = "SELL"
    BUY_STOP = "BUY_STOP"
    SELL_STOP = "SELL_STOP"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class Trade:
    """Represents a single trade."""
    
    def __init__(self, trade_id: str, instrument: str, order_type: OrderType,
                 entry_price: float, volume: float, entry_time: datetime):
        self.trade_id = trade_id
        self.instrument = instrument
        self.order_type = order_type
        self.entry_price = entry_price
        self.volume = volume
        self.entry_time = entry_time
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[datetime] = None
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.status = OrderStatus.OPEN
        self.profit_loss: Optional[float] = None
        self.profit_loss_pips: Optional[float] = None
        self.win = False
    
    def close(self, exit_price: float, exit_time: datetime) -> None:
        """Close the trade."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = OrderStatus.CLOSED
        
        # Calculate P&L
        if self.order_type == OrderType.BUY:
            self.profit_loss = (exit_price - self.entry_price) * self.volume
            self.profit_loss_pips = (exit_price - self.entry_price) / 0.0001
        else:  # SELL
            self.profit_loss = (self.entry_price - exit_price) * self.volume
            self.profit_loss_pips = (self.entry_price - exit_price) / 0.0001
        
        self.win = self.profit_loss > 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'trade_id': self.trade_id,
            'instrument': self.instrument,
            'type': self.order_type.value,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'volume': self.volume,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'profit_loss': self.profit_loss,
            'profit_loss_pips': self.profit_loss_pips,
            'status': self.status.value,
            'win': self.win
        }


class TradingEngine:
    """Main trading engine orchestrating all trading operations."""
    
    def __init__(self, broker, risk_manager, signal_generator, config: Dict):
        """
        Initialize trading engine.
        
        Args:
            broker: Broker connection object
            risk_manager: Risk management system
            signal_generator: Signal generation system
            config: Configuration dictionary
        """
        self.broker = broker
        self.risk_manager = risk_manager
        self.signal_generator = signal_generator
        self.config = config
        
        self.logger = logging.getLogger(__name__)
        self.trades: Dict[str, Trade] = {}
        self.open_positions: Dict[str, Trade] = {}
        self.trade_counter = 0
        self.account_balance = 0.0
        self.equity = 0.0
        self.margin_used = 0.0
        self.margin_available = 0.0
    
    def initialize(self) -> bool:
        """Initialize trading engine and connect to broker."""
        try:
            # Connect to broker
            if not self.broker.connect():
                self.logger.error("Failed to connect to broker")
                return False
            
            # Get account information
            account_info = self.broker.get_account_info()
            self.account_balance = account_info['balance']
            self.equity = account_info['equity']
            self.margin_available = account_info['margin_available']
            
            self.logger.info(f"Trading engine initialized. Balance: {self.account_balance}")
            return True
        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            return False
    
    def update_account_info(self) -> None:
        """Update account information from broker."""
        try:
            account_info = self.broker.get_account_info()
            self.account_balance = account_info['balance']
            self.equity = account_info['equity']
            self.margin_used = account_info['margin_used']
            self.margin_available = account_info['margin_available']
        except Exception as e:
            self.logger.error(f"Failed to update account info: {str(e)}")
    
    def process_signal(self, signal: Dict) -> Optional[str]:
        """
        Process a trading signal and execute trade if conditions are met.
        
        Args:
            signal: Signal dictionary with:
                - instrument: Trading instrument (e.g., 'EURUSD')
                - type: 'BUY' or 'SELL'
                - confidence: Confidence level 0-1
                - entry_price: Entry price
                - stop_loss: Stop loss price
                - take_profit: Take profit price
        
        Returns:
            Trade ID if trade opened, None otherwise
        """
        try:
            # Check if we should open a trade
            risk_check = self.risk_manager.validate_trade(signal, self.equity)
            
            if not risk_check['valid']:
                self.logger.info(f"Trade rejected: {risk_check['reason']}")
                return None
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                self.equity,
                signal['entry_price'],
                signal['stop_loss'],
                self.config.get('risk_per_trade', 0.02)
            )
            
            # Execute trade
            trade_id = self._execute_trade(
                instrument=signal['instrument'],
                order_type=OrderType[signal['type']],
                entry_price=signal['entry_price'],
                volume=position_size,
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit']
            )
            
            self.logger.info(f"Trade opened: {trade_id}")
            return trade_id
        
        except Exception as e:
            self.logger.error(f"Error processing signal: {str(e)}")
            return None
    
    def _execute_trade(self, instrument: str, order_type: OrderType,
                      entry_price: float, volume: float,
                      stop_loss: float, take_profit: float) -> str:
        """Execute a trade with broker."""
        # Create trade object
        trade_id = f"TRADE_{self.trade_counter:06d}"
        self.trade_counter += 1
        
        trade = Trade(
            trade_id=trade_id,
            instrument=instrument,
            order_type=order_type,
            entry_price=entry_price,
            volume=volume,
            entry_time=datetime.utcnow()
        )
        trade.stop_loss = stop_loss
        trade.take_profit = take_profit
        
        # Send to broker
        order_result = self.broker.open_trade(
            instrument=instrument,
            order_type=order_type.value,
            volume=volume,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        if order_result['success']:
            self.trades[trade_id] = trade
            self.open_positions[trade_id] = trade
            self.update_account_info()
            return trade_id
        else:
            self.logger.error(f"Failed to execute trade: {order_result['error']}")
            raise Exception(order_result['error'])
    
    def check_exit_conditions(self) -> None:
        """Check exit conditions for all open trades."""
        trades_to_close = []
        
        for trade_id, trade in self.open_positions.items():
            # Get current price
            current_price = self.broker.get_current_price(trade.instrument)
            
            # Check stop loss
            if trade.stop_loss and self._check_stop_loss(trade, current_price):
                trades_to_close.append((trade_id, current_price, 'stop_loss'))
            
            # Check take profit
            elif trade.take_profit and self._check_take_profit(trade, current_price):
                trades_to_close.append((trade_id, current_price, 'take_profit'))
        
        # Close trades
        for trade_id, exit_price, reason in trades_to_close:
            self.close_trade(trade_id, exit_price, reason)
    
    def _check_stop_loss(self, trade: Trade, current_price: float) -> bool:
        """Check if stop loss is hit."""
        if trade.order_type == OrderType.BUY:
            return current_price <= trade.stop_loss
        else:  # SELL
            return current_price >= trade.stop_loss
    
    def _check_take_profit(self, trade: Trade, current_price: float) -> bool:
        """Check if take profit is hit."""
        if trade.order_type == OrderType.BUY:
            return current_price >= trade.take_profit
        else:  # SELL
            return current_price <= trade.take_profit
    
    def close_trade(self, trade_id: str, exit_price: float, reason: str = 'manual') -> None:
        """Close an open trade."""
        if trade_id not in self.open_positions:
            self.logger.warning(f"Trade {trade_id} not found")
            return
        
        trade = self.open_positions[trade_id]
        trade.close(exit_price, datetime.utcnow())
        
        # Close with broker
        self.broker.close_trade(trade_id, exit_price)
        
        # Remove from open positions
        del self.open_positions[trade_id]
        
        self.update_account_info()
        
        self.logger.info(
            f"Trade closed: {trade_id} | "
            f"P&L: {trade.profit_loss:.2f} | "
            f"Reason: {reason}"
        )
    
    def get_open_trades(self) -> List[Dict]:
        """Get all open trades."""
        return [trade.to_dict() for trade in self.open_positions.values()]
    
    def get_trade_history(self) -> List[Dict]:
        """Get all trades (open and closed)."""
        return [trade.to_dict() for trade in self.trades.values()]
    
    def get_performance_stats(self) -> Dict:
        """Calculate performance statistics."""
        closed_trades = [t for t in self.trades.values() if t.status == OrderStatus.CLOSED]
        
        if not closed_trades:
            return {}
        
        wins = [t for t in closed_trades if t.win]
        losses = [t for t in closed_trades if not t.win]
        
        total_pnl = sum(t.profit_loss for t in closed_trades)
        win_rate = len(wins) / len(closed_trades)
        avg_win = sum(t.profit_loss for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.profit_loss for t in losses) / len(losses) if losses else 0
        profit_factor = abs(sum(t.profit_loss for t in wins) / sum(t.profit_loss for t in losses)) if losses else 0
        
        return {
            'total_trades': len(closed_trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': abs(avg_loss),
            'profit_factor': profit_factor,
            'largest_win': max((t.profit_loss for t in wins), default=0),
            'largest_loss': min((t.profit_loss for t in losses), default=0)
        }
    
    def shutdown(self) -> None:
        """Shutdown trading engine."""
        # Close all open positions
        for trade_id in list(self.open_positions.keys()):
            current_price = self.broker.get_current_price(
                self.open_positions[trade_id].instrument
            )
            self.close_trade(trade_id, current_price, reason='shutdown')
        
        # Disconnect from broker
        self.broker.disconnect()
        self.logger.info("Trading engine shutdown complete")
