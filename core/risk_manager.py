"""Risk management system for forex trading."""

import logging
from typing import Dict, Optional, Tuple
import numpy as np


class RiskManager:
    """Manages risk parameters and position sizing."""
    
    def __init__(self, config: Dict):
        """
        Initialize risk manager.
        
        Args:
            config: Configuration dictionary with risk parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk parameters
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2%
        self.max_position_size = config.get('max_position_size', 100000)
        self.max_daily_loss = config.get('max_daily_loss', 0.05)  # 5%
        self.max_open_trades = config.get('max_open_trades', 5)
        self.max_leverage = config.get('leverage', 10)
        self.daily_loss = 0.0
    
    def validate_trade(self, signal: Dict, current_equity: float) -> Dict:
        """
        Validate if a trade should be executed based on risk rules.
        
        Args:
            signal: Trading signal dictionary
            current_equity: Current account equity
        
        Returns:
            Validation result with 'valid' (bool) and 'reason' (str)
        """
        validation_result = {
            'valid': True,
            'reason': 'Trade approved',
            'checks': {}
        }
        
        # Check daily loss limit
        daily_loss_limit = current_equity * self.max_daily_loss
        if abs(self.daily_loss) >= daily_loss_limit:
            validation_result['valid'] = False
            validation_result['reason'] = f"Daily loss limit exceeded: ${abs(self.daily_loss):.2f}/${daily_loss_limit:.2f}"
            validation_result['checks']['daily_loss'] = False
            return validation_result
        validation_result['checks']['daily_loss'] = True
        
        # Check position size
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        risk_amount = abs(entry_price - stop_loss)
        
        if risk_amount <= 0:
            validation_result['valid'] = False
            validation_result['reason'] = "Invalid stop loss placement"
            validation_result['checks']['sl_placement'] = False
            return validation_result
        validation_result['checks']['sl_placement'] = True
        
        # Check confidence level
        min_confidence = self.config.get('confidence_threshold', 0.65)
        if signal.get('confidence', 0) < min_confidence:
            validation_result['valid'] = False
            validation_result['reason'] = f"Confidence too low: {signal.get('confidence', 0):.2f} < {min_confidence}"
            validation_result['checks']['confidence'] = False
            return validation_result
        validation_result['checks']['confidence'] = True
        
        # All checks passed
        return validation_result
    
    def calculate_position_size(self, equity: float, entry_price: float,
                               stop_loss: float, risk_percentage: Optional[float] = None) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            equity: Current account equity
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_percentage: Risk percentage (uses config default if None)
        
        Returns:
            Position size in volume/lots
        """
        if risk_percentage is None:
            risk_percentage = self.risk_per_trade
        
        # Risk amount in dollars
        risk_amount = equity * risk_percentage
        
        # Pips at risk
        pips_at_risk = abs(entry_price - stop_loss) / 0.0001
        
        if pips_at_risk <= 0:
            self.logger.warning("Invalid pips at risk calculation")
            return 0.0
        
        # Lot size: Risk Amount / (Pips at Risk * Pip Value)
        # For EURUSD, 1 standard lot = $10 per pip
        pip_value = 10.0  # Standard pip value
        
        position_size = risk_amount / (pips_at_risk * (pip_value / 100000))
        
        # Enforce max position size
        position_size = min(position_size, self.max_position_size)
        
        # Round to nearest 0.01 lot
        position_size = round(position_size, 2)
        
        return position_size
    
    def update_daily_loss(self, trade_pnl: float) -> None:
        """
        Update daily loss tracking.
        
        Args:
            trade_pnl: Profit/loss from closed trade
        """
        self.daily_loss += trade_pnl
    
    def reset_daily_loss(self) -> None:
        """Reset daily loss counter (call at market open)."""
        self.daily_loss = 0.0
    
    def calculate_margin_required(self, position_size: float, entry_price: float,
                                 leverage: Optional[float] = None) -> float:
        """
        Calculate margin required for a position.
        
        Args:
            position_size: Position size in lots
            entry_price: Entry price
            leverage: Leverage ratio (uses config default if None)
        
        Returns:
            Margin required in dollars
        """
        if leverage is None:
            leverage = self.max_leverage
        
        # For EURUSD: 1 standard lot = 100,000 units
        notional_value = position_size * 100000 * entry_price
        margin_required = notional_value / leverage
        
        return margin_required
    
    def validate_margin(self, position_size: float, entry_price: float,
                       available_margin: float) -> bool:
        """
        Check if sufficient margin is available for position.
        
        Args:
            position_size: Position size in lots
            entry_price: Entry price
            available_margin: Available margin in account
        
        Returns:
            True if sufficient margin, False otherwise
        """
        margin_required = self.calculate_margin_required(position_size, entry_price)
        return margin_required <= available_margin * 0.8  # Use 80% safety factor
    
    def get_risk_metrics(self, open_trades: list, current_prices: Dict,
                        equity: float) -> Dict:
        """
        Calculate current risk metrics.
        
        Args:
            open_trades: List of open trade objects
            current_prices: Dictionary of current prices by instrument
            equity: Current equity
        
        Returns:
            Risk metrics dictionary
        """
        if not open_trades:
            return {
                'total_risk': 0.0,
                'risk_percentage': 0.0,
                'unrealized_pnl': 0.0,
                'max_adverse_move': 0.0,
                'correlation_risk': 'Low'
            }
        
        total_risk = 0.0
        unrealized_pnl = 0.0
        max_adverse_move = 0.0
        
        for trade in open_trades:
            current_price = current_prices.get(trade.instrument, trade.entry_price)
            
            # Calculate unrealized P&L
            if trade.order_type.value == 'BUY':
                unrealized = (current_price - trade.entry_price) * trade.volume
                adverse_move = trade.entry_price - trade.stop_loss
            else:  # SELL
                unrealized = (trade.entry_price - current_price) * trade.volume
                adverse_move = trade.stop_loss - trade.entry_price
            
            unrealized_pnl += unrealized
            risk = abs(adverse_move * trade.volume)
            total_risk += risk
            max_adverse_move = max(max_adverse_move, adverse_move)
        
        return {
            'total_risk': total_risk,
            'risk_percentage': (total_risk / equity) * 100,
            'unrealized_pnl': unrealized_pnl,
            'max_adverse_move': max_adverse_move,
            'num_open_trades': len(open_trades),
            'risk_status': 'HIGH' if (total_risk / equity) > 0.05 else 'NORMAL'
        }
