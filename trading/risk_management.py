# trading/risk_management.py
"""
Risk management and position sizing module.
Calculates stop-loss, take-profit, and position sizes based on various methods.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class RiskParams:
    """Risk management parameters."""
    max_risk_per_trade: float = 0.02  # 2% max risk per trade
    profit_target_multiplier: float = 2.0  # Risk:Reward ratio
    atr_stop_multiplier: float = 2.0  # ATR-based stop loss
    volatility_target: float = 0.15  # Target portfolio volatility
    max_position_size: float = 0.10  # Max 10% per position
    min_profit_target: float = 0.005  # Min 0.5% profit target


def calculate_atr_based_levels(
    current_price: float,
    atr_value: float,
    direction: str,  # 'long' or 'short'
    multiplier: float = 2.0
) -> Dict[str, float]:
    """Calculate stop-loss and take-profit based on ATR."""
    
    if direction.lower() == 'long':
        stop_loss = current_price - (atr_value * multiplier)
        # Risk amount
        risk_amount = current_price - stop_loss
        take_profit = current_price + (risk_amount * 2.0)  # 2:1 R:R
    else:  # short
        stop_loss = current_price + (atr_value * multiplier)
        risk_amount = stop_loss - current_price
        take_profit = current_price - (risk_amount * 2.0)
    
    return {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_amount': risk_amount,
        'reward_amount': abs(take_profit - current_price)
    }


def calculate_volatility_based_levels(
    current_price: float,
    volatility: float,  # annualized volatility
    direction: str,
    confidence_level: float = 2.0  # 2 standard deviations
) -> Dict[str, float]:
    """Calculate levels based on volatility (daily moves)."""
    
    # Convert annual vol to daily
    daily_vol = volatility / np.sqrt(252)
    price_move = current_price * daily_vol * confidence_level
    
    if direction.lower() == 'long':
        stop_loss = current_price - price_move
        risk_amount = price_move
        take_profit = current_price + (price_move * 2.0)
    else:
        stop_loss = current_price + price_move  
        risk_amount = price_move
        take_profit = current_price - (price_move * 2.0)
    
    return {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_amount': risk_amount,
        'reward_amount': price_move * 2.0
    }


def calculate_support_resistance_levels(
    price_history: pd.Series,
    current_price: float,
    direction: str,
    lookback: int = 20
) -> Dict[str, float]:
    """Calculate levels based on recent support/resistance."""
    
    recent_prices = price_history.tail(lookback)
    
    if direction.lower() == 'long':
        # Use recent low as support for stop loss
        support = recent_prices.min()
        stop_loss = support * 0.98  # Small buffer below support
        risk_amount = current_price - stop_loss
        
        # Use recent high resistance area for take profit  
        resistance = recent_prices.max()
        take_profit = max(current_price + (risk_amount * 2.0), resistance * 1.02)
    else:
        # Use recent high as resistance for stop loss
        resistance = recent_prices.max() 
        stop_loss = resistance * 1.02
        risk_amount = stop_loss - current_price
        
        # Use recent low support area for take profit
        support = recent_prices.min()
        take_profit = min(current_price - (risk_amount * 2.0), support * 0.98)
    
    return {
        'stop_loss': stop_loss,
        'take_profit': take_profit, 
        'risk_amount': risk_amount,
        'reward_amount': abs(take_profit - current_price)
    }


def calculate_position_size(
    account_balance: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss: float,
    max_position_pct: float = 0.10
) -> Dict[str, float]:
    """Calculate position size based on risk management rules."""
    
    # Risk amount per share
    risk_per_share = abs(entry_price - stop_loss)
    
    # Maximum dollar risk for this trade
    max_risk_dollars = account_balance * risk_per_trade
    
    # Position size based on risk
    risk_based_shares = max_risk_dollars / risk_per_share
    risk_based_position_value = risk_based_shares * entry_price
    
    # Position size based on max position percentage
    max_position_value = account_balance * max_position_pct
    max_position_shares = max_position_value / entry_price
    
    # Take the smaller of the two
    final_shares = min(risk_based_shares, max_position_shares)
    final_position_value = final_shares * entry_price
    final_risk_dollars = final_shares * risk_per_share
    final_risk_pct = final_risk_dollars / account_balance
    
    return {
        'shares': int(final_shares),
        'position_value': final_position_value,
        'risk_dollars': final_risk_dollars,
        'risk_percentage': final_risk_pct,
        'position_percentage': final_position_value / account_balance
    }


class RiskManager:
    """Main risk management class."""
    
    def __init__(self, params: RiskParams):
        self.params = params
    
    def calculate_trade_levels(
        self,
        symbol: str,
        current_price: float,
        direction: str,
        atr_value: Optional[float] = None,
        volatility: Optional[float] = None,
        price_history: Optional[pd.Series] = None,
        method: str = 'atr'
    ) -> Dict[str, float]:
        """Calculate stop-loss and take-profit levels using specified method."""
        
        if method == 'atr' and atr_value is not None:
            return calculate_atr_based_levels(
                current_price, atr_value, direction, self.params.atr_stop_multiplier
            )
        elif method == 'volatility' and volatility is not None:
            return calculate_volatility_based_levels(
                current_price, volatility, direction
            )
        elif method == 'support_resistance' and price_history is not None:
            return calculate_support_resistance_levels(
                price_history, current_price, direction
            )
        else:
            # Fallback to simple percentage-based levels
            risk_pct = 0.02  # 2% risk
            if direction.lower() == 'long':
                stop_loss = current_price * (1 - risk_pct)
                take_profit = current_price * (1 + risk_pct * self.params.profit_target_multiplier)
            else:
                stop_loss = current_price * (1 + risk_pct)
                take_profit = current_price * (1 - risk_pct * self.params.profit_target_multiplier)
            
            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_amount': current_price * risk_pct,
                'reward_amount': current_price * risk_pct * self.params.profit_target_multiplier
            }
    
    def size_position(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float
    ) -> Dict[str, float]:
        """Calculate position size for a trade."""
        return calculate_position_size(
            account_balance,
            self.params.max_risk_per_trade,
            entry_price,
            stop_loss,
            self.params.max_position_size
        )
    
    def validate_trade(
        self,
        levels: Dict[str, float],
        position_info: Dict[str, float]
    ) -> Tuple[bool, str]:
        """Validate if a trade meets risk management criteria."""
        
        # Check minimum profit target
        reward_risk_ratio = levels['reward_amount'] / levels['risk_amount']
        if reward_risk_ratio < 1.5:
            return False, f"Risk:Reward ratio too low: {reward_risk_ratio:.2f}"
        
        # Check position size
        if position_info['position_percentage'] > self.params.max_position_size:
            return False, f"Position size too large: {position_info['position_percentage']:.1%}"
        
        # Check risk percentage
        if position_info['risk_percentage'] > self.params.max_risk_per_trade:
            return False, f"Risk per trade too high: {position_info['risk_percentage']:.1%}"
        
        return True, "Trade validation passed"


# Example usage
if __name__ == "__main__":
    # Example trade setup
    risk_params = RiskParams(
        max_risk_per_trade=0.02,
        profit_target_multiplier=2.0,
        atr_stop_multiplier=2.0
    )
    
    rm = RiskManager(risk_params)
    
    # Calculate levels for a long trade
    current_price = 150.0
    atr = 3.5
    account_balance = 100000.0
    
    levels = rm.calculate_trade_levels(
        symbol="AAPL",
        current_price=current_price,
        direction="long",
        atr_value=atr,
        method="atr"
    )
    
    position = rm.size_position(account_balance, current_price, levels['stop_loss'])
    
    is_valid, message = rm.validate_trade(levels, position)
    
    print(f"Trade Levels: {levels}")
    print(f"Position Info: {position}")
    print(f"Valid Trade: {is_valid} - {message}")