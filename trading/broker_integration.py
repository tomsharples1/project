# trading/broker_integration.py
"""
Broker integration for executing trades.
Supports Alpaca (paper trading), Interactive Brokers, and a local simulator.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import time
from abc import ABC, abstractmethod

# For Alpaca integration (install: pip install alpaca-trade-api)
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: alpaca-trade-api not installed. Install with: pip install alpaca-trade-api")

# For Interactive Brokers (install: pip install ib-insync)
try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    print("Warning: ib-insync not installed. Install with: pip install ib-insync")


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled" 
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Order:
    """Order representation."""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: Optional[float] = None
    timestamp: datetime = None
    broker_order_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass  
class Position:
    """Position representation."""
    symbol: str
    quantity: int
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0


class BrokerInterface(ABC):
    """Abstract base class for broker interfaces."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from broker."""
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict:
        """Get account information."""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass
    
    @abstractmethod
    def get_buying_power(self) -> float:
        """Get available buying power."""
        pass
    
    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Submit order and return broker order ID."""
        pass
    
    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order by broker ID."""
        pass
    
    @abstractmethod
    def get_order_status(self, broker_order_id: str) -> OrderStatus:
        """Get order status by broker ID."""
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        pass


class AlpacaBroker(BrokerInterface):
    """Alpaca broker interface for paper trading."""
    
    def __init__(self):
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-trade-api not installed")
        
        self.api_key = 'PK04F356F2M6A5W45FYA' 
        self.secret_key = 'V9UwmBmdBtwBG9v0aYFCind1xbZBKt9KcIEQFSUe'
        self.api = None
        
    def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            base_url = 'https://paper-api.alpaca.markets'
            self.api = tradeapi.REST(self.api_key, self.secret_key, base_url, api_version='v2')
            
            # Test connection
            account = self.api.get_account()
            print(f"Connected to Alpaca. Account status: {account.status}")
            return True
        except Exception as e:
            print(f"Failed to connect to Alpaca: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Alpaca."""
        self.api = None
    
    def get_account_info(self) -> Dict:
        """Get account information."""
        if not self.api:
            raise ConnectionError("Not connected to broker")
        
        account = self.api.get_account()
        return {
            'equity': float(account.equity),
            'buying_power': float(account.buying_power),
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'day_trade_count': int(account.daytrade_count),
            'pattern_day_trader': account.pattern_day_trader
        }
    
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        if not self.api:
            raise ConnectionError("Not connected to broker")
        
        positions = []
        for pos in self.api.list_positions():
            positions.append(Position(
                symbol=pos.symbol,
                quantity=int(pos.qty),
                avg_price=float(pos.avg_cost),
                market_value=float(pos.market_value),
                unrealized_pnl=float(pos.unrealized_pl)
            ))
        return positions
    
    def get_buying_power(self) -> float:
        """Get available buying power."""
        account = self.api.get_account()
        return float(account.buying_power)
    
    def submit_order(self, order: Order) -> str:
        """Submit order to Alpaca."""
        if not self.api:
            raise ConnectionError("Not connected to broker")
        
        # Convert order type
        if order.order_type == OrderType.MARKET:
            alpaca_order = self.api.submit_order(
                symbol=order.symbol,
                qty=order.quantity,
                side=order.side,
                type='market',
                time_in_force='day'
            )
        elif order.order_type == OrderType.LIMIT:
            alpaca_order = self.api.submit_order(
                symbol=order.symbol,
                qty=order.quantity,
                side=order.side,
                type='limit',
                limit_price=order.price,
                time_in_force='day'
            )
        elif order.order_type == OrderType.STOP:
            alpaca_order = self.api.submit_order(
                symbol=order.symbol,
                qty=order.quantity,
                side=order.side,
                type='stop',
                stop_price=order.stop_price,
                time_in_force='day'
            )
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")
        
        return alpaca_order.id
    
    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order."""
        try:
            self.api.cancel_order(broker_order_id)
            return True
        except Exception as e:
            print(f"Failed to cancel order {broker_order_id}: {e}")
            return False
    
    def get_order_status(self, broker_order_id: str) -> OrderStatus:
        """Get order status."""
        try:
            order = self.api.get_order(broker_order_id)
            status_map = {
                'new': OrderStatus.PENDING,
                'partially_filled': OrderStatus.PARTIALLY_FILLED,
                'filled': OrderStatus.FILLED,
                'done_for_day': OrderStatus.CANCELLED,
                'canceled': OrderStatus.CANCELLED,
                'expired': OrderStatus.CANCELLED,
                'replaced': OrderStatus.CANCELLED,
                'pending_cancel': OrderStatus.PENDING,
                'pending_replace': OrderStatus.PENDING,
                'accepted': OrderStatus.PENDING,
                'pending_new': OrderStatus.PENDING,
                'accepted_for_bidding': OrderStatus.PENDING,
                'stopped': OrderStatus.CANCELLED,
                'rejected': OrderStatus.REJECTED,
                'suspended': OrderStatus.CANCELLED
            }
            return status_map.get(order.status, OrderStatus.PENDING)
        except Exception as e:
            print(f"Failed to get order status for {broker_order_id}: {e}")
            return OrderStatus.REJECTED
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        try:
            quote = self.api.get_latest_trade(symbol)
            return float(quote.price)
        except Exception as e:
            print(f"Failed to get price for {symbol}: {e}")
            return 0.0


class SimulatedBroker(BrokerInterface):
    """Simulated broker for backtesting and testing."""
    
    def __init__(self, initial_cash: float = 100000.0, commission: float = 0.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission = commission
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0
        self.connected = False
        self.price_data: Dict[str, float] = {}  # Symbol -> current price
        
    def connect(self) -> bool:
        """Connect to simulated broker."""
        self.connected = True
        return True
    
    def disconnect(self):
        """Disconnect from simulated broker."""
        self.connected = False
    
    def set_price(self, symbol: str, price: float):
        """Set current price for a symbol (for simulation)."""
        self.price_data[symbol] = price
        
        # Update position market values
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.market_value = pos.quantity * price
            pos.unrealized_pnl = (price - pos.avg_price) * pos.quantity
    
    def get_account_info(self) -> Dict:
        """Get simulated account information."""
        if not self.connected:
            raise ConnectionError("Not connected to broker")
        
        total_value = self.cash
        for pos in self.positions.values():
            total_value += pos.market_value
        
        return {
            'equity': total_value,
            'buying_power': self.cash,
            'cash': self.cash,
            'portfolio_value': total_value,
            'day_trade_count': 0,
            'pattern_day_trader': False
        }
    
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        return list(self.positions.values())
    
    def get_buying_power(self) -> float:
        """Get available buying power."""
        return self.cash
    
    def submit_order(self, order: Order) -> str:
        """Submit order to simulated broker."""
        if not self.connected:
            raise ConnectionError("Not connected to broker")
        
        self.order_counter += 1
        broker_order_id = f"SIM_{self.order_counter:06d}"
        
        # Store order
        order.broker_order_id = broker_order_id
        self.orders[broker_order_id] = order
        
        # For simulation, immediately try to fill market orders
        if order.order_type == OrderType.MARKET:
            self._try_fill_order(broker_order_id)
        
        return broker_order_id
    
    def _try_fill_order(self, broker_order_id: str):
        """Try to fill an order (simulation)."""
        order = self.orders[broker_order_id]
        current_price = self.price_data.get(order.symbol, 0.0)
        
        if current_price == 0.0:
            order.status = OrderStatus.REJECTED
            return
        
        # Check if we can afford the order
        total_cost = order.quantity * current_price + self.commission
        if order.side == 'buy' and total_cost > self.cash:
            order.status = OrderStatus.REJECTED
            return
        
        # Fill the order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = current_price
        
        if order.side == 'buy':
            self.cash -= total_cost
            
            # Update position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                new_qty = pos.quantity + order.quantity
                new_avg = (pos.avg_price * pos.quantity + current_price * order.quantity) / new_qty
                pos.quantity = new_qty
                pos.avg_price = new_avg
                pos.market_value = new_qty * current_price
                pos.unrealized_pnl = (current_price - new_avg) * new_qty
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_price=current_price,
                    market_value=order.quantity * current_price,
                    unrealized_pnl=0.0
                )
        
        else:  # sell
            if order.symbol not in self.positions or self.positions[order.symbol].quantity < order.quantity:
                order.status = OrderStatus.REJECTED
                return
            
            pos = self.positions[order.symbol]
            realized_pnl = (current_price - pos.avg_price) * order.quantity
            self.cash += order.quantity * current_price - self.commission
            
            # Update position
            pos.quantity -= order.quantity
            pos.realized_pnl += realized_pnl
            pos.market_value = pos.quantity * current_price
            pos.unrealized_pnl = (current_price - pos.avg_price) * pos.quantity
            
            # Remove position if quantity is 0
            if pos.quantity == 0:
                del self.positions[order.symbol]
    
    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order."""
        if broker_order_id in self.orders:
            order = self.orders[broker_order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                return True
        return False
    
    def get_order_status(self, broker_order_id: str) -> OrderStatus:
        """Get order status."""
        if broker_order_id in self.orders:
            return self.orders[broker_order_id].status
        return OrderStatus.REJECTED
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        return self.price_data.get(symbol, 0.0)


class TradingEngine:
    """Main trading engine that coordinates signals, risk management, and execution."""
    
    def __init__(self, broker: BrokerInterface):
        self.broker = broker
        self.active_orders: Dict[str, Order] = {}
        self.positions_tracking: Dict[str, Dict] = {}  # Track our own position info
        
    def connect(self) -> bool:
        """Connect to broker."""
        return self.broker.connect()
    
    def disconnect(self):
        """Disconnect from broker."""
        self.broker.disconnect()
    
    def execute_signal(self, signal) -> Optional[str]:
        """Execute a trading signal."""
        from trading.signals import SignalType
        
        if signal.signal in [SignalType.HOLD]:
            return None
        
        # Create entry order
        side = 'buy' if signal.signal in [SignalType.BUY, SignalType.STRONG_BUY] else 'sell'
        
        entry_order = Order(
            id=f"{signal.symbol}_{int(signal.timestamp.timestamp())}",
            symbol=signal.symbol,
            side=side,
            quantity=signal.position_size,
            order_type=OrderType.MARKET
        )
        
        try:
            broker_order_id = self.broker.submit_order(entry_order)
            self.active_orders[broker_order_id] = entry_order
            
            # Track position for stop loss and take profit
            self.positions_tracking[signal.symbol] = {
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'quantity': signal.position_size,
                'side': side,
                'entry_order_id': broker_order_id
            }
            
            print(f"Executed {side} order for {signal.symbol}: {signal.position_size} shares")
            return broker_order_id
            
        except Exception as e:
            print(f"Failed to execute signal for {signal.symbol}: {e}")
            return None
    
    def check_exit_conditions(self):
        """Check if any positions need to be closed based on stop loss or take profit."""
        positions = self.broker.get_positions()
        
        for position in positions:
            if position.symbol in self.positions_tracking:
                tracking = self.positions_tracking[position.symbol]
                current_price = self.broker.get_current_price(position.symbol)
                
                should_exit = False
                exit_reason = ""
                
                # Check stop loss and take profit based on position side
                if tracking['side'] == 'buy':  # Long position
                    if current_price <= tracking['stop_loss']:
                        should_exit = True
                        exit_reason = "Stop Loss"
                    elif current_price >= tracking['take_profit']:
                        should_exit = True
                        exit_reason = "Take Profit"
                
                else:  # Short position
                    if current_price >= tracking['stop_loss']:
                        should_exit = True
                        exit_reason = "Stop Loss"
                    elif current_price <= tracking['take_profit']:
                        should_exit = True
                        exit_reason = "Take Profit"
                
                if should_exit:
                    self._exit_position(position.symbol, exit_reason)
    
    def _exit_position(self, symbol: str, reason: str):
        """Exit a position."""
        try:
            tracking = self.positions_tracking[symbol]
            exit_side = 'sell' if tracking['side'] == 'buy' else 'buy'
            
            exit_order = Order(
                id=f"{symbol}_exit_{int(datetime.now().timestamp())}",
                symbol=symbol,
                side=exit_side,
                quantity=abs(tracking['quantity']),
                order_type=OrderType.MARKET
            )
            
            broker_order_id = self.broker.submit_order(exit_order)
            print(f"Exited position {symbol} ({reason}): {exit_order.quantity} shares")
            
            # Remove from tracking
            del self.positions_tracking[symbol]
            
        except Exception as e:
            print(f"Failed to exit position {symbol}: {e}")
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary."""
        account = self.broker.get_account_info()
        positions = self.broker.get_positions()
        
        return {
            'account': account,
            'positions': [asdict(pos) for pos in positions],
            'active_orders': len(self.active_orders),
            'tracked_positions': len(self.positions_tracking)
        }


# Example configuration and usage
def setup_alpaca_paper_trading(): 
    broker = AlpacaBroker()
    return TradingEngine(broker)


def setup_simulated_trading(initial_cash: float = 100000.0):
    """Setup simulated trading for testing."""
    broker = SimulatedBroker(initial_cash=initial_cash)
    return TradingEngine(broker)


if __name__ == "__main__":
    # Example with simulated broker
    engine = setup_simulated_trading()
    
    if engine.connect():
        print("Connected to simulated broker")
        
        # Set some prices for testing
        engine.broker.set_price("AAPL", 150.0)
        engine.broker.set_price("TSLA", 200.0)
        
        # Get account info
        account = engine.broker.get_account_info()
        print(f"Account equity: ${account['equity']:,.2f}")
        
        # Create a test order
        test_order = Order(
            id="test_1",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        order_id = engine.broker.submit_order(test_order)
        print(f"Submitted order: {order_id}")
        
        # Check positions
        positions = engine.broker.get_positions()
        for pos in positions:
            print(f"Position: {pos.symbol} - {pos.quantity} shares @ ${pos.avg_price:.2f}")
        
        engine.disconnect()
    else:
        print("Failed to connect to broker")