# main_trading_system.py
"""
Main orchestrator for the complete algorithmic trading system.
Integrates data fetching, feature engineering, ML prediction, signal generation,
risk management, and trade execution.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import yfinance as yf
import joblib

# Import our custom modules
from features.build_features import build_features
from features.label_triple_barrier import triple_barrier_labels, forward_return_labels
from trading.risk_management import RiskManager, RiskParams
from trading.signals import SignalGenerator, signals_to_dataframe
from trading.broker_integration import TradingEngine, setup_simulated_trading, setup_alpaca_paper_trading


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TradingSystemConfig:
    """Configuration for the trading system."""
    
    def __init__(self):
        # Data settings
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META']
        self.data_period = '5y'  # yfinance period
        self.data_interval = '1d'  # daily data
        
        # Model settings
        self.model_path = 'models/trained_model.pkl'
        self.meta_path = 'models/model_meta.json'
        
        # Risk management
        self.risk_params = RiskParams(
            max_risk_per_trade=0.02,  # 2% max risk per trade
            profit_target_multiplier=2.0,  # 2:1 R:R ratio
            max_position_size=0.15,  # Max 15% per position
        )
        
        # Signal generation
        self.signal_config = {
            'ml_threshold_buy': 0.65,
            'ml_threshold_sell': 0.35,
            'min_confidence': 0.7,
            'volume_threshold': 1.2
        }
        
        # Trading settings
        self.account_balance = 100000.0  # Starting balance
        self.use_paper_trading = True  # Set to True for Alpaca paper trading
        self.max_positions = 5  # Maximum concurrent positions
        
        # Paths
        self.data_dir = 'data'
        self.artifacts_dir = 'artifacts'
        self.models_dir = 'models'
        
        # Create directories
        for directory in [self.data_dir, self.artifacts_dir, self.models_dir]:
            os.makedirs(directory, exist_ok=True)


class AlgorithmicTradingSystem:
    """Main algorithmic trading system."""
    
    def __init__(self, config: TradingSystemConfig):
        self.config = config
        self.risk_manager = RiskManager(config.risk_params)
        self.signal_generator = SignalGenerator(**config.signal_config)
        self.trading_engine = None
        self.model = None
        self.model_meta = None
        
        # Data storage
        self.ohlcv_data = None
        self.features_data = None
        self.predictions = None
        self.current_signals = []
        
    def setup_trading_engine(self) -> bool:
        """Setup the trading engine (broker connection)."""
        try:
            if self.config.use_paper_trading:
                self.trading_engine = setup_alpaca_paper_trading()
            else:
                self.trading_engine = setup_simulated_trading(self.config.account_balance)
            
            if self.trading_engine and self.trading_engine.connect():
                logger.info("Successfully connected to trading engine")
                return True
            else:
                logger.warning("Failed to connect to trading engine, using simulation")
                self.trading_engine = setup_simulated_trading(self.config.account_balance)
                return self.trading_engine.connect()
        except Exception as e:
            logger.error(f"Error setting up trading engine: {e}")
            # Fallback to simulation
            self.trading_engine = setup_simulated_trading(self.config.account_balance)
            return self.trading_engine.connect()
    
    def load_model(self) -> bool:
        """Load the trained ML model."""
        try:
            if not os.path.exists(self.config.model_path):
                logger.error(f"Model file not found: {self.config.model_path}")
                return False
                
            if not os.path.exists(self.config.meta_path):
                logger.error(f"Model meta file not found: {self.config.meta_path}")
                return False
            
            self.model = joblib.load(self.config.model_path)
            with open(self.config.meta_path, 'r') as f:
                self.model_meta = json.load(f)
            
            logger.info(f"Loaded model trained on {len(self.model_meta['feature_names'])} features")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def fetch_market_data(self) -> bool:
        """Fetch latest market data."""
        try:
            logger.info(f"Fetching market data for {len(self.config.symbols)} symbols...")
            
            # Download data using yfinance
            data = yf.download(
                self.config.symbols,
                period=self.config.data_period,
                interval=self.config.data_interval,
                group_by="ticker",
                auto_adjust=False,
                progress=False
            )
            
            # Reshape to MultiIndex (timestamp, symbol)
            parts = []
            for symbol in self.config.symbols:
                try:
                    if len(self.config.symbols) == 1:
                        symbol_data = data.copy()
                    else:
                        symbol_data = data[symbol].copy()
                    
                    symbol_data.columns = symbol_data.columns.str.lower()
                    symbol_data = symbol_data.dropna()
                    
                    if len(symbol_data) == 0:
                        logger.warning(f"No data for symbol {symbol}")
                        continue
                    
                    symbol_data['symbol'] = symbol
                    symbol_data = symbol_data.reset_index().set_index(['Date', 'symbol'])
                    symbol_data.index.names = ['timestamp', 'symbol']
                    parts.append(symbol_data)
                    
                except Exception as e:
                    logger.warning(f"Error processing data for {symbol}: {e}")
                    continue
            
            if not parts:
                logger.error("No market data retrieved")
                return False
            
            self.ohlcv_data = pd.concat(parts).sort_index()
            logger.info(f"Retrieved {len(self.ohlcv_data)} rows of market data")
            
            # Save data
            data_path = os.path.join(self.config.data_dir, 'current_market_data.parquet')
            self.ohlcv_data.to_parquet(data_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return False
    
    def generate_features(self) -> bool:
        """Generate features from market data."""
        try:
            if self.ohlcv_data is None:
                logger.error("No market data available for feature generation")
                return False
            
            logger.info("Generating features...")
            self.features_data = build_features(self.ohlcv_data)
            
            # Save features
            features_path = os.path.join(self.config.data_dir, 'current_features.parquet')
            self.features_data.to_parquet(features_path)
            
            logger.info(f"Generated {len(self.features_data.columns)} features for {len(self.features_data)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Error generating features: {e}")
            return False
    
    def generate_predictions(self) -> bool:
        """Generate ML predictions on current data."""
        try:
            if self.model is None or self.features_data is None:
                logger.error("Model or features not available")
                return False
            
            logger.info("Generating ML predictions...")
            
            # Get the latest data for each symbol (most recent timestamp)
            latest_data = self.features_data.groupby(level=1).tail(1)
            
            # Align features with model expectations
            feature_names = self.model_meta['feature_names']
            missing_features = [col for col in feature_names if col not in latest_data.columns]
            
            # Add missing features as NaN
            for col in missing_features:
                latest_data[col] = np.nan
            
            # Select and order features
            X = latest_data[feature_names].copy()
            X = X.replace([np.inf, -np.inf], np.nan)
            
            # Generate predictions
            if self.model_meta['task'] == 'classification':
                pred_proba = self.model.predict_proba(X)
                if pred_proba.shape[1] == 2:
                    # Binary classification - take positive class probability
                    predictions = pd.Series(pred_proba[:, 1], index=X.index, name='proba_pos')
                else:
                    # Multi-class - create DataFrame with all probabilities
                    class_names = [f"proba_{cls}" for cls in self.model_meta['classes']]
                    predictions = pd.DataFrame(pred_proba, index=X.index, columns=class_names)
            else:
                # Regression
                pred_values = self.model.predict(X)
                predictions = pd.Series(pred_values, index=X.index, name='y_pred')
            
            self.predictions = predictions
            
            # Save predictions
            pred_path = os.path.join(self.config.artifacts_dir, 'current_predictions.parquet')
            if isinstance(predictions, pd.Series):
                predictions.to_frame().to_parquet(pred_path)
            else:
                predictions.to_parquet(pred_path)
            
            logger.info(f"Generated predictions for {len(predictions)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return False
    
    def generate_trading_signals(self) -> bool:
        """Generate trading signals from predictions."""
        try:
            if self.predictions is None or self.features_data is None or self.ohlcv_data is None:
                logger.error("Required data not available for signal generation")
                return False
            
            logger.info("Generating trading signals...")
            
            # Get latest data for each symbol
            latest_features = self.features_data.groupby(level=1).tail(1)
            latest_prices = self.ohlcv_data.groupby(level=1).tail(1)
            
            # Generate signals
            self.current_signals = self.signal_generator.batch_generate_signals(
                features_df=latest_features,
                predictions_df=self.predictions.to_frame() if isinstance(self.predictions, pd.Series) else self.predictions,
                prices_df=latest_prices,
                risk_manager=self.risk_manager,
                account_balance=self.config.account_balance
            )
            
            # Save signals
            if self.current_signals:
                signals_df = signals_to_dataframe(self.current_signals)
                signals_path = os.path.join(self.config.artifacts_dir, 'current_signals.parquet')
                signals_df.to_parquet(signals_path)
                
                logger.info(f"Generated {len(self.current_signals)} trading signals")
                
                # Log top signals
                for signal in self.current_signals[:3]:
                    logger.info(f"Signal: {signal.symbol} {signal.signal.name} "
                              f"(confidence: {signal.confidence:.3f}, R:R: {signal.risk_reward_ratio:.2f})")
            else:
                logger.info("No trading signals generated")
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return False
    
    def execute_trades(self) -> bool:
        """Execute the generated trading signals."""
        try:
            if not self.current_signals or not self.trading_engine:
                logger.info("No signals to execute or trading engine not available")
                return False
            
            logger.info("Executing trades...")
            
            # Get current positions to avoid over-concentration
            current_positions = self.trading_engine.broker.get_positions()
            position_symbols = {pos.symbol for pos in current_positions}
            
            executed_count = 0
            max_new_positions = self.config.max_positions - len(position_symbols)
            
            for signal in self.current_signals[:max_new_positions]:
                # Skip if we already have a position in this symbol
                if signal.symbol in position_symbols:
                    logger.info(f"Skipping {signal.symbol} - already have position")
                    continue
                
                # Set current price for simulated broker
                if hasattr(self.trading_engine.broker, 'set_price'):
                    self.trading_engine.broker.set_price(signal.symbol, signal.entry_price)
                
                # Execute the signal
                order_id = self.trading_engine.execute_signal(signal)
                if order_id:
                    executed_count += 1
                    logger.info(f"Executed trade: {signal.symbol} {signal.signal.name} "
                              f"({signal.position_size} shares @ ${signal.entry_price:.2f})")
            
            logger.info(f"Executed {executed_count} trades")
            return True
            
        except Exception as e:
            logger.error(f"Error executing trades: {e}")
            return False
    
    def monitor_positions(self):
        """Monitor existing positions for exit conditions."""
        try:
            if not self.trading_engine:
                return
            
            # Check exit conditions (stop loss / take profit)
            self.trading_engine.check_exit_conditions()
            
            # Update prices for simulated broker
            if hasattr(self.trading_engine.broker, 'set_price') and self.ohlcv_data is not None:
                latest_prices = self.ohlcv_data.groupby(level=1).tail(1)
                for idx, row in latest_prices.iterrows():
                    symbol = idx[1]
                    price = row['close']
                    self.trading_engine.broker.set_price(symbol, price)
            
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status."""
        try:
            if not self.trading_engine:
                return {}
            
            return self.trading_engine.get_portfolio_summary()
            
        except Exception as e:
            logger.error(f"Error getting portfolio status: {e}")
            return {}
    
    def run_trading_cycle(self) -> bool:
        """Run a complete trading cycle."""
        try:
            logger.info("=" * 60)
            logger.info("Starting trading cycle")
            logger.info("=" * 60)
            
            # Step 1: Fetch market data
            if not self.fetch_market_data():
                logger.error("Failed to fetch market data")
                return False
            
            # Step 2: Generate features
            if not self.generate_features():
                logger.error("Failed to generate features")
                return False
            
            # Step 3: Generate predictions
            if not self.generate_predictions():
                logger.error("Failed to generate predictions")
                return False
            
            # Step 4: Generate signals
            if not self.generate_trading_signals():
                logger.error("Failed to generate trading signals")
                return False
            
            # Step 5: Execute trades
            if not self.execute_trades():
                logger.info("No trades executed this cycle")
            
            # Step 6: Monitor existing positions
            self.monitor_positions()
            
            # Step 7: Log portfolio status
            portfolio = self.get_portfolio_status()
            if portfolio:
                account = portfolio.get('account', {})
                logger.info(f"Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
                logger.info(f"Cash: ${account.get('cash', 0):,.2f}")
                logger.info(f"Positions: {len(portfolio.get('positions', []))}")
            
            logger.info("Trading cycle completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            return False
    
    def run_live_trading(self, check_interval_minutes: int = 60):
        """Run live trading with periodic checks."""
        logger.info("Starting live trading system...")
        logger.info(f"Check interval: {check_interval_minutes} minutes")
        
        try:
            while True:
                # Run trading cycle
                success = self.run_trading_cycle()
                
                if not success:
                    logger.warning("Trading cycle failed, continuing...")
                
                # Wait for next cycle
                logger.info(f"Waiting {check_interval_minutes} minutes until next cycle...")
                time.sleep(check_interval_minutes * 60)
                
        except KeyboardInterrupt:
            logger.info("Shutting down trading system...")
        except Exception as e:
            logger.error(f"Fatal error in live trading: {e}")
        finally:
            if self.trading_engine:
                self.trading_engine.disconnect()
            logger.info("Trading system shutdown complete")


def train_initial_model(config: TradingSystemConfig):
    """Train an initial model if none exists."""
    from models.train import main as train_main
    import sys
    
    logger.info("No model found, training initial model...")
    
    # Fetch data for training
    system = AlgorithmicTradingSystem(config)
    if not system.fetch_market_data():
        logger.error("Failed to fetch training data")
        return False
    
    if not system.generate_features():
        logger.error("Failed to generate features for training")
        return False
    
    # Generate labels using triple barrier method
    labels = triple_barrier_labels(
        system.ohlcv_data,
        up_mult=1.0,
        down_mult=1.0,
        max_holding=10
    )
    
    # Save training data
    features_path = os.path.join(config.data_dir, 'training_features.parquet')
    labels_path = os.path.join(config.data_dir, 'training_labels.parquet') 
    
    system.features_data.to_parquet(features_path)
    labels.to_parquet(labels_path)
    
    # Train model using command line interface
    train_args = [
        '--features_path', features_path,
        '--labels_path', labels_path,
        '--task', 'classification',
        '--label_col', 'label',
        '--drop_class_zero',  # Drop neutral labels
        '--test_frac', '0.2',
        '--model_out', config.model_path,
        '--meta_out', config.meta_path,
        '--eval_test_now'
    ]
    
    # Save original sys.argv
    original_argv = sys.argv
    try:
        sys.argv = ['train.py'] + train_args
        train_main()
        logger.info("Model training completed")
        return True
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return False
    finally:
        sys.argv = original_argv


def main():
    """Main entry point."""
    # Setup configuration
    config = TradingSystemConfig()
    
    # Create trading system
    system = AlgorithmicTradingSystem(config)
    
    # Setup trading engine
    if not system.setup_trading_engine():
        logger.error("Failed to setup trading engine")
        return
    
    # Load or train model
    if not system.load_model():
        logger.warning("Model not found, training new model...")
        if not train_initial_model(config):
            logger.error("Failed to train initial model")
            return
        if not system.load_model():
            logger.error("Failed to load newly trained model")
            return
    
    # Choose mode
    mode = input("Choose mode:\n1. Single run\n2. Live trading\n3. Backtest simulation\nEnter choice (1-3): ").strip()
    
    if mode == '1':
        # Single trading cycle
        logger.info("Running single trading cycle...")
        system.run_trading_cycle()
        
        # Show final portfolio
        portfolio = system.get_portfolio_status()
        print("\n" + "="*50)
        print("FINAL PORTFOLIO STATUS")
        print("="*50)
        if portfolio:
            account = portfolio.get('account', {})
            print(f"Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
            print(f"Cash: ${account.get('cash', 0):,.2f}")
            print(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
            
            positions = portfolio.get('positions', [])
            if positions:
                print(f"\nPositions ({len(positions)}):")
                for pos in positions:
                    pnl_pct = (pos['unrealized_pnl'] / pos['market_value']) * 100 if pos['market_value'] != 0 else 0
                    print(f"  {pos['symbol']}: {pos['quantity']} shares @ ${pos['avg_price']:.2f} "
                          f"(${pos['market_value']:,.2f}, PnL: {pnl_pct:+.1f}%)")
    
    elif mode == '2':
        # Live trading
        interval = input("Enter check interval in minutes (default 60): ").strip()
        interval = int(interval) if interval.isdigit() else 60
        system.run_live_trading(check_interval_minutes=interval)
    
    elif mode == '3':
        # Backtest simulation with historical data
        logger.info("Running backtest simulation...")
        
        # Use simulated broker with current data
        system.trading_engine = setup_simulated_trading(config.account_balance)
        system.trading_engine.connect()
        
        # Run single cycle to simulate backtest
        system.run_trading_cycle()
        
        # Show results
        portfolio = system.get_portfolio_status()
        if portfolio:
            account = portfolio.get('account', {})
            initial_value = config.account_balance
            final_value = account.get('portfolio_value', initial_value)
            returns = ((final_value - initial_value) / initial_value) * 100
            
            print("\n" + "="*50)
            print("BACKTEST RESULTS")
            print("="*50)
            print(f"Initial Value: ${initial_value:,.2f}")
            print(f"Final Value: ${final_value:,.2f}")
            print(f"Total Return: {returns:+.2f}%")
            print(f"Number of Positions: {len(portfolio.get('positions', []))}")
    
    else:
        print("Invalid choice")
    
    # Cleanup
    if system.trading_engine:
        system.trading_engine.disconnect()


if __name__ == "__main__":
    main()