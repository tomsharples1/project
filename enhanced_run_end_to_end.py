# enhanced_run_end_to_end.py
"""
Enhanced end-to-end demonstration of the complete algorithmic trading system.
This script demonstrates the full pipeline from data fetching to trade execution.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from features.build_features import build_features
from features.label_triple_barrier import triple_barrier_labels, forward_return_labels
from trading.risk_management import RiskManager, RiskParams
from trading.signals import SignalGenerator, signals_to_dataframe
from trading.broker_integration import setup_simulated_trading

# Create directories
os.makedirs("data", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("="*60)
print("ALGORITHMIC TRADING SYSTEM - COMPLETE DEMONSTRATION")
print("="*60)

# ---------- STEP 1: Download Market Data ----------
print("\n1. DOWNLOADING MARKET DATA")
print("-" * 30)

# Extended list of stocks for better diversification
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
print(f"Fetching data for: {', '.join(tickers)}")

# Download more data for better model training
data = yf.download(
    tickers, 
    period='5y',
    interval="1d", 
    group_by="ticker", 
    auto_adjust=False,
    progress=True
)

# Reshape to MultiIndex (timestamp, symbol) with lowercase OHLCV
parts = []
for sym in tickers:
    try:
        if len(tickers) == 1:
            tmp = data.copy()
        else:
            tmp = data[sym].copy()
        
        tmp.columns = tmp.columns.str.lower()
        tmp = tmp.dropna()  # Remove any missing data
        
        if len(tmp) == 0:
            print(f"Warning: No data for {sym}")
            continue
        
        tmp["symbol"] = sym
        tmp = tmp.reset_index().set_index(["Date", "symbol"]).rename_axis(["timestamp", "symbol"])
        parts.append(tmp)
        print(f"  {sym}: {len(tmp)} days of data")
    except Exception as e:
        print(f"Error processing {sym}: {e}")
        continue

if not parts:
    raise Exception("No market data retrieved!")

ohlcv = pd.concat(parts).swaplevel(0, 1).sort_index()
print(f"\nTotal dataset: {len(ohlcv)} rows across {len(tickers)} symbols")

# Save raw data
ohlcv.to_parquet("data/raw_market_data.parquet", engine='pyarrow')

# ---------- STEP 2: Feature Engineering ----------
print("\n2. FEATURE ENGINEERING")
print("-" * 30)

print("Generating technical indicators...")
features = build_features(
    ohlcv,
    return_windows=(1, 5, 10, 20),  # Multiple return periods
    vol_windows=(5, 20, 60),        # Volatility windows
    rsi_window=14,
    macd_params=(12, 26, 9),
    vwap_window=20,
    atr_window=14,
    include_cross_sectionals=True   # Include cross-sectional features
)

print(f"Generated features: {list(features.columns)}")
print(f"Feature matrix shape: {features.shape}")

# Check for data quality
print(f"Missing data percentage: {(features.isnull().sum().sum() / features.size) * 100:.1f}%")

features.to_parquet("data/features.parquet", engine='pyarrow')

# ---------- STEP 3: Label Generation ----------
print("\n3. LABEL GENERATION")
print("-" * 30)

print("Generating classification labels (Triple Barrier)...")
# More conservative barrier settings for better signals
labels_cls = triple_barrier_labels(
    ohlcv, 
    up_mult=1.2,      # Slightly wider barriers
    down_mult=1.2, 
    vol_lookback=50, 
    max_holding=15,   # Longer holding period
    min_ret=0.005     # Minimum return threshold (0.5%)
)

print(f"Classification labels shape: {labels_cls.shape}")
print("Label distribution:")
print(labels_cls['label'].value_counts().sort_index())

# Calculate success metrics
total_labels = len(labels_cls)
profitable = len(labels_cls[labels_cls['label'] == 1])
losses = len(labels_cls[labels_cls['label'] == -1])
neutral = len(labels_cls[labels_cls['label'] == 0])

print(f"Win rate: {profitable/total_labels*100:.1f}%")
print(f"Loss rate: {losses/total_labels*100:.1f}%")
print(f"Neutral rate: {neutral/total_labels*100:.1f}%")

labels_cls.to_parquet("data/labels_cls.parquet", engine='pyarrow')

print("\nGenerating regression labels (Forward Returns)...")
y_reg = forward_return_labels(ohlcv, horizon=10, use_log=False)  # 10-day forward returns
y_reg_df = y_reg.to_frame("y")
print(f"Regression labels shape: {y_reg_df.shape}")
print(f"Mean forward return: {y_reg.mean():.3f}")
print(f"Std forward return: {y_reg.std():.3f}")

y_reg_df.to_parquet("data/labels_reg.parquet", engine='pyarrow')

# ---------- STEP 4: Train ML Model ----------
print("\n4. MACHINE LEARNING MODEL TRAINING")
print("-" * 30)

# We'll use the modular train.py script
import subprocess
import sys

# Train classification model
print("Training classification model...")
train_cmd = [
    sys.executable, '-m', 'models.train',
    '--features_path', 'data/features.parquet',
    '--labels_path', 'data/labels_cls.parquet',
    '--task', 'classification',
    '--label_col', 'label',
    '--drop_class_zero',  # Focus on clear signals
    '--test_frac', '0.2',
    '--embargo', '5',  # 5 day embargo to prevent leakage
    '--n_splits', '5',  # 5-fold CV
    '--learning_rate', '0.05',
    '--n_estimators', '500',
    '--model_out', 'models/trained_model.pkl',
    '--meta_out', 'models/model_meta.json',
    '--eval_test_now'
]

try:
    result = subprocess.run(train_cmd, capture_output=True, text=True, check=True)
    print("✅ Model training completed successfully")
    print("Training results:")
    print(result.stdout[-500:])  # Show last 500 chars of output
except subprocess.CalledProcessError as e:
    print(f"❌ Model training failed: {e}")
    print("Error output:", e.stderr)

# ---------- STEP 5: Generate Predictions ----------
print("\n5. GENERATING PREDICTIONS")
print("-" * 30)

# Get recent data for prediction (last 30 days)
recent_cutoff = ohlcv.index.get_level_values(0).max() - pd.Timedelta(days=30)
recent_features = features[features.index.get_level_values(0) >= recent_cutoff]

print(f"Generating predictions for {len(recent_features)} recent observations...")

# Generate predictions
predict_cmd = [
    sys.executable, '-m', 'models.infer',
    '--features_path', 'data/features.parquet',
    '--model_path', 'models/trained_model.pkl',
    '--meta_path', 'models/model_meta.json',
    '--pred_out', 'artifacts/predictions.parquet',
    '--start_date', str(recent_cutoff.date())
]

try:
    result = subprocess.run(predict_cmd, capture_output=True, text=True, check=True)
    print("✅ Predictions generated successfully")
    
    # Load and display predictions
    predictions = pd.read_parquet('artifacts/predictions.parquet')
    print(f"Predictions shape: {predictions.shape}")
    print("\nSample predictions:")
    print(predictions.head())
    
    if 'proba_pos' in predictions.columns:
        print(f"\nPrediction statistics:")
        print(f"Mean positive probability: {predictions['proba_pos'].mean():.3f}")
        print(f"High confidence predictions (>0.7): {len(predictions[predictions['proba_pos'] > 0.7])}")
        print(f"Low confidence predictions (<0.3): {len(predictions[predictions['proba_pos'] < 0.3])}")
        
except subprocess.CalledProcessError as e:
    print(f"❌ Prediction generation failed: {e}")
    print("Error output:", e.stderr)

# ---------- STEP 6: Risk Management & Signal Generation ----------
print("\n6. RISK MANAGEMENT & SIGNAL GENERATION")
print("-" * 30)

# Setup risk management
risk_params = RiskParams(
    max_risk_per_trade=0.015,  # 1.5% max risk per trade
    profit_target_multiplier=2.5,  # 2.5:1 R:R ratio
    max_position_size=0.12,  # Max 12% per position
)

risk_manager = RiskManager(risk_params)

# Setup signal generator with conservative settings
signal_generator = SignalGenerator(
    ml_threshold_buy=0.7,    # Conservative buy threshold
    ml_threshold_sell=0.3,   # Conservative sell threshold
    volume_threshold=1.3,    # Require 30% above average volume
    volatility_max=0.04,     # Max 4% daily volatility
    min_confidence=0.75,     # High confidence requirement
)

print("Risk Management Parameters:")
print(f"  Max risk per trade: {risk_params.max_risk_per_trade*100:.1f}%")
print(f"  Profit target multiplier: {risk_params.profit_target_multiplier}x")
print(f"  Max position size: {risk_params.max_position_size*100:.1f}%")

# Generate signals for the most recent data
try:
    predictions = pd.read_parquet('artifacts/predictions.parquet')
    
    # Get the latest data for each symbol
    latest_features = recent_features.groupby(level=1).tail(1)
    latest_prices = ohlcv[ohlcv.index.get_level_values(0) >= recent_cutoff].groupby(level=1).tail(1)
    
    # Align predictions with features and prices
    common_index = latest_features.index.intersection(predictions.index).intersection(latest_prices.index)
    
    if len(common_index) == 0:
        print("❌ No common data for signal generation")
    else:
        print(f"Generating signals for {len(common_index)} symbols...")
        
        signals = signal_generator.batch_generate_signals(
            features_df=latest_features.loc[common_index],
            predictions_df=predictions.loc[common_index],
            prices_df=latest_prices.loc[common_index],
            risk_manager=risk_manager,
            account_balance=100000.0  # $100k account
        )
        
        print(f"✅ Generated {len(signals)} trading signals")
        
        if signals:
            # Convert to DataFrame for analysis
            signals_df = signals_to_dataframe(signals)
            signals_df.to_parquet('artifacts/trading_signals.parquet')
            
            print("\n📊 TRADING SIGNALS SUMMARY")
            print("-" * 40)
            
            for i, signal in enumerate(signals[:5]):  # Show top 5 signals
                print(f"\n{i+1}. {signal.symbol} - {signal.signal.name}")
                print(f"   Confidence: {signal.confidence:.1%}")
                print(f"   Entry Price: ${signal.entry_price:.2f}")
                print(f"   Stop Loss: ${signal.stop_loss:.2f} ({((signal.stop_loss/signal.entry_price-1)*100):+.1f}%)")
                print(f"   Take Profit: ${signal.take_profit:.2f} ({((signal.take_profit/signal.entry_price-1)*100):+.1f}%)")
                print(f"   Position Size: {signal.position_size} shares (${signal.position_size * signal.entry_price:,.0f})")
                print(f"   Risk:Reward: {signal.risk_reward_ratio:.1f}:1")
                print(f"   ML Score: {signal.model_prediction:.3f}")
                
        else:
            print("⚠️  No signals met the strict criteria")
            print("   Try lowering thresholds or check market conditions")
            
except Exception as e:
    print(f"❌ Signal generation failed: {e}")
    signals = []

# ---------- STEP 7: Trading Simulation ----------
print("\n7. TRADING SIMULATION")
print("-" * 30)

# Setup simulated broker
trading_engine = setup_simulated_trading(initial_cash=100000.0)

if trading_engine.connect():
    print("✅ Connected to simulated trading engine")
    
    # Set current prices for simulation
    for idx, row in latest_prices.iterrows():
        symbol = idx[1]
        price = row['close']
        trading_engine.broker.set_price(symbol, price)
    
    # Execute signals
    if signals:
        print(f"\nExecuting {len(signals)} trading signals...")
        executed_trades = 0
        
        for signal in signals[:3]:  # Execute top 3 signals
            order_id = trading_engine.execute_signal(signal)
            if order_id:
                executed_trades += 1
                print(f"  ✅ Executed: {signal.symbol} {signal.signal.name} "
                      f"({signal.position_size} shares @ ${signal.entry_price:.2f})")
            else:
                print(f"  ❌ Failed to execute: {signal.symbol}")
        
        print(f"\n📈 Successfully executed {executed_trades} trades")
        
        # Simulate some price movements and check exits
        print("\nSimulating market movements...")
        
        for symbol in [s.symbol for s in signals[:executed_trades]]:
            try:
                current_price = latest_prices.xs(symbol, level=1)['close'].iloc[0]
                # Simulate random price movement (-3% to +3%)
                price_change = np.random.uniform(-0.03, 0.03)
                new_price = current_price * (1 + price_change)
                trading_engine.broker.set_price(symbol, new_price)
                print(f"  {symbol}: ${current_price:.2f} → ${new_price:.2f} ({price_change:+.1%})")
            except Exception as e:
                print(f"  Error updating {symbol}: {e}")
        
        # Check for exit conditions
        trading_engine.check_exit_conditions()
        
    else:
        print("⚠️  No signals to execute")
    
    # Portfolio summary
    portfolio = trading_engine.get_portfolio_summary()
    account = portfolio['account']
    positions = portfolio['positions']
    
    print("\n💼 FINAL PORTFOLIO STATUS")
    print("-" * 30)
    print(f"Account Value: ${account['portfolio_value']:,.2f}")
    print(f"Cash: ${account['cash']:,.2f}")
    print(f"Positions: {len(positions)}")
    
    if positions:
        total_pnl = 0
        print("\nActive Positions:")
        for pos in positions:
            pnl = pos['unrealized_pnl']
            pnl_pct = (pnl / pos['market_value']) * 100 if pos['market_value'] != 0 else 0
            total_pnl += pnl
            print(f"  {pos['symbol']}: {pos['quantity']} shares @ ${pos['avg_price']:.2f}")
            print(f"    Market Value: ${pos['market_value']:,.2f}")
            print(f"    P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%)")
        
        print(f"\nTotal Unrealized P&L: ${total_pnl:+,.2f}")
        roi = ((account['portfolio_value'] - 100000) / 100000) * 100
        print(f"Total Return: {roi:+.2f}%")
    
    trading_engine.disconnect()

else:
    print("❌ Failed to connect to trading engine")

# ---------- STEP 8: Performance Analysis ----------
print("\n8. PERFORMANCE ANALYSIS")
print("-" * 30)

# Evaluate model performance if we have test labels
try:
    eval_cmd = [
        sys.executable, '-m', 'models.metrics',
        '--y_true_path', 'data/labels_cls.parquet',
        '--y_pred_path', 'artifacts/predictions.parquet',
        '--mode', 'classification',
        '--report_out', 'artifacts/model_metrics.json',
        '--decile'
    ]
    
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Model evaluation completed")
        
        # Load and display metrics
        import json
        with open('artifacts/model_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        print("📊 Model Performance Metrics:")
        for key, value in metrics.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        print(f"    {k}: {v:.3f}")
                    else:
                        print(f"    {k}: {v}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    else:
        print("⚠️  Model evaluation had issues")
        print(result.stderr)

except Exception as e:
    print(f"⚠️  Could not complete model evaluation: {e}")

# ---------- SUMMARY ----------
print("\n" + "="*60)
print("🎯 SYSTEM DEMONSTRATION COMPLETE")
print("="*60)

print("\n📁 Generated Files:")
files_info = [
    ("data/raw_market_data.parquet", "Raw OHLCV market data"),
    ("data/features.parquet", "Technical indicator features"),
    ("data/labels_cls.parquet", "Classification labels (triple barrier)"),
    ("data/labels_reg.parquet", "Regression labels (forward returns)"),
    ("models/trained_model.pkl", "Trained LightGBM model"),
    ("models/model_meta.json", "Model metadata and configuration"),
    ("artifacts/predictions.parquet", "ML predictions on recent data"),
    ("artifacts/trading_signals.parquet", "Generated trading signals"),
    ("artifacts/model_metrics.json", "Model performance metrics"),
]

for filepath, description in files_info:
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / 1024  # KB
        print(f"  ✅ {filepath} ({size:.1f} KB) - {description}")
    else:
        print(f"  ❌ {filepath} - {description}")

print("\n🚀 Next Steps:")
print("1. Run 'python main_trading_system.py' for the full trading system")
print("2. Explore the generated signals in artifacts/trading_signals.parquet")
print("3. Analyze model performance in artifacts/model_metrics.json")
print("4. Set up Alpaca Paper Trading for real broker integration")
print("5. Customize risk parameters and signal thresholds")

print("\n⚠️  Remember:")
print("- This is for educational purposes only")
print("- Always start with paper trading")
print("- Never risk money you can't afford to lose")
print("- Thoroughly backtest before live trading")

print("\n" + "="*60)