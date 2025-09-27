# trading/signals.py
"""
Trading signal generation using ML predictions and technical analysis.
Combines model predictions with additional filters and risk management.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class SignalType(Enum):
    """Signal types."""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class TradingSignal:
    """Trading signal with all relevant information."""
    symbol: str
    timestamp: pd.Timestamp
    signal: SignalType
    confidence: float  # 0-1
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: int
    risk_reward_ratio: float
    model_prediction: float
    technical_score: float
    volume_score: float
    volatility_score: float
    reasoning: str


class SignalGenerator:
    """Generate trading signals from ML predictions and technical analysis."""
    
    def __init__(
        self,
        ml_threshold_buy: float = 0.6,
        ml_threshold_sell: float = 0.4,
        volume_threshold: float = 1.2,  # Relative to average volume
        volatility_max: float = 0.05,   # Max daily volatility
        min_confidence: float = 0.7,    # Minimum confidence for signal
    ):
        self.ml_threshold_buy = ml_threshold_buy
        self.ml_threshold_sell = ml_threshold_sell  
        self.volume_threshold = volume_threshold
        self.volatility_max = volatility_max
        self.min_confidence = min_confidence
    
    def calculate_technical_score(self, features: pd.Series) -> float:
        """Calculate technical analysis score from features."""
        score = 0.0
        weight_sum = 0.0
        
        # RSI scoring (oversold/overbought)
        if 'rsi_14' in features:
            rsi = features['rsi_14']
            if not pd.isna(rsi):
                if rsi < 30:  # Oversold - bullish
                    score += 0.3 * (30 - rsi) / 30
                elif rsi > 70:  # Overbought - bearish  
                    score -= 0.3 * (rsi - 70) / 30
                weight_sum += 0.3
        
        # MACD scoring
        if 'macd_12_26' in features and 'macd_signal_9' in features:
            macd = features['macd_12_26']
            signal = features['macd_signal_9']
            if not pd.isna(macd) and not pd.isna(signal):
                macd_diff = macd - signal
                # Normalize by recent price for comparison
                if 'close' in features:
                    macd_diff_norm = macd_diff / features['close']
                    score += 0.2 * np.tanh(macd_diff_norm * 1000)  # Scale and bound
                weight_sum += 0.2
        
        # Moving average crossovers (if we add them)
        # Add momentum indicators
        if 'ret_5' in features:
            ret_5 = features['ret_5']
            if not pd.isna(ret_5):
                score += 0.2 * np.tanh(ret_5 * 10)  # Recent momentum
                weight_sum += 0.2
        
        # Volume-weighted average price distance
        if 'dist_vwap_20' in features:
            vwap_dist = features['dist_vwap_20']
            if not pd.isna(vwap_dist):
                # Positive distance = above VWAP = bullish
                score += 0.1 * np.tanh(vwap_dist * 5)
                weight_sum += 0.1
        
        # Volatility (lower is generally better for entry)
        if 'vol_20' in features:
            vol = features['vol_20']
            if not pd.isna(vol):
                # Penalize high volatility
                if vol > self.volatility_max:
                    score -= 0.2 * min((vol - self.volatility_max) / self.volatility_max, 1.0)
                weight_sum += 0.2
        
        # Normalize score
        if weight_sum > 0:
            score = score / weight_sum
            
        return np.clip(score, -1.0, 1.0)
    
    def calculate_volume_score(self, current_volume: float, avg_volume: float) -> float:
        """Calculate volume confirmation score."""
        if avg_volume <= 0:
            return 0.0
        
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio >= self.volume_threshold:
            # High volume confirms signal
            return min(1.0, (volume_ratio - 1.0) / 2.0)
        else:
            # Low volume weakens signal
            return -0.5 * (1.0 - volume_ratio)
    
    def calculate_volatility_score(self, volatility: float) -> float:
        """Calculate volatility score (lower vol = higher score for entries)."""
        if pd.isna(volatility):
            return 0.0
        
        if volatility <= 0.02:  # Low volatility - good for entry
            return 1.0
        elif volatility <= self.volatility_max:  # Moderate volatility
            return 1.0 - (volatility - 0.02) / (self.volatility_max - 0.02)
        else:  # High volatility - avoid
            return -1.0
    
    def generate_signal(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        features: pd.Series,
        ml_prediction: float,
        current_price: float,
        current_volume: float,
        avg_volume: float,
        atr_value: float,
        risk_manager,
        account_balance: float
    ) -> Optional[TradingSignal]:
        """Generate a complete trading signal."""
        
        # Calculate component scores
        technical_score = self.calculate_technical_score(features)
        volume_score = self.calculate_volume_score(current_volume, avg_volume)
        volatility_score = self.calculate_volatility_score(features.get('vol_20', np.nan))
        
        # Determine base signal direction from ML prediction
        if ml_prediction >= self.ml_threshold_buy:
            direction = "long"
            base_signal = SignalType.BUY
        elif ml_prediction <= self.ml_threshold_sell:
            direction = "short" 
            base_signal = SignalType.SELL
        else:
            base_signal = SignalType.HOLD
            direction = "hold"
        
        # Calculate levels using risk manager
        if direction != "hold":
            levels = risk_manager.calculate_trade_levels(
                symbol=symbol,
                current_price=current_price,
                direction=direction,
                atr_value=atr_value,
                method="atr"
            )
            
            position_info = risk_manager.size_position(
                account_balance=account_balance,
                entry_price=current_price,
                stop_loss=levels['stop_loss']
            )
            
            # Validate trade
            is_valid, validation_msg = risk_manager.validate_trade(levels, position_info)
            if not is_valid:
                return None
        else:
            return None  # No signal for holds
        
        # Calculate combined confidence
        ml_confidence = abs(ml_prediction - 0.5) * 2  # Convert to 0-1 scale
        technical_confidence = abs(technical_score)
        volume_confidence = max(0, volume_score)
        volatility_confidence = max(0, volatility_score)
        
        # Weighted combination
        weights = [0.4, 0.3, 0.15, 0.15]  # ML, Technical, Volume, Volatility
        overall_confidence = (
            weights[0] * ml_confidence +
            weights[1] * technical_confidence + 
            weights[2] * volume_confidence +
            weights[3] * volatility_confidence
        )
        
        # Filter by minimum confidence
        if overall_confidence < self.min_confidence:
            return None
        
        # Adjust signal strength based on confidence and scores
        if overall_confidence > 0.85 and technical_score > 0.5:
            if base_signal == SignalType.BUY:
                final_signal = SignalType.STRONG_BUY
            elif base_signal == SignalType.SELL:
                final_signal = SignalType.STRONG_SELL
            else:
                final_signal = base_signal
        else:
            final_signal = base_signal
        
        # Create reasoning
        reasoning_parts = []
        reasoning_parts.append(f"ML prediction: {ml_prediction:.3f}")
        reasoning_parts.append(f"Technical score: {technical_score:.3f}")
        reasoning_parts.append(f"Volume ratio: {current_volume/avg_volume:.2f}")
        reasoning_parts.append(f"Volatility: {features.get('vol_20', 0):.3f}")
        reasoning = " | ".join(reasoning_parts)
        
        return TradingSignal(
            symbol=symbol,
            timestamp=timestamp,
            signal=final_signal,
            confidence=overall_confidence,
            entry_price=current_price,
            stop_loss=levels['stop_loss'],
            take_profit=levels['take_profit'],
            position_size=position_info['shares'],
            risk_reward_ratio=levels['reward_amount'] / levels['risk_amount'],
            model_prediction=ml_prediction,
            technical_score=technical_score,
            volume_score=volume_score,
            volatility_score=volatility_score,
            reasoning=reasoning
        )
    
    def batch_generate_signals(
        self,
        features_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        prices_df: pd.DataFrame,  # OHLCV data
        risk_manager,
        account_balance: float,
        volume_lookback: int = 20
    ) -> List[TradingSignal]:
        """Generate signals for multiple stocks/timestamps."""
        
        signals = []
        
        # Ensure all dataframes have the same MultiIndex structure
        common_index = features_df.index.intersection(predictions_df.index).intersection(prices_df.index)
        
        for idx in common_index:
            try:
                symbol = idx[1]  # Assuming (timestamp, symbol) MultiIndex
                timestamp = pd.Timestamp(idx[0])
                
                features = features_df.loc[idx]
                
                # Get ML prediction (assuming binary classification with proba_pos)
                if 'proba_pos' in predictions_df.columns:
                    ml_pred = predictions_df.loc[idx, 'proba_pos']
                elif 'y_pred' in predictions_df.columns:
                    ml_pred = predictions_df.loc[idx, 'y_pred'] 
                else:
                    ml_pred = predictions_df.loc[idx].iloc[0]  # First column
                
                # Get current market data
                current_data = prices_df.loc[idx]
                current_price = current_data['close']
                current_volume = current_data['volume']
                
                # Calculate average volume
                symbol_data = prices_df.xs(symbol, level=1)
                recent_volumes = symbol_data['volume'].rolling(volume_lookback).mean()
                avg_volume = recent_volumes.loc[timestamp] if timestamp in recent_volumes.index else current_volume
                
                # Get ATR
                atr_value = features.get('atr_14', current_price * 0.02)  # Fallback to 2% of price
                
                # Generate signal
                signal = self.generate_signal(
                    symbol=symbol,
                    timestamp=timestamp,
                    features=features,
                    ml_prediction=ml_pred,
                    current_price=current_price,
                    current_volume=current_volume,
                    avg_volume=avg_volume,
                    atr_value=atr_value,
                    risk_manager=risk_manager,
                    account_balance=account_balance
                )
                
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                print(f"Error processing {idx}: {e}")
                continue
        
        # Sort by confidence (highest first)
        signals.sort(key=lambda x: x.confidence, reverse=True)
        return signals


def signals_to_dataframe(signals: List[TradingSignal]) -> pd.DataFrame:
    """Convert list of signals to DataFrame for analysis."""
    if not signals:
        return pd.DataFrame()
    
    data = []
    for signal in signals:
        data.append({
            'symbol': signal.symbol,
            'timestamp': signal.timestamp,
            'signal': signal.signal.name,
            'confidence': signal.confidence,
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'position_size': signal.position_size,
            'risk_reward_ratio': signal.risk_reward_ratio,
            'model_prediction': signal.model_prediction,
            'technical_score': signal.technical_score,
            'volume_score': signal.volume_score,
            'volatility_score': signal.volatility_score,
            'reasoning': signal.reasoning
        })
    
    return pd.DataFrame(data)


# Example usage and testing
if __name__ == "__main__":
    from trading.risk_management import RiskManager, RiskParams
    
    # Create sample data
    features = pd.Series({
        'rsi_14': 25,  # Oversold
        'macd_12_26': 0.5,
        'macd_signal_9': 0.2,
        'ret_5': 0.02,
        'vol_20': 0.03,
        'dist_vwap_20': 0.01
    })
    
    # Setup components
    risk_params = RiskParams()
    risk_manager = RiskManager(risk_params)
    signal_gen = SignalGenerator()
    
    # Generate signal
    signal = signal_gen.generate_signal(
        symbol="AAPL",
        timestamp=pd.Timestamp.now(),
        features=features,
        ml_prediction=0.75,  # Strong buy prediction
        current_price=150.0,
        current_volume=1000000,
        avg_volume=800000,
        atr_value=3.5,
        risk_manager=risk_manager,
        account_balance=100000
    )
    
    if signal:
        print(f"Generated Signal: {signal.signal.name}")
        print(f"Confidence: {signal.confidence:.3f}")
        print(f"Entry: ${signal.entry_price:.2f}")
        print(f"Stop Loss: ${signal.stop_loss:.2f}")  
        print(f"Take Profit: ${signal.take_profit:.2f}")
        print(f"Position Size: {signal.position_size} shares")
        print(f"Risk:Reward: {signal.risk_reward_ratio:.2f}")
        print(f"Reasoning: {signal.reasoning}")
    else:
        print("No signal generated")


# Additional utility functions for signal analysis
def analyze_signal_performance(signals_df: pd.DataFrame, actual_returns: pd.Series) -> pd.DataFrame:
    """Analyze the performance of generated signals against actual returns."""
    if signals_df.empty:
        return pd.DataFrame()
    
    # This would require actual forward returns to evaluate
    # For now, just return signal statistics
    return signals_df.groupby('signal').agg({
        'confidence': ['mean', 'std', 'count'],
        'risk_reward_ratio': ['mean', 'std'],
        'model_prediction': ['mean', 'std'],
        'technical_score': ['mean', 'std']
    })


def filter_signals_by_sector(signals: List[TradingSignal], sector_map: Dict[str, str], 
                           allowed_sectors: List[str]) -> List[TradingSignal]:
    """Filter signals by sector to ensure diversification."""
    return [s for s in signals if sector_map.get(s.symbol, 'Unknown') in allowed_sectors]


def portfolio_correlation_filter(signals: List[TradingSignal], correlation_matrix: pd.DataFrame, 
                                max_correlation: float = 0.7) -> List[TradingSignal]:
    """Filter signals to avoid highly correlated positions."""
    if not signals or correlation_matrix.empty:
        return signals
    
    selected = []
    selected_symbols = []
    
    for signal in signals:
        if not selected_symbols:
            selected.append(signal)
            selected_symbols.append(signal.symbol)
        else:
            # Check correlation with already selected symbols
            correlations = [abs(correlation_matrix.loc[signal.symbol, sym]) 
                          for sym in selected_symbols 
                          if signal.symbol in correlation_matrix.index and sym in correlation_matrix.columns]
            
            if not correlations or max(correlations) < max_correlation:
                selected.append(signal)
                selected_symbols.append(signal.symbol)
    
    return selected