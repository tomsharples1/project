# features/advanced_labeling.py
"""
Advanced labeling strategies for algorithmic trading models.
Focuses on predictions that actually matter for trading profitability.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def cross_sectional_ranking_labels(
    ohlcv: pd.DataFrame,
    forward_periods: List[int] = [5, 10, 20],
    ranking_method: str = 'quantiles',
    n_quantiles: int = 5,
    min_stocks_per_period: int = 10
) -> pd.DataFrame:
    """
    Create cross-sectional ranking labels - predict which stocks will outperform.
    This is what many successful quant funds actually predict.
    
    Returns labels like: TOP_QUINTILE (1), SECOND_QUINTILE (0.75), etc.
    Much more actionable than simple binary up/down predictions.
    """
    ohlcv = ohlcv.copy()
    if not isinstance(ohlcv.index, pd.MultiIndex):
        ohlcv = ohlcv.set_index('symbol', append=True).swaplevel().sort_index()
    
    results = []
    
    for period in forward_periods:
        print(f"Computing cross-sectional rankings for {period}-day forward returns...")
        
        # Calculate forward returns for each stock
        forward_rets = []
        for symbol in ohlcv.index.get_level_values(1).unique():
            symbol_data = ohlcv.xs(symbol, level=1)
            fwd_ret = (symbol_data['close'].shift(-period) / symbol_data['close'] - 1)
            fwd_ret.name = f'fwd_ret_{period}d'
            forward_rets.append(fwd_ret.to_frame().assign(symbol=symbol))
        
        fwd_ret_df = pd.concat(forward_rets).set_index('symbol', append=True).swaplevel()
        
        # Cross-sectional ranking by date
        def rank_by_date(group):
            if len(group) < min_stocks_per_period:
                return pd.Series(index=group.index, dtype=float)
            
            if ranking_method == 'quantiles':
                # Quintile ranking (1 = top 20%, 0 = bottom 20%)
                ranks = pd.qcut(
                    group[f'fwd_ret_{period}d'].rank(method='first'), 
                    n_quantiles, 
                    labels=False
                )
                # Normalize to 0-1 scale
                normalized_ranks = ranks / (n_quantiles - 1)
                return pd.Series(normalized_ranks, index=group.index)
            
            elif ranking_method == 'z_score':
                # Z-score ranking (standardized cross-sectional)
                z_scores = stats.zscore(group[f'fwd_ret_{period}d'].dropna())
                # Convert to 0-1 scale using sigmoid
                normalized = 1 / (1 + np.exp(-z_scores))
                return pd.Series(normalized, index=group.dropna().index)
        
        rankings = fwd_ret_df.groupby(level=0).apply(rank_by_date)
        rankings.name = f'rank_{period}d'
        
        # Create categorical labels
        if ranking_method == 'quantiles':
            def rank_to_category(rank):
                if pd.isna(rank):
                    return 'UNKNOWN'
                elif rank >= 0.8:
                    return 'TOP_QUINTILE'     # Top 20%
                elif rank >= 0.6:
                    return 'SECOND_QUINTILE'  # 60-80%
                elif rank >= 0.4:
                    return 'MIDDLE_QUINTILE'  # 40-60%
                elif rank >= 0.2:
                    return 'FOURTH_QUINTILE'  # 20-40%
                else:
                    return 'BOTTOM_QUINTILE'  # Bottom 20%
            
            categorical = rankings.map(rank_to_category)
            categorical.name = f'category_{period}d'
        
        result_df = pd.DataFrame({
            f'fwd_ret_{period}d': fwd_ret_df[f'fwd_ret_{period}d'],
            f'rank_{period}d': rankings,
            f'category_{period}d': categorical
        })
        
        results.append(result_df)
    
    # Combine all periods
    final_result = pd.concat(results, axis=1)
    return final_result.dropna(how='all')


def volatility_regime_labels(
    ohlcv: pd.DataFrame,
    lookback_vol: int = 60,
    lookback_trend: int = 20,
    forward_horizon: int = 10
) -> pd.DataFrame:
    """
    Predict market regime changes - very valuable for risk management.
    Labels: LOW_VOL_BULL, HIGH_VOL_BEAR, etc.
    """
    if not isinstance(ohlcv.index, pd.MultiIndex):
        ohlcv = ohlcv.set_index('symbol', append=True).swaplevel().sort_index()
    
    results = []
    
    for symbol in ohlcv.index.get_level_values(1).unique():
        symbol_data = ohlcv.xs(symbol, level=1).sort_index()
        
        # Current regime indicators
        returns = symbol_data['close'].pct_change()
        current_vol = returns.rolling(lookback_vol).std() * np.sqrt(252)  # Annualized
        current_trend = symbol_data['close'].rolling(lookback_trend).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0]
        )
        
        # Future regime (what we want to predict)
        future_vol = current_vol.shift(-forward_horizon)
        future_trend = current_trend.shift(-forward_horizon)
        
        # Create regime labels
        vol_threshold = current_vol.quantile(0.7)  # Top 30% = high vol
        
        def create_regime_label(curr_vol, curr_trend, fut_vol, fut_trend):
            if pd.isna(fut_vol) or pd.isna(fut_trend):
                return 'UNKNOWN'
            
            # Current state
            high_vol_now = curr_vol > vol_threshold
            bull_trend_now = curr_trend > 0.02  # 2% trend threshold
            
            # Future state (what we're predicting)
            high_vol_future = fut_vol > vol_threshold
            bull_trend_future = fut_trend > 0.02
            
            if high_vol_future and bull_trend_future:
                return 'HIGH_VOL_BULL'
            elif high_vol_future and not bull_trend_future:
                return 'HIGH_VOL_BEAR'
            elif not high_vol_future and bull_trend_future:
                return 'LOW_VOL_BULL'
            else:
                return 'LOW_VOL_BEAR'
        
        regimes = pd.Series([
            create_regime_label(cv, ct, fv, ft)
            for cv, ct, fv, ft in zip(current_vol, current_trend, future_vol, future_trend)
        ], index=symbol_data.index)
        
        regime_df = pd.DataFrame({
            'current_vol': current_vol,
            'current_trend': current_trend,
            'future_vol': future_vol,
            'future_trend': future_trend,
            'regime_label': regimes,
            'symbol': symbol
        })
        
        results.append(regime_df)
    
    combined = pd.concat(results).set_index('symbol', append=True).swaplevel().sort_index()
    return combined


def earnings_momentum_labels(
    ohlcv: pd.DataFrame,
    earnings_dates: Optional[pd.DataFrame] = None,
    window_before: int = 20,
    window_after: int = 10
) -> pd.DataFrame:
    """
    Predict earnings momentum - stocks often trend before/after earnings.
    This is a more event-driven approach.
    """
    if not isinstance(ohlcv.index, pd.MultiIndex):
        ohlcv = ohlcv.set_index('symbol', append=True).swaplevel().sort_index()
    
    results = []
    
    for symbol in ohlcv.index.get_level_values(1).unique():
        symbol_data = ohlcv.xs(symbol, level=1).sort_index()
        returns = symbol_data['close'].pct_change()
        
        # Simple momentum without earnings dates (quarterly approximation)
        # In practice, you'd want actual earnings dates
        quarterly_dates = symbol_data.index[::63]  # Approximate quarterly
        
        momentum_labels = pd.Series('NEUTRAL', index=symbol_data.index)
        
        for earn_date in quarterly_dates:
            try:
                # Pre-earnings period
                pre_start = earn_date - pd.Timedelta(days=window_before)
                pre_end = earn_date
                pre_returns = returns.loc[pre_start:pre_end]
                
                # Post-earnings period  
                post_start = earn_date
                post_end = earn_date + pd.Timedelta(days=window_after)
                post_returns = returns.loc[post_start:post_end]
                
                if len(pre_returns) > 5 and len(post_returns) > 3:
                    pre_momentum = pre_returns.sum()
                    post_momentum = post_returns.sum()
                    
                    # Label the pre-earnings period with post-earnings outcome
                    if post_momentum > 0.05:  # Strong post-earnings pop
                        momentum_labels.loc[pre_start:pre_end] = 'STRONG_POSITIVE'
                    elif post_momentum > 0.02:
                        momentum_labels.loc[pre_start:pre_end] = 'POSITIVE'
                    elif post_momentum < -0.05:
                        momentum_labels.loc[pre_start:pre_end] = 'STRONG_NEGATIVE'
                    elif post_momentum < -0.02:
                        momentum_labels.loc[pre_start:pre_end] = 'NEGATIVE'
                        
            except (KeyError, IndexError):
                continue
        
        result_df = pd.DataFrame({
            'momentum_label': momentum_labels,
            'symbol': symbol
        })
        results.append(result_df)
    
    combined = pd.concat(results).set_index('symbol', append=True).swaplevel().sort_index()
    return combined


def mean_reversion_labels(
    ohlcv: pd.DataFrame,
    lookback_periods: List[int] = [5, 10, 20],
    reversion_threshold: float = 0.02,
    hold_period: int = 5
) -> pd.DataFrame:
    """
    Predict mean reversion opportunities - when extreme moves reverse.
    Good for short-term trading strategies.
    """
    if not isinstance(ohlcv.index, pd.MultiIndex):
        ohlcv = ohlcv.set_index('symbol', append=True).swaplevel().sort_index()
    
    results = []
    
    for symbol in ohlcv.index.get_level_values(1).unique():
        symbol_data = ohlcv.xs(symbol, level=1).sort_index()
        returns = symbol_data['close'].pct_change()
        
        for lookback in lookback_periods:
            # Identify extreme moves
            rolling_ret = returns.rolling(lookback).sum()
            
            # Future reversion (what we want to predict)
            future_ret = returns.shift(-hold_period).rolling(hold_period).sum()
            
            # Create labels
            def create_reversion_label(past_ret, fut_ret):
                if pd.isna(past_ret) or pd.isna(fut_ret):
                    return 'UNKNOWN'
                
                # Strong past move up -> predict reversion down
                if past_ret > reversion_threshold:
                    if fut_ret < -reversion_threshold/2:
                        return 'REVERT_DOWN'  # Correct prediction
                    else:
                        return 'NO_REVERT'    # Momentum continues
                
                # Strong past move down -> predict reversion up  
                elif past_ret < -reversion_threshold:
                    if fut_ret > reversion_threshold/2:
                        return 'REVERT_UP'    # Correct prediction
                    else:
                        return 'NO_REVERT'    # Continued weakness
                
                else:
                    return 'NEUTRAL'          # No extreme move
            
            reversion_labels = pd.Series([
                create_reversion_label(pr, fr)
                for pr, fr in zip(rolling_ret, future_ret)
            ], index=symbol_data.index)
            
            result_df = pd.DataFrame({
                f'past_ret_{lookback}d': rolling_ret,
                f'future_ret_{hold_period}d': future_ret,
                f'reversion_label_{lookback}d': reversion_labels,
                'symbol': symbol
            })
            
            results.append(result_df)
    
    combined = pd.concat(results, axis=1)
    # Remove duplicate symbol columns
    symbol_cols = [col for col in combined.columns if col == 'symbol']
    if len(symbol_cols) > 1:
        combined = combined.drop(columns=symbol_cols[1:])
    
    return combined.set_index('symbol', append=True).swaplevel().sort_index()


# Utility function to combine multiple labeling strategies
def create_multi_target_labels(
    ohlcv: pd.DataFrame,
    include_ranking: bool = True,
    include_regime: bool = True,
    include_reversion: bool = False,
    include_earnings: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Create multiple types of labels for ensemble or multi-task learning.
    """
    labels = {}
    
    if include_ranking:
        print("Creating cross-sectional ranking labels...")
        labels['ranking'] = cross_sectional_ranking_labels(ohlcv)
    
    if include_regime:
        print("Creating volatility regime labels...")
        labels['regime'] = volatility_regime_labels(ohlcv)
    
    if include_reversion:
        print("Creating mean reversion labels...")
        labels['reversion'] = mean_reversion_labels(ohlcv)
    
    if include_earnings:
        print("Creating earnings momentum labels...")
        labels['earnings'] = earnings_momentum_labels(ohlcv)
    
    return labels


# Example usage
if __name__ == "__main__":
    # This would be run with your actual OHLCV data
    print("Advanced labeling strategies for algorithmic trading")
    print("="*50)
    
    # Simulate some data for demonstration
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    np.random.seed(42)
    data = []
    
    for symbol in symbols:
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
        df = pd.DataFrame({
            'timestamp': dates,
            'symbol': symbol,
            'close': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'volume': np.random.randint(1000000, 5000000, len(dates))
        })
        data.append(df)
    
    ohlcv_demo = pd.concat(data).set_index(['timestamp', 'symbol'])
    
    # Test different labeling strategies
    print("\n1. Cross-sectional ranking labels:")
    ranking_labels = cross_sectional_ranking_labels(ohlcv_demo)
    print(f"Shape: {ranking_labels.shape}")
    print(f"Categories: {ranking_labels['category_10d'].value_counts()}")
    
    print("\n2. Volatility regime labels:")
    regime_labels = volatility_regime_labels(ohlcv_demo)
    print(f"Shape: {regime_labels.shape}")
    print(f"Regimes: {regime_labels['regime_label'].value_counts()}")
    
    print("\n3. Mean reversion labels:")
    reversion_labels = mean_reversion_labels(ohlcv_demo)
    print(f"Shape: {reversion_labels.shape}")