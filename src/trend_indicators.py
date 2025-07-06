import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta


class TrendIndicators:
    """
    トレンド系指標を計算するクラス
    移動平均、MACD、RSI、ボリンジャーバンド等を実装
    """
    
    def __init__(self):
        """初期化"""
        self.logger = logging.getLogger(__name__)
    
    def sma(self, prices: List[float], period: int) -> List[float]:
        """
        単純移動平均（SMA）を計算
        
        Args:
            prices: 価格データ
            period: 移動平均期間
            
        Returns:
            移動平均値のリスト
        """
        if len(prices) < period:
            return []
        
        sma_values = []
        for i in range(period - 1, len(prices)):
            sma = sum(prices[i - period + 1:i + 1]) / period
            sma_values.append(sma)
        
        return sma_values
    
    def ema(self, prices: List[float], period: int) -> List[float]:
        """
        指数移動平均（EMA）を計算
        
        Args:
            prices: 価格データ
            period: 移動平均期間
            
        Returns:
            指数移動平均値のリスト
        """
        if len(prices) < period:
            return []
        
        multiplier = 2 / (period + 1)
        ema_values = []
        
        # 初期値はSMAを使用
        initial_sma = sum(prices[:period]) / period
        ema_values.append(initial_sma)
        
        for i in range(period, len(prices)):
            ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)
        
        return ema_values
    
    def macd(self, prices: List[float], fast_period: int = 12, 
             slow_period: int = 26, signal_period: int = 9) -> Dict[str, List[float]]:
        """
        MACD（Moving Average Convergence Divergence）を計算
        
        Args:
            prices: 価格データ
            fast_period: 短期EMA期間
            slow_period: 長期EMA期間
            signal_period: シグナル線EMA期間
            
        Returns:
            MACD線、シグナル線、ヒストグラムの辞書
        """
        if len(prices) < slow_period:
            return {'macd': [], 'signal': [], 'histogram': []}
        
        # 短期・長期EMAを計算
        ema_fast = self.ema(prices, fast_period)
        ema_slow = self.ema(prices, slow_period)
        
        # MACD線を計算（短期EMA - 長期EMA）
        macd_line = []
        start_idx = slow_period - fast_period
        for i in range(len(ema_slow)):
            macd_value = ema_fast[i + start_idx] - ema_slow[i]
            macd_line.append(macd_value)
        
        # シグナル線を計算（MACD線のEMA）
        signal_line = self.ema(macd_line, signal_period)
        
        # ヒストグラムを計算（MACD線 - シグナル線）
        histogram = []
        signal_start_idx = signal_period - 1
        for i in range(len(signal_line)):
            hist_value = macd_line[i + signal_start_idx] - signal_line[i]
            histogram.append(hist_value)
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """
        RSI（Relative Strength Index）を計算
        
        Args:
            prices: 価格データ
            period: RSI計算期間
            
        Returns:
            RSI値のリスト
        """
        if len(prices) < period + 1:
            return []
        
        # 価格変動を計算
        price_changes = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            price_changes.append(change)
        
        rsi_values = []
        for i in range(period - 1, len(price_changes)):
            gains = []
            losses = []
            
            for j in range(i - period + 1, i + 1):
                if price_changes[j] > 0:
                    gains.append(price_changes[j])
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(price_changes[j]))
            
            avg_gain = sum(gains) / period
            avg_loss = sum(losses) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        return rsi_values
    
    def bollinger_bands(self, prices: List[float], period: int = 20, 
                       std_dev: float = 2.0) -> Dict[str, List[float]]:
        """
        ボリンジャーバンドを計算
        
        Args:
            prices: 価格データ
            period: 移動平均期間
            std_dev: 標準偏差の倍数
            
        Returns:
            上限線、中央線（SMA）、下限線の辞書
        """
        if len(prices) < period:
            return {'upper': [], 'middle': [], 'lower': []}
        
        sma_values = self.sma(prices, period)
        upper_band = []
        lower_band = []
        
        for i in range(period - 1, len(prices)):
            # 標準偏差を計算
            price_slice = prices[i - period + 1:i + 1]
            std = np.std(price_slice)
            sma_idx = i - period + 1
            
            upper_band.append(sma_values[sma_idx] + (std * std_dev))
            lower_band.append(sma_values[sma_idx] - (std * std_dev))
        
        return {
            'upper': upper_band,
            'middle': sma_values,
            'lower': lower_band
        }
    
    def stochastic(self, highs: List[float], lows: List[float], 
                  closes: List[float], k_period: int = 14, 
                  d_period: int = 3) -> Dict[str, List[float]]:
        """
        ストキャスティクスを計算
        
        Args:
            highs: 高値データ
            lows: 安値データ
            closes: 終値データ
            k_period: %K期間
            d_period: %D期間
            
        Returns:
            %K線、%D線の辞書
        """
        if len(closes) < k_period:
            return {'k': [], 'd': []}
        
        k_values = []
        
        for i in range(k_period - 1, len(closes)):
            high_max = max(highs[i - k_period + 1:i + 1])
            low_min = min(lows[i - k_period + 1:i + 1])
            
            if high_max == low_min:
                k_value = 50  # 価格変動がない場合
            else:
                k_value = ((closes[i] - low_min) / (high_max - low_min)) * 100
            
            k_values.append(k_value)
        
        # %D線は%K線の移動平均
        d_values = self.sma(k_values, d_period)
        
        return {
            'k': k_values,
            'd': d_values
        }
    
    def williams_r(self, highs: List[float], lows: List[float], 
                   closes: List[float], period: int = 14) -> List[float]:
        """
        ウィリアムズ%Rを計算
        
        Args:
            highs: 高値データ
            lows: 安値データ
            closes: 終値データ
            period: 計算期間
            
        Returns:
            ウィリアムズ%R値のリスト
        """
        if len(closes) < period:
            return []
        
        williams_r_values = []
        
        for i in range(period - 1, len(closes)):
            high_max = max(highs[i - period + 1:i + 1])
            low_min = min(lows[i - period + 1:i + 1])
            
            if high_max == low_min:
                williams_r = -50  # 価格変動がない場合
            else:
                williams_r = ((high_max - closes[i]) / (high_max - low_min)) * -100
            
            williams_r_values.append(williams_r)
        
        return williams_r_values


class TrendSignalAnalyzer:
    """
    トレンド系指標から買いサインを判定するクラス
    """
    
    def __init__(self):
        """初期化"""
        self.indicators = TrendIndicators()
        self.logger = logging.getLogger(__name__)
    
    def analyze_ma_crossover(self, prices: List[float], 
                           short_period: int = 5, 
                           long_period: int = 25) -> Dict[str, any]:
        """
        移動平均クロスオーバーを分析
        
        Args:
            prices: 価格データ
            short_period: 短期移動平均期間
            long_period: 長期移動平均期間
            
        Returns:
            分析結果の辞書
        """
        if len(prices) < long_period:
            return {'signal': None, 'strength': 0, 'details': 'データ不足'}
        
        short_ma = self.indicators.sma(prices, short_period)
        long_ma = self.indicators.sma(prices, long_period)
        
        if len(short_ma) < 2 or len(long_ma) < 2:
            return {'signal': None, 'strength': 0, 'details': 'データ不足'}
        
        # 現在と前回の移動平均を比較
        current_short = short_ma[-1]
        current_long = long_ma[-1]
        prev_short = short_ma[-2]
        prev_long = long_ma[-2]
        
        # ゴールデンクロスの判定
        if prev_short <= prev_long and current_short > current_long:
            strength = (current_short - current_long) / current_long * 100
            return {
                'signal': 'buy',
                'strength': min(strength, 100),
                'details': f'ゴールデンクロス発生 短期MA:{current_short:.2f} 長期MA:{current_long:.2f}'
            }
        
        # デッドクロスの判定
        elif prev_short >= prev_long and current_short < current_long:
            strength = (current_long - current_short) / current_long * 100
            return {
                'signal': 'sell',
                'strength': min(strength, 100),
                'details': f'デッドクロス発生 短期MA:{current_short:.2f} 長期MA:{current_long:.2f}'
            }
        
        # トレンド継続の判定
        elif current_short > current_long:
            strength = (current_short - current_long) / current_long * 100
            return {
                'signal': 'hold_buy',
                'strength': min(strength, 100),
                'details': f'上昇トレンド継続 短期MA:{current_short:.2f} 長期MA:{current_long:.2f}'
            }
        
        else:
            strength = (current_long - current_short) / current_long * 100
            return {
                'signal': 'hold_sell',
                'strength': min(strength, 100),
                'details': f'下降トレンド継続 短期MA:{current_short:.2f} 長期MA:{current_long:.2f}'
            }
    
    def analyze_macd_signal(self, prices: List[float]) -> Dict[str, any]:
        """
        MACDシグナルを分析
        
        Args:
            prices: 価格データ
            
        Returns:
            分析結果の辞書
        """
        macd_data = self.indicators.macd(prices)
        
        if len(macd_data['histogram']) < 2:
            return {'signal': None, 'strength': 0, 'details': 'データ不足'}
        
        current_hist = macd_data['histogram'][-1]
        prev_hist = macd_data['histogram'][-2]
        current_macd = macd_data['macd'][-1]
        current_signal = macd_data['signal'][-1]
        
        # ヒストグラムの変化を確認
        if prev_hist <= 0 and current_hist > 0:
            strength = min(abs(current_hist) * 10, 100)
            return {
                'signal': 'buy',
                'strength': strength,
                'details': f'MACDヒストグラム買いシグナル ヒスト:{current_hist:.4f}'
            }
        
        elif prev_hist >= 0 and current_hist < 0:
            strength = min(abs(current_hist) * 10, 100)
            return {
                'signal': 'sell',
                'strength': strength,
                'details': f'MACDヒストグラム売りシグナル ヒスト:{current_hist:.4f}'
            }
        
        # MACD線とシグナル線の位置関係
        elif current_macd > current_signal and current_hist > 0:
            strength = min(abs(current_hist) * 10, 100)
            return {
                'signal': 'hold_buy',
                'strength': strength,
                'details': f'MACD強気継続 MACD:{current_macd:.4f} シグナル:{current_signal:.4f}'
            }
        
        else:
            strength = min(abs(current_hist) * 10, 100)
            return {
                'signal': 'hold_sell',
                'strength': strength,
                'details': f'MACD弱気継続 MACD:{current_macd:.4f} シグナル:{current_signal:.4f}'
            }
    
    def analyze_rsi_signal(self, prices: List[float]) -> Dict[str, any]:
        """
        RSIシグナルを分析
        
        Args:
            prices: 価格データ
            
        Returns:
            分析結果の辞書
        """
        rsi_values = self.indicators.rsi(prices)
        
        if len(rsi_values) < 2:
            return {'signal': None, 'strength': 0, 'details': 'データ不足'}
        
        current_rsi = rsi_values[-1]
        prev_rsi = rsi_values[-2]
        
        # RSI買いシグナル（30以下から上昇）
        if prev_rsi <= 30 and current_rsi > 30:
            strength = min((current_rsi - 30) * 2, 100)
            return {
                'signal': 'buy',
                'strength': strength,
                'details': f'RSI買いシグナル（売られすぎから回復） RSI:{current_rsi:.1f}'
            }
        
        # RSI売りシグナル（70以上から下降）
        elif prev_rsi >= 70 and current_rsi < 70:
            strength = min((70 - current_rsi) * 2, 100)
            return {
                'signal': 'sell',
                'strength': strength,
                'details': f'RSI売りシグナル（買われすぎから下落） RSI:{current_rsi:.1f}'
            }
        
        # RSI中立域での判定
        elif 30 < current_rsi < 70:
            if current_rsi > 50:
                strength = (current_rsi - 50) * 2
                return {
                    'signal': 'hold_buy',
                    'strength': strength,
                    'details': f'RSI中立域（上向き） RSI:{current_rsi:.1f}'
                }
            else:
                strength = (50 - current_rsi) * 2
                return {
                    'signal': 'hold_sell',
                    'strength': strength,
                    'details': f'RSI中立域（下向き） RSI:{current_rsi:.1f}'
                }
        
        # 極端な値の継続
        elif current_rsi <= 30:
            strength = (30 - current_rsi) * 2
            return {
                'signal': 'potential_buy',
                'strength': strength,
                'details': f'RSI売られすぎ継続 RSI:{current_rsi:.1f}'
            }
        
        else:  # current_rsi >= 70
            strength = (current_rsi - 70) * 2
            return {
                'signal': 'potential_sell',
                'strength': strength,
                'details': f'RSI買われすぎ継続 RSI:{current_rsi:.1f}'
            }
    
    def analyze_bollinger_bands(self, prices: List[float]) -> Dict[str, any]:
        """
        ボリンジャーバンドを分析
        
        Args:
            prices: 価格データ
            
        Returns:
            分析結果の辞書
        """
        bb_data = self.indicators.bollinger_bands(prices)
        
        if len(bb_data['upper']) < 2:
            return {'signal': None, 'strength': 0, 'details': 'データ不足'}
        
        current_price = prices[-1]
        upper_band = bb_data['upper'][-1]
        middle_band = bb_data['middle'][-1]
        lower_band = bb_data['lower'][-1]
        
        # バンドの位置による判定
        band_width = upper_band - lower_band
        price_position = (current_price - lower_band) / band_width * 100
        
        if current_price <= lower_band:
            strength = min((lower_band - current_price) / lower_band * 100, 100)
            return {
                'signal': 'buy',
                'strength': strength,
                'details': f'ボリンジャーバンド下限タッチ（買いシグナル） 価格:{current_price:.2f} 下限:{lower_band:.2f}'
            }
        
        elif current_price >= upper_band:
            strength = min((current_price - upper_band) / upper_band * 100, 100)
            return {
                'signal': 'sell',
                'strength': strength,
                'details': f'ボリンジャーバンド上限タッチ（売りシグナル） 価格:{current_price:.2f} 上限:{upper_band:.2f}'
            }
        
        elif current_price > middle_band:
            strength = (current_price - middle_band) / (upper_band - middle_band) * 100
            return {
                'signal': 'hold_buy',
                'strength': strength,
                'details': f'ボリンジャーバンド上半分 価格:{current_price:.2f} 中央:{middle_band:.2f}'
            }
        
        else:
            strength = (middle_band - current_price) / (middle_band - lower_band) * 100
            return {
                'signal': 'hold_sell',
                'strength': strength,
                'details': f'ボリンジャーバンド下半分 価格:{current_price:.2f} 中央:{middle_band:.2f}'
            }
    
    def get_comprehensive_signal(self, prices: List[float], 
                               highs: List[float] = None,
                               lows: List[float] = None) -> Dict[str, any]:
        """
        複数の指標を組み合わせた総合的なシグナル判定
        
        Args:
            prices: 価格データ
            highs: 高値データ（オプション）
            lows: 安値データ（オプション）
            
        Returns:
            総合判定結果の辞書
        """
        if len(prices) < 30:
            return {'signal': None, 'strength': 0, 'details': 'データ不足'}
        
        # 各指標の分析
        ma_result = self.analyze_ma_crossover(prices)
        macd_result = self.analyze_macd_signal(prices)
        rsi_result = self.analyze_rsi_signal(prices)
        bb_result = self.analyze_bollinger_bands(prices)
        
        # シグナルの重み付け
        signal_weights = {
            'buy': 0,
            'sell': 0,
            'hold_buy': 0,
            'hold_sell': 0,
            'potential_buy': 0,
            'potential_sell': 0
        }
        
        results = [ma_result, macd_result, rsi_result, bb_result]
        details = []
        
        for result in results:
            if result['signal'] in signal_weights:
                weight = result['strength'] / 100
                signal_weights[result['signal']] += weight
                details.append(result['details'])
        
        # 買いシグナルの合計
        buy_score = signal_weights['buy'] * 3 + signal_weights['hold_buy'] * 2 + signal_weights['potential_buy'] * 1
        sell_score = signal_weights['sell'] * 3 + signal_weights['hold_sell'] * 2 + signal_weights['potential_sell'] * 1
        
        # 最終判定
        if buy_score > sell_score and buy_score > 2.0:
            final_signal = 'buy'
            strength = min(buy_score * 25, 100)
        elif sell_score > buy_score and sell_score > 2.0:
            final_signal = 'sell'
            strength = min(sell_score * 25, 100)
        elif buy_score > sell_score:
            final_signal = 'hold_buy'
            strength = min(buy_score * 25, 100)
        else:
            final_signal = 'hold_sell'
            strength = min(sell_score * 25, 100)
        
        return {
            'signal': final_signal,
            'strength': strength,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'details': details,
            'individual_results': {
                'ma': ma_result,
                'macd': macd_result,
                'rsi': rsi_result,
                'bollinger': bb_result
            }
        }


if __name__ == "__main__":
    # サンプルテスト
    from random_walk import create_sample_data
    
    print("トレンド系指標のテスト実行")
    
    # サンプルデータ生成
    sample_prices = create_sample_data(50, 1000)
    
    # 指標計算
    indicators = TrendIndicators()
    analyzer = TrendSignalAnalyzer()
    
    # 各指標のテスト
    sma_values = indicators.sma(sample_prices, 10)
    rsi_values = indicators.rsi(sample_prices)
    macd_data = indicators.macd(sample_prices)
    
    print(f"SMA(10)最新値: {sma_values[-1]:.2f}")
    print(f"RSI最新値: {rsi_values[-1]:.2f}")
    print(f"MACD最新値: {macd_data['macd'][-1]:.4f}")
    
    # 総合シグナル判定
    comprehensive = analyzer.get_comprehensive_signal(sample_prices)
    print(f"\n総合判定:")
    print(f"シグナル: {comprehensive['signal']}")
    print(f"強度: {comprehensive['strength']:.1f}")
    print(f"買いスコア: {comprehensive['buy_score']:.2f}")
    print(f"売りスコア: {comprehensive['sell_score']:.2f}")
    
    print("\n各指標の詳細:")
    for detail in comprehensive['details']:
        print(f"- {detail}")