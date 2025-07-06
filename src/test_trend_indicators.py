import pytest
import numpy as np
from trend_indicators import TrendIndicators, TrendSignalAnalyzer
from random_walk import create_sample_data


class TestTrendIndicators:
    """
    トレンド系指標のテストクラス
    """
    
    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.indicators = TrendIndicators()
        self.sample_prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 
                             111, 110, 112, 114, 113, 115, 117, 116, 118, 120]
    
    def test_sma_calculation(self):
        """単純移動平均の計算テスト"""
        # 10期間のSMAを計算
        sma_values = self.indicators.sma(self.sample_prices, 10)
        
        # 結果の長さを確認
        expected_length = len(self.sample_prices) - 10 + 1
        assert len(sma_values) == expected_length
        
        # 最初の値を手動計算で確認
        expected_first_sma = sum(self.sample_prices[:10]) / 10
        assert abs(sma_values[0] - expected_first_sma) < 1e-10
        
        # データ不足の場合
        short_prices = [100, 102, 103]
        sma_short = self.indicators.sma(short_prices, 10)
        assert len(sma_short) == 0
    
    def test_ema_calculation(self):
        """指数移動平均の計算テスト"""
        # 10期間のEMAを計算
        ema_values = self.indicators.ema(self.sample_prices, 10)
        
        # 結果の長さを確認
        expected_length = len(self.sample_prices) - 10 + 1
        assert len(ema_values) == expected_length
        
        # 初期値がSMAと同じことを確認
        expected_initial_sma = sum(self.sample_prices[:10]) / 10
        assert abs(ema_values[0] - expected_initial_sma) < 1e-10
        
        # EMAは単調でないことを確認（価格変動に敏感）
        assert len(set(ema_values)) > 1
    
    def test_macd_calculation(self):
        """MACD計算のテスト"""
        # より長いデータでテスト
        long_prices = create_sample_data(50, 100)
        macd_data = self.indicators.macd(long_prices)
        
        # 必要なキーが存在することを確認
        assert 'macd' in macd_data
        assert 'signal' in macd_data
        assert 'histogram' in macd_data
        
        # データが存在することを確認
        assert len(macd_data['macd']) > 0
        assert len(macd_data['signal']) > 0
        assert len(macd_data['histogram']) > 0
        
        # ヒストグラムがMACD線とシグナル線の差であることを確認
        if len(macd_data['histogram']) > 0:
            macd_line = macd_data['macd']
            signal_line = macd_data['signal']
            histogram = macd_data['histogram']
            
            # 最後の値で確認
            signal_start_idx = 9 - 1  # signal_period - 1
            expected_hist = macd_line[-1] - signal_line[-1]
            assert abs(histogram[-1] - expected_hist) < 1e-10
    
    def test_rsi_calculation(self):
        """RSI計算のテスト"""
        # より明確な傾向を持つデータで테스ト
        rising_prices = [100 + i for i in range(20)]  # 上昇トレンド
        falling_prices = [120 - i for i in range(20)]  # 下降トレンド
        
        # 上昇トレンドのRSI
        rsi_rising = self.indicators.rsi(rising_prices)
        assert len(rsi_rising) > 0
        assert rsi_rising[-1] > 50  # 上昇トレンドではRSIは50以上
        
        # 下降トレンドのRSI
        rsi_falling = self.indicators.rsi(falling_prices)
        assert len(rsi_falling) > 0
        assert rsi_falling[-1] < 50  # 下降トレンドではRSIは50以下
        
        # RSIは0-100の範囲内
        for rsi in rsi_rising + rsi_falling:
            assert 0 <= rsi <= 100
    
    def test_bollinger_bands(self):
        """ボリンジャーバンド計算のテスト"""
        bb_data = self.indicators.bollinger_bands(self.sample_prices, period=10)
        
        # 必要なキーが存在することを確認
        assert 'upper' in bb_data
        assert 'middle' in bb_data
        assert 'lower' in bb_data
        
        # データが存在することを確認
        assert len(bb_data['upper']) > 0
        assert len(bb_data['middle']) > 0
        assert len(bb_data['lower']) > 0
        
        # 上限 > 中央 > 下限の関係を確認
        for i in range(len(bb_data['upper'])):
            assert bb_data['upper'][i] > bb_data['middle'][i]
            assert bb_data['middle'][i] > bb_data['lower'][i]
    
    def test_stochastic(self):
        """ストキャスティクス計算のテスト"""
        # サンプルの高値、安値、終値データ
        highs = [p + 2 for p in self.sample_prices]
        lows = [p - 2 for p in self.sample_prices]
        closes = self.sample_prices
        
        stoch_data = self.indicators.stochastic(highs, lows, closes)
        
        # 必要なキーが存在することを確認
        assert 'k' in stoch_data
        assert 'd' in stoch_data
        
        # データが存在することを確認
        assert len(stoch_data['k']) > 0
        assert len(stoch_data['d']) > 0
        
        # %K、%Dは0-100の範囲内
        for k in stoch_data['k']:
            assert 0 <= k <= 100
        for d in stoch_data['d']:
            assert 0 <= d <= 100
    
    def test_williams_r(self):
        """ウィリアムズ%R計算のテスト"""
        # サンプルの高値、安値、終値データ
        highs = [p + 2 for p in self.sample_prices]
        lows = [p - 2 for p in self.sample_prices]
        closes = self.sample_prices
        
        williams_r = self.indicators.williams_r(highs, lows, closes)
        
        # データが存在することを確認
        assert len(williams_r) > 0
        
        # ウィリアムズ%Rは-100から0の範囲内
        for wr in williams_r:
            assert -100 <= wr <= 0


class TestTrendSignalAnalyzer:
    """
    トレンドシグナル分析のテストクラス
    """
    
    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.analyzer = TrendSignalAnalyzer()
    
    def test_ma_crossover_analysis(self):
        """移動平均クロスオーバー分析のテスト"""
        # ゴールデンクロスのシナリオ
        # 短期MAが長期MAを上抜く
        golden_cross_prices = [100] * 20 + [101, 102, 103, 104, 105] + [106] * 10
        
        result = self.analyzer.analyze_ma_crossover(golden_cross_prices, 5, 25)
        
        # シグナルが存在することを確認
        assert result['signal'] is not None
        assert 'strength' in result
        assert 'details' in result
        assert result['strength'] >= 0
    
    def test_macd_signal_analysis(self):
        """MACDシグナル分析のテスト"""
        # 十分な長さのデータを生成
        prices = create_sample_data(50, 100)
        
        result = self.analyzer.analyze_macd_signal(prices)
        
        # 基本的な結果構造を確認
        assert 'signal' in result
        assert 'strength' in result
        assert 'details' in result
        
        # 強度が適切な範囲内
        if result['signal'] is not None:
            assert 0 <= result['strength'] <= 100
    
    def test_rsi_signal_analysis(self):
        """RSIシグナル分析のテスト"""
        # 売られすぎからの回復シナリオ
        oversold_recovery = [100] * 10 + [90, 85, 80, 85, 90, 95, 100]
        
        result = self.analyzer.analyze_rsi_signal(oversold_recovery)
        
        # 基本的な結果構造を確認
        assert 'signal' in result
        assert 'strength' in result
        assert 'details' in result
        
        if result['signal'] is not None:
            assert 0 <= result['strength'] <= 100
    
    def test_bollinger_bands_analysis(self):
        """ボリンジャーバンド分析のテスト"""
        # 十分な長さのデータ
        prices = create_sample_data(30, 100)
        
        result = self.analyzer.analyze_bollinger_bands(prices)
        
        # 基本的な結果構造を確認
        assert 'signal' in result
        assert 'strength' in result
        assert 'details' in result
        
        if result['signal'] is not None:
            assert 0 <= result['strength'] <= 100
    
    def test_comprehensive_signal(self):
        """総合シグナル分析のテスト"""
        # 十分な長さのデータ
        prices = create_sample_data(50, 100)
        
        result = self.analyzer.get_comprehensive_signal(prices)
        
        # 基本的な結果構造を確認
        assert 'signal' in result
        assert 'strength' in result
        assert 'buy_score' in result
        assert 'sell_score' in result
        assert 'details' in result
        assert 'individual_results' in result
        
        # 個別結果の確認
        individual = result['individual_results']
        assert 'ma' in individual
        assert 'macd' in individual
        assert 'rsi' in individual
        assert 'bollinger' in individual
        
        # スコアの範囲確認
        assert result['buy_score'] >= 0
        assert result['sell_score'] >= 0
        
        if result['signal'] is not None:
            assert 0 <= result['strength'] <= 100
    
    def test_signal_types(self):
        """シグナルタイプの妥当性テスト"""
        valid_signals = ['buy', 'sell', 'hold_buy', 'hold_sell', 
                        'potential_buy', 'potential_sell', None]
        
        # 異なるパターンのデータでテスト
        test_patterns = [
            create_sample_data(50, 100),
            [100 + i for i in range(50)],  # 上昇トレンド
            [150 - i for i in range(50)],  # 下降トレンド
            [100] * 50,  # 横ばい
        ]
        
        for prices in test_patterns:
            result = self.analyzer.get_comprehensive_signal(prices)
            assert result['signal'] in valid_signals
    
    def test_edge_cases(self):
        """エッジケースのテスト"""
        # データ不足
        short_data = [100, 101, 102]
        result = self.analyzer.get_comprehensive_signal(short_data)
        assert result['signal'] is None
        assert result['strength'] == 0
        
        # 同じ価格の連続
        flat_data = [100] * 50
        result = self.analyzer.get_comprehensive_signal(flat_data)
        # エラーが発生しないことを確認
        assert 'signal' in result
        
        # 極端な価格変動
        extreme_data = [100, 200, 50, 300, 25, 400, 10] * 7  # 50データポイント
        result = self.analyzer.get_comprehensive_signal(extreme_data)
        # エラーが発生しないことを確認
        assert 'signal' in result


class TestIntegrationTrend:
    """
    トレンド分析の統合テスト
    """
    
    def test_full_trend_analysis_workflow(self):
        """完全なトレンド分析ワークフローのテスト"""
        # リアルなサンプルデータ
        prices = create_sample_data(60, 1000)
        
        # 指標計算
        indicators = TrendIndicators()
        analyzer = TrendSignalAnalyzer()
        
        # 各指標が正常に計算されることを確認
        sma = indicators.sma(prices, 20)
        ema = indicators.ema(prices, 20)
        macd = indicators.macd(prices)
        rsi = indicators.rsi(prices)
        bb = indicators.bollinger_bands(prices)
        
        assert len(sma) > 0
        assert len(ema) > 0
        assert len(macd['macd']) > 0
        assert len(rsi) > 0
        assert len(bb['middle']) > 0
        
        # 総合分析が正常に動作することを確認
        comprehensive = analyzer.get_comprehensive_signal(prices)
        
        assert comprehensive['signal'] is not None
        assert isinstance(comprehensive['strength'], (int, float))
        assert isinstance(comprehensive['buy_score'], (int, float))
        assert isinstance(comprehensive['sell_score'], (int, float))
        assert isinstance(comprehensive['details'], list)
    
    def test_trend_signal_consistency(self):
        """トレンドシグナルの一貫性テスト"""
        # 明確な上昇トレンドデータ
        uptrend_data = [1000 + i * 2 + np.random.normal(0, 1) for i in range(50)]
        
        # 明確な下降トレンドデータ
        downtrend_data = [1000 - i * 2 + np.random.normal(0, 1) for i in range(50)]
        
        analyzer = TrendSignalAnalyzer()
        
        # 上昇トレンドでは買いバイアスが期待される
        uptrend_result = analyzer.get_comprehensive_signal(uptrend_data)
        
        # 下降トレンドでは売りバイアスが期待される
        downtrend_result = analyzer.get_comprehensive_signal(downtrend_data)
        
        # 結果が論理的に一貫していることを確認
        # （必ずしも特定のシグナルが出るとは限らないが、エラーは発生しない）
        assert uptrend_result['signal'] is not None or uptrend_result['strength'] == 0
        assert downtrend_result['signal'] is not None or downtrend_result['strength'] == 0


def test_trend_indicators_robustness():
    """
    トレンド指標の堅牢性テスト
    """
    indicators = TrendIndicators()
    analyzer = TrendSignalAnalyzer()
    
    # 極端なケースでも例外が発生しないことを確認
    extreme_cases = [
        [1, 1000000, 1, 1000000] * 12,  # 極端な変動
        [0.001, 0.002, 0.003] * 16,     # 非常に小さな値
        [float('inf')] * 50,            # 無限大（実際には処理前にフィルタされる想定）
        [-100, -50, -25] * 16,          # 負の値
    ]
    
    for case in extreme_cases:
        try:
            # 無限大やNaNを除去
            filtered_case = [x for x in case if np.isfinite(x) and x > 0]
            
            if len(filtered_case) >= 30:
                result = analyzer.get_comprehensive_signal(filtered_case)
                # 例外が発生しないことを確認
                assert 'signal' in result
                
        except Exception as e:
            # 期待される例外（データ不足など）の場合はOK
            if "データ不足" not in str(e):
                raise e


if __name__ == "__main__":
    # テストを実行
    pytest.main([__file__, "-v"])