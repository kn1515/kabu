import pytest
import numpy as np
from random_walk import RandomWalkModel, create_sample_data
from visualizer import TradingVisualizer


class TestRandomWalkModel:
    """
    ランダムウォークモデルのテストクラス
    """
    
    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.model = RandomWalkModel(window_size=10)
        self.sample_prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
    
    def test_model_initialization(self):
        """モデルの初期化テスト"""
        assert self.model.drift == 0.0
        assert self.model.volatility == 0.02
        assert self.model.window_size == 10
        assert self.model.confidence_level == 0.95
        assert len(self.model.price_history) == 0
        assert len(self.model.returns_history) == 0
    
    def test_add_price(self):
        """価格追加機能のテスト"""
        # 最初の価格を追加
        self.model.add_price(100)
        assert len(self.model.price_history) == 1
        assert len(self.model.returns_history) == 0
        
        # 2番目の価格を追加
        self.model.add_price(102)
        assert len(self.model.price_history) == 2
        assert len(self.model.returns_history) == 1
        
        # 収益率の計算が正しいかチェック
        expected_return = (102 - 100) / 100
        assert abs(self.model.returns_history[0] - expected_return) < 1e-10
    
    def test_window_size_limit(self):
        """ウィンドウサイズ制限のテスト"""
        # ウィンドウサイズを超える価格を追加
        for i, price in enumerate(self.sample_prices + [110, 111, 112]):
            self.model.add_price(price)
        
        # ウィンドウサイズを超えないことを確認
        assert len(self.model.price_history) <= self.model.window_size
        assert len(self.model.returns_history) <= self.model.window_size
        
        # 最新の価格が保持されていることを確認
        assert self.model.price_history[-1] == 112
    
    def test_parameter_update(self):
        """パラメータ更新機能のテスト"""
        # 複数の価格を追加
        for price in self.sample_prices:
            self.model.add_price(price)
        
        initial_drift = self.model.drift
        initial_volatility = self.model.volatility
        
        # パラメータを更新
        self.model.update_parameters()
        
        # パラメータが更新されていることを確認
        assert self.model.drift != initial_drift
        assert self.model.volatility != initial_volatility
        
        # 収益率の平均と標準偏差が正しく計算されていることを確認
        expected_drift = np.mean(self.model.returns_history)
        expected_volatility = np.std(self.model.returns_history)
        
        assert abs(self.model.drift - expected_drift) < 1e-10
        assert abs(self.model.volatility - expected_volatility) < 1e-10
    
    def test_simulation_future_prices(self):
        """将来価格シミュレーション機能のテスト"""
        # 価格データを追加
        for price in self.sample_prices:
            self.model.add_price(price)
        
        # シミュレーション実行
        steps = 5
        num_simulations = 100
        simulated_prices = self.model.simulate_future_prices(steps, num_simulations)
        
        # 結果の形状を確認
        assert simulated_prices.shape == (num_simulations, steps)
        
        # 価格が正の値であることを確認
        assert np.all(simulated_prices > 0)
    
    def test_price_forecast(self):
        """価格予測機能のテスト"""
        # 価格データを追加
        for price in self.sample_prices:
            self.model.add_price(price)
        
        # 予測実行
        steps = 5
        mean_prices, lower_bound, upper_bound = self.model.get_price_forecast(steps)
        
        # 結果の形状を確認
        assert len(mean_prices) == steps
        assert len(lower_bound) == steps
        assert len(upper_bound) == steps
        
        # 信頼区間が正しく設定されていることを確認
        assert np.all(lower_bound <= mean_prices)
        assert np.all(mean_prices <= upper_bound)
    
    def test_buy_sell_probability(self):
        """売買確率計算のテスト"""
        # 価格データを追加
        for price in self.sample_prices:
            self.model.add_price(price)
        
        # 売買確率計算
        buy_prob, sell_prob = self.model.calculate_buy_sell_probability()
        
        # 確率が0-1の範囲内であることを確認
        assert 0 <= buy_prob <= 1
        assert 0 <= sell_prob <= 1
    
    def test_trading_signal_generation(self):
        """取引シグナル生成のテスト"""
        # 価格データを追加
        for price in self.sample_prices:
            self.model.add_price(price)
        
        # 取引シグナル生成
        signal = self.model.generate_trading_signal()
        
        # シグナルが正しい値であることを確認
        assert signal in ['buy', 'sell', None]
    
    def test_empty_price_history(self):
        """空の価格履歴に対するテスト"""
        # 価格履歴が空の場合の動作を確認
        buy_prob, sell_prob = self.model.calculate_buy_sell_probability()
        assert buy_prob == 0.0
        assert sell_prob == 0.0
        
        # シミュレーションが例外を発生させることを確認
        with pytest.raises(ValueError):
            self.model.simulate_future_prices(5, 10)
    
    def test_statistics(self):
        """統計情報取得のテスト"""
        # 価格データを追加
        for price in self.sample_prices:
            self.model.add_price(price)
        
        # 統計情報を取得
        stats = self.model.get_statistics()
        
        # 必要なキーが存在することを確認
        required_keys = ['drift', 'volatility', 'price_count', 'current_price', 
                        'window_size', 'confidence_level']
        for key in required_keys:
            assert key in stats
        
        # 値が正しいことを確認
        assert stats['price_count'] == len(self.sample_prices)
        assert stats['current_price'] == self.sample_prices[-1]
        assert stats['window_size'] == self.model.window_size


class TestSampleDataGeneration:
    """
    サンプルデータ生成のテスト
    """
    
    def test_sample_data_generation(self):
        """サンプルデータ生成のテスト"""
        days = 50
        initial_price = 1000.0
        
        sample_data = create_sample_data(days, initial_price)
        
        # データの長さを確認
        assert len(sample_data) == days
        
        # 初期価格を確認
        assert sample_data[0] == initial_price
        
        # 全ての価格が正の値であることを確認
        assert all(price > 0 for price in sample_data)


class TestTradingVisualizer:
    """
    可視化機能のテスト
    """
    
    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.visualizer = TradingVisualizer()
        self.sample_prices = create_sample_data(50, 1000)
    
    def test_visualizer_initialization(self):
        """可視化クラスの初期化テスト"""
        assert self.visualizer.figsize == (15, 10)
    
    def test_plot_methods_exist(self):
        """プロットメソッドの存在確認"""
        # 主要なプロットメソッドが存在することを確認
        assert hasattr(self.visualizer, 'plot_price_with_signals')
        assert hasattr(self.visualizer, 'plot_random_walk_analysis')
        assert hasattr(self.visualizer, 'plot_model_comparison')
        assert hasattr(self.visualizer, 'plot_performance_metrics')


class TestIntegration:
    """
    統合テスト
    """
    
    def test_full_workflow(self):
        """完全なワークフローのテスト"""
        # サンプルデータ生成
        sample_prices = create_sample_data(30, 1000)
        
        # モデル初期化
        model = RandomWalkModel(window_size=20)
        
        # 価格データを順次追加
        for price in sample_prices:
            model.add_price(price)
        
        # 各機能が正常に動作することを確認
        model.update_parameters()
        
        # 予測機能
        mean_prices, lower_bound, upper_bound = model.get_price_forecast(10)
        assert len(mean_prices) == 10
        
        # 売買確率
        buy_prob, sell_prob = model.calculate_buy_sell_probability()
        assert isinstance(buy_prob, (int, float))
        assert isinstance(sell_prob, (int, float))
        
        # 取引シグナル
        signal = model.generate_trading_signal()
        assert signal in ['buy', 'sell', None]
        
        # 統計情報
        stats = model.get_statistics()
        assert 'current_price' in stats
        assert stats['current_price'] == sample_prices[-1]


def test_model_robustness():
    """
    モデルの堅牢性テスト
    """
    model = RandomWalkModel(window_size=5)
    
    # 極端な価格変動をテスト
    extreme_prices = [100, 200, 50, 300, 25, 400, 10]
    
    for price in extreme_prices:
        model.add_price(price)
    
    # 例外が発生しないことを確認
    try:
        model.update_parameters()
        model.calculate_buy_sell_probability()
        model.generate_trading_signal()
        model.get_price_forecast(5)
        success = True
    except Exception as e:
        success = False
        print(f"Robustness test failed: {e}")
    
    assert success, "モデルは極端な価格変動に対して堅牢である必要があります"


if __name__ == "__main__":
    # テストを実行
    pytest.main([__file__, "-v"])