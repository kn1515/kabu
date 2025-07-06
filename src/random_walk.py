import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import logging
from datetime import datetime, timedelta


class RandomWalkModel:
    """
    ランダムウォークモデルを実装するクラス
    株価予測と売買判断に使用
    """
    
    def __init__(self, drift: float = 0.0, volatility: float = 0.02, 
                 window_size: int = 50, confidence_level: float = 0.95):
        """
        ランダムウォークモデルの初期化
        
        Args:
            drift: ドリフト項（日次期待収益率）
            volatility: ボラティリティ（日次標準偏差）
            window_size: 移動統計計算のウィンドウサイズ
            confidence_level: 信頼区間の水準
        """
        self.drift = drift
        self.volatility = volatility
        self.window_size = window_size
        self.confidence_level = confidence_level
        self.price_history = []
        self.returns_history = []
        
    def add_price(self, price: float) -> None:
        """
        新しい株価データを追加
        
        Args:
            price: 株価
        """
        self.price_history.append(price)
        
        # 収益率を計算
        if len(self.price_history) > 1:
            return_rate = (price - self.price_history[-2]) / self.price_history[-2]
            self.returns_history.append(return_rate)
        
        # ウィンドウサイズを超えた場合、古いデータを削除
        if len(self.price_history) > self.window_size:
            self.price_history.pop(0)
        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)
    
    def update_parameters(self) -> None:
        """
        直近データに基づいてドリフトとボラティリティを更新
        """
        if len(self.returns_history) < 2:
            return
        
        # ドリフト（平均収益率）を更新
        self.drift = np.mean(self.returns_history)
        
        # ボラティリティ（標準偏差）を更新
        self.volatility = np.std(self.returns_history)
        
        logging.info(f"パラメータ更新 - ドリフト: {self.drift:.4f}, ボラティリティ: {self.volatility:.4f}")
    
    def simulate_future_prices(self, steps: int = 10, num_simulations: int = 1000) -> np.ndarray:
        """
        ランダムウォークによる将来価格シミュレーション
        
        Args:
            steps: 予測ステップ数
            num_simulations: シミュレーション回数
            
        Returns:
            シミュレーション結果の配列 (num_simulations, steps)
        """
        if not self.price_history:
            raise ValueError("価格履歴が空です")
        
        current_price = self.price_history[-1]
        
        # 正規分布に従う乱数を生成
        random_shocks = np.random.normal(0, self.volatility, (num_simulations, steps))
        
        # 各シミュレーションの価格パスを計算
        price_paths = np.zeros((num_simulations, steps + 1))
        price_paths[:, 0] = current_price
        
        for i in range(steps):
            # 幾何ブラウン運動による価格更新
            price_paths[:, i + 1] = price_paths[:, i] * (1 + self.drift + random_shocks[:, i])
        
        return price_paths[:, 1:]  # 初期価格を除く
    
    def get_price_forecast(self, steps: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        価格予測と信頼区間を取得
        
        Args:
            steps: 予測ステップ数
            
        Returns:
            (予測価格, 下位信頼区間, 上位信頼区間)
        """
        simulated_prices = self.simulate_future_prices(steps)
        
        # 統計量を計算
        mean_prices = np.mean(simulated_prices, axis=0)
        
        # 信頼区間を計算
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(simulated_prices, lower_percentile, axis=0)
        upper_bound = np.percentile(simulated_prices, upper_percentile, axis=0)
        
        return mean_prices, lower_bound, upper_bound
    
    def calculate_buy_sell_probability(self, threshold: float = 0.02) -> Tuple[float, float]:
        """
        買い/売りの確率を計算
        
        Args:
            threshold: 判断の閾値（収益率）
            
        Returns:
            (買い確率, 売り確率)
        """
        if len(self.price_history) < 2:
            return 0.0, 0.0
        
        # 1ステップ先の価格をシミュレーション
        future_prices = self.simulate_future_prices(steps=1, num_simulations=10000)
        current_price = self.price_history[-1]
        
        # 収益率を計算
        returns = (future_prices[:, 0] - current_price) / current_price
        
        # 買い/売りの確率を計算
        buy_prob = np.mean(returns > threshold)
        sell_prob = np.mean(returns < -threshold)
        
        return buy_prob, sell_prob
    
    def generate_trading_signal(self, buy_threshold: float = 0.6, 
                              sell_threshold: float = 0.6) -> Optional[str]:
        """
        取引シグナルを生成
        
        Args:
            buy_threshold: 買いシグナルの確率閾値
            sell_threshold: 売りシグナルの確率閾値
            
        Returns:
            'buy', 'sell', または None
        """
        self.update_parameters()
        buy_prob, sell_prob = self.calculate_buy_sell_probability()
        
        logging.info(f"ランダムウォーク確率 - 買い: {buy_prob:.3f}, 売り: {sell_prob:.3f}")
        
        if buy_prob > buy_threshold:
            return "buy"
        elif sell_prob > sell_threshold:
            return "sell"
        else:
            return None
    
    def plot_simulation(self, steps: int = 20, num_paths: int = 100) -> None:
        """
        シミュレーション結果を可視化
        
        Args:
            steps: 予測ステップ数
            num_paths: 表示するパス数
        """
        if not self.price_history:
            print("価格履歴が空です")
            return
        
        # シミュレーション実行
        simulated_prices = self.simulate_future_prices(steps, num_paths)
        current_price = self.price_history[-1]
        
        # 予測と信頼区間を取得
        mean_prices, lower_bound, upper_bound = self.get_price_forecast(steps)
        
        # グラフ作成
        plt.figure(figsize=(12, 8))
        
        # 履歴価格をプロット
        historical_x = list(range(-len(self.price_history), 0))
        plt.plot(historical_x, self.price_history, 'b-', linewidth=2, label='実際の価格')
        
        # シミュレーションパスをプロット
        future_x = list(range(steps))
        for i in range(min(num_paths, 50)):  # 最大50パスまで表示
            plt.plot(future_x, simulated_prices[i], 'lightgray', alpha=0.3, linewidth=0.5)
        
        # 予測価格と信頼区間をプロット
        plt.plot(future_x, mean_prices, 'r-', linewidth=2, label='予測価格')
        plt.fill_between(future_x, lower_bound, upper_bound, alpha=0.2, color='red', 
                        label=f'{self.confidence_level*100}% 信頼区間')
        
        # 現在価格を強調
        plt.axhline(y=current_price, color='green', linestyle='--', alpha=0.7, 
                   label=f'現在価格: {current_price:.2f}')
        
        plt.xlabel('時間')
        plt.ylabel('価格')
        plt.title('ランダムウォークによる株価予測')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_statistics(self) -> dict:
        """
        モデルの統計情報を取得
        
        Returns:
            統計情報の辞書
        """
        return {
            'drift': self.drift,
            'volatility': self.volatility,
            'price_count': len(self.price_history),
            'current_price': self.price_history[-1] if self.price_history else None,
            'window_size': self.window_size,
            'confidence_level': self.confidence_level
        }


def create_sample_data(days: int = 100, initial_price: float = 1000.0) -> List[float]:
    """
    サンプルデータ生成（テスト用）
    
    Args:
        days: データ日数
        initial_price: 初期価格
        
    Returns:
        価格データのリスト
    """
    np.random.seed(42)  # 再現性のため
    
    prices = [initial_price]
    for _ in range(days - 1):
        # 簡単なランダムウォークで価格を生成
        change = np.random.normal(0, 0.02)  # 2%の日次ボラティリティ
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0))  # 負の価格を防ぐ
    
    return prices


if __name__ == "__main__":
    # サンプル実行
    print("ランダムウォークモデルのテスト実行")
    
    # サンプルデータを生成
    sample_prices = create_sample_data(50)
    
    # モデルを初期化
    model = RandomWalkModel(window_size=30)
    
    # 価格データを追加
    for price in sample_prices:
        model.add_price(price)
    
    # 統計情報を表示
    stats = model.get_statistics()
    print(f"統計情報: {stats}")
    
    # 取引シグナルを生成
    signal = model.generate_trading_signal()
    print(f"取引シグナル: {signal}")
    
    # 予測結果を表示
    mean_prices, lower_bound, upper_bound = model.get_price_forecast(steps=10)
    print(f"10日後の予測価格: {mean_prices[-1]:.2f}")
    print(f"信頼区間: [{lower_bound[-1]:.2f}, {upper_bound[-1]:.2f}]")
    
    # 可視化（コメントアウト - 実際の使用時は有効化）
    # model.plot_simulation(steps=20, num_paths=100)