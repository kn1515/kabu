import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from random_walk import RandomWalkModel


class TradingVisualizer:
    """
    株価データとランダムウォーク予測の可視化クラス
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        可視化クラスの初期化
        
        Args:
            figsize: 図のサイズ
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')  # より見やすいスタイル
        
    def plot_price_with_signals(self, prices: List[float], 
                               buy_signals: List[int] = None,
                               sell_signals: List[int] = None,
                               title: str = "株価チャートと売買シグナル") -> None:
        """
        価格チャートと売買シグナルを表示
        
        Args:
            prices: 価格データ
            buy_signals: 買いシグナルのインデックス
            sell_signals: 売りシグナルのインデックス
            title: グラフタイトル
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 価格チャート
        ax.plot(prices, 'b-', linewidth=2, label='株価')
        
        # 買いシグナル
        if buy_signals:
            buy_prices = [prices[i] for i in buy_signals if i < len(prices)]
            ax.scatter(buy_signals, buy_prices, color='green', marker='^', 
                      s=100, label='買いシグナル', zorder=5)
        
        # 売りシグナル
        if sell_signals:
            sell_prices = [prices[i] for i in sell_signals if i < len(prices)]
            ax.scatter(sell_signals, sell_prices, color='red', marker='v', 
                      s=100, label='売りシグナル', zorder=5)
        
        ax.set_xlabel('時間')
        ax.set_ylabel('価格')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_random_walk_analysis(self, model: RandomWalkModel, 
                                 forecast_steps: int = 20) -> None:
        """
        ランダムウォーク分析結果を表示
        
        Args:
            model: ランダムウォークモデル
            forecast_steps: 予測ステップ数
        """
        if not model.price_history:
            print("価格履歴が空です")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. 価格履歴と予測
        historical_prices = model.price_history
        mean_prices, lower_bound, upper_bound = model.get_price_forecast(forecast_steps)
        
        # 履歴価格
        hist_x = list(range(len(historical_prices)))
        ax1.plot(hist_x, historical_prices, 'b-', linewidth=2, label='実際の価格')
        
        # 予測価格
        forecast_x = list(range(len(historical_prices), len(historical_prices) + forecast_steps))
        ax1.plot(forecast_x, mean_prices, 'r-', linewidth=2, label='予測価格')
        ax1.fill_between(forecast_x, lower_bound, upper_bound, alpha=0.3, color='red',
                        label='95% 信頼区間')
        
        ax1.set_title('価格履歴と予測')
        ax1.set_xlabel('時間')
        ax1.set_ylabel('価格')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 収益率の分布
        if model.returns_history:
            returns = np.array(model.returns_history)
            ax2.hist(returns, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(returns.mean(), color='red', linestyle='--', 
                       label=f'平均: {returns.mean():.4f}')
            ax2.axvline(returns.mean() + returns.std(), color='orange', linestyle='--',
                       label=f'±1σ: {returns.std():.4f}')
            ax2.axvline(returns.mean() - returns.std(), color='orange', linestyle='--')
            ax2.set_title('収益率の分布')
            ax2.set_xlabel('収益率')
            ax2.set_ylabel('頻度')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, '収益率データなし', transform=ax2.transAxes,
                    ha='center', va='center', fontsize=12)
        
        # 3. 売買確率の推移
        probabilities = []
        for i in range(max(10, len(historical_prices) - 50), len(historical_prices)):
            if i < len(historical_prices):
                # 一時的にモデルを作成して確率を計算
                temp_model = RandomWalkModel(window_size=30)
                for j in range(max(0, i-29), i+1):
                    temp_model.add_price(historical_prices[j])
                
                buy_prob, sell_prob = temp_model.calculate_buy_sell_probability()
                probabilities.append((buy_prob, sell_prob))
        
        if probabilities:
            prob_x = list(range(len(probabilities)))
            buy_probs = [p[0] for p in probabilities]
            sell_probs = [p[1] for p in probabilities]
            
            ax3.plot(prob_x, buy_probs, 'g-', label='買い確率', linewidth=2)
            ax3.plot(prob_x, sell_probs, 'r-', label='売り確率', linewidth=2)
            ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax3.set_title('売買確率の推移')
            ax3.set_xlabel('時間')
            ax3.set_ylabel('確率')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '確率データなし', transform=ax3.transAxes,
                    ha='center', va='center', fontsize=12)
        
        # 4. シミュレーション結果（複数パス）
        simulated_prices = model.simulate_future_prices(forecast_steps, 50)
        
        for i in range(min(20, len(simulated_prices))):
            ax4.plot(range(forecast_steps), simulated_prices[i], 'lightgray', alpha=0.5, linewidth=1)
        
        ax4.plot(range(forecast_steps), mean_prices, 'r-', linewidth=3, label='平均予測')
        ax4.axhline(y=model.price_history[-1], color='blue', linestyle='--', 
                   label=f'現在価格: {model.price_history[-1]:.2f}')
        ax4.set_title('シミュレーション結果')
        ax4.set_xlabel('時間')
        ax4.set_ylabel('価格')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, prices: List[float], 
                            ma_signals: List[Tuple[int, str]],
                            rw_signals: List[Tuple[int, str]],
                            combined_signals: List[Tuple[int, str]]) -> None:
        """
        移動平均とランダムウォークのシグナル比較
        
        Args:
            prices: 価格データ
            ma_signals: 移動平均シグナル [(インデックス, 'buy'/'sell')]
            rw_signals: ランダムウォークシグナル
            combined_signals: 組み合わせシグナル
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.figsize, sharex=True)
        
        # 1. 移動平均シグナル
        ax1.plot(prices, 'b-', linewidth=2, label='株価')
        
        ma_buys = [i for i, signal in ma_signals if signal == 'buy']
        ma_sells = [i for i, signal in ma_signals if signal == 'sell']
        
        if ma_buys:
            buy_prices = [prices[i] for i in ma_buys if i < len(prices)]
            ax1.scatter(ma_buys, buy_prices, color='green', marker='^', s=60, alpha=0.8)
        if ma_sells:
            sell_prices = [prices[i] for i in ma_sells if i < len(prices)]
            ax1.scatter(ma_sells, sell_prices, color='red', marker='v', s=60, alpha=0.8)
        
        ax1.set_title('移動平均シグナル')
        ax1.set_ylabel('価格')
        ax1.grid(True, alpha=0.3)
        
        # 2. ランダムウォークシグナル
        ax2.plot(prices, 'b-', linewidth=2, label='株価')
        
        rw_buys = [i for i, signal in rw_signals if signal == 'buy']
        rw_sells = [i for i, signal in rw_signals if signal == 'sell']
        
        if rw_buys:
            buy_prices = [prices[i] for i in rw_buys if i < len(prices)]
            ax2.scatter(rw_buys, buy_prices, color='green', marker='^', s=60, alpha=0.8)
        if rw_sells:
            sell_prices = [prices[i] for i in rw_sells if i < len(prices)]
            ax2.scatter(rw_sells, sell_prices, color='red', marker='v', s=60, alpha=0.8)
        
        ax2.set_title('ランダムウォークシグナル')
        ax2.set_ylabel('価格')
        ax2.grid(True, alpha=0.3)
        
        # 3. 組み合わせシグナル
        ax3.plot(prices, 'b-', linewidth=2, label='株価')
        
        combined_buys = [i for i, signal in combined_signals if signal == 'buy']
        combined_sells = [i for i, signal in combined_signals if signal == 'sell']
        
        if combined_buys:
            buy_prices = [prices[i] for i in combined_buys if i < len(prices)]
            ax3.scatter(combined_buys, buy_prices, color='green', marker='^', 
                       s=100, alpha=0.8, label='買い')
        if combined_sells:
            sell_prices = [prices[i] for i in combined_sells if i < len(prices)]
            ax3.scatter(combined_sells, sell_prices, color='red', marker='v', 
                       s=100, alpha=0.8, label='売り')
        
        ax3.set_title('組み合わせシグナル（最終判断）')
        ax3.set_xlabel('時間')
        ax3.set_ylabel('価格')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_performance_metrics(self, prices: List[float], 
                               trades: List[Tuple[int, str, float]]) -> None:
        """
        パフォーマンス指標の可視化
        
        Args:
            prices: 価格データ
            trades: 取引履歴 [(インデックス, 'buy'/'sell', 価格)]
        """
        if not trades:
            print("取引履歴がありません")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. 累積収益
        portfolio_value = [1000000]  # 初期資金100万円
        position = 0
        cash = 1000000
        
        for i, (trade_idx, action, price) in enumerate(trades):
            if action == 'buy' and cash > price * 100:
                position += 100
                cash -= price * 100
            elif action == 'sell' and position > 0:
                position -= 100
                cash += price * 100
            
            # 現在のポートフォリオ価値
            current_value = cash + position * prices[min(trade_idx, len(prices)-1)]
            portfolio_value.append(current_value)
        
        ax1.plot(portfolio_value, 'g-', linewidth=2, label='ポートフォリオ価値')
        ax1.axhline(y=1000000, color='red', linestyle='--', alpha=0.7, label='初期資金')
        ax1.set_title('累積パフォーマンス')
        ax1.set_xlabel('取引回数')
        ax1.set_ylabel('ポートフォリオ価値')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 取引損益分布
        trade_profits = []
        for i in range(0, len(trades)-1, 2):
            if i+1 < len(trades):
                buy_trade = trades[i]
                sell_trade = trades[i+1]
                if buy_trade[1] == 'buy' and sell_trade[1] == 'sell':
                    profit = (sell_trade[2] - buy_trade[2]) * 100
                    trade_profits.append(profit)
        
        if trade_profits:
            ax2.hist(trade_profits, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax2.set_title('取引損益分布')
            ax2.set_xlabel('損益')
            ax2.set_ylabel('頻度')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, '取引データ不足', transform=ax2.transAxes,
                    ha='center', va='center', fontsize=12)
        
        # 3. 勝率とリスクリターン
        if trade_profits:
            win_rate = sum(1 for p in trade_profits if p > 0) / len(trade_profits)
            avg_profit = np.mean(trade_profits)
            profit_std = np.std(trade_profits)
            
            metrics_text = f"""
            取引回数: {len(trade_profits)}
            勝率: {win_rate:.2%}
            平均利益: {avg_profit:.2f}円
            標準偏差: {profit_std:.2f}円
            シャープレシオ: {avg_profit/profit_std:.2f}
            """
            ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes,
                    verticalalignment='top', fontsize=10, family='monospace')
            ax3.set_title('パフォーマンス指標')
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis('off')
        
        # 4. 月次収益率
        monthly_returns = []
        for i in range(len(portfolio_value)-1):
            if i % 30 == 0:  # 30取引を1ヶ月とする
                monthly_return = (portfolio_value[i+1] - portfolio_value[i]) / portfolio_value[i]
                monthly_returns.append(monthly_return)
        
        if monthly_returns:
            ax4.bar(range(len(monthly_returns)), monthly_returns, 
                   color=['green' if r > 0 else 'red' for r in monthly_returns])
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.set_title('月次収益率')
            ax4.set_xlabel('月')
            ax4.set_ylabel('収益率')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '月次データ不足', transform=ax4.transAxes,
                    ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        plt.show()


def demo_visualization():
    """
    可視化機能のデモンストレーション
    """
    from random_walk import create_sample_data
    
    # サンプルデータ生成
    prices = create_sample_data(100, 1000)
    
    # ランダムウォークモデル
    model = RandomWalkModel(window_size=30)
    for price in prices:
        model.add_price(price)
    
    # 可視化
    visualizer = TradingVisualizer()
    
    # 基本的な価格チャート
    buy_signals = [20, 40, 60, 80]
    sell_signals = [30, 50, 70, 90]
    visualizer.plot_price_with_signals(prices, buy_signals, sell_signals)
    
    # ランダムウォーク分析
    visualizer.plot_random_walk_analysis(model, forecast_steps=20)
    
    print("可視化デモ完了")


if __name__ == "__main__":
    demo_visualization()