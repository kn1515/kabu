# 株の自動売買ツール（楽天証券API対応）

## 概要
# 楽天証券のMarket Speed II（マーケットスピード II）APIを活用し、株価の取得、売買判断、注文発注を自動で行うPythonベースのツールです。

# 必要条件
# - 楽天証券の口座とMarket Speed IIの契約
# - 楽天証券APIの申請とトークン取得
# - Windows PC（Market Speed II はWindows専用）
# - Python 3.x + 楽天API SDK（Rakuten API Kit）

## 主な機能
# 1. 指定銘柄の株価モニタリング（リアルタイム）
# 2. 売買ロジックによる自動注文判断（移動平均クロス）
# 3. API経由の売買注文実行
# 4. Slack通知・ログ保存

import time
import logging
import requests
import pandas as pd
from collections import deque
import json
from datetime import datetime

import os
from dotenv import load_dotenv
from random_walk import RandomWalkModel
from visualizer import TradingVisualizer
from trend_indicators import TrendSignalAnalyzer

load_dotenv()

RAKUTEN_API_BASE_URL = "http://localhost:18080/kabusapi"
TOKEN = os.getenv("RAKUTEN_API_TOKEN", "<あなたのトークン>")
HEADERS = {
    "Content-Type": "application/json",
    "X-API-KEY": TOKEN
}

# ログ設定
logging.basicConfig(filename='trading.log', level=logging.INFO)

# 株価履歴
price_history = deque(maxlen=100)

# ランダムウォークモデル
random_walk_model = RandomWalkModel(window_size=50)

# 可視化
visualizer = TradingVisualizer()

# トレンド分析
trend_analyzer = TrendSignalAnalyzer()

# 取引履歴
trade_history = []

# 銘柄選択設定
STOCK_SELECTION_CONFIG = {
    "min_volume": 1000000,      # 最小取引量
    "min_price_change": 0.05,   # 最小価格変動率（5%）
    "max_price": 10000,         # 最大価格
    "min_price": 100,           # 最小価格
    "market_cap_min": 1000000000,  # 最小時価総額（10億円）
    "selection_interval": 3600,  # 銘柄選択間隔（秒）
    "trend_signal_weight": 0.4,  # トレンドシグナルの重み
    "min_trend_strength": 30,    # 最小トレンド強度
}

# 現在の対象銘柄
current_symbol = "7203"  # 初期値：トヨタ
last_selection_time = 0

# 株価取得
def get_price(symbol: str, exchange: int = 1):
    res = requests.get(
        f"{RAKUTEN_API_BASE_URL}/board",
        headers=HEADERS,
        params={"symbol": symbol, "exchange": exchange}
    )
    if res.status_code == 200:
        return res.json().get("CurrentPrice")
    else:
        logging.error(f"株価取得失敗: {res.text}")
        return None

# 株価履歴取得（トレンド分析用）
def get_price_history(symbol: str, exchange: int = 1, days: int = 30):
    """
    指定銘柄の過去の株価履歴を取得
    
    Args:
        symbol: 銘柄コード
        exchange: 市場コード
        days: 取得日数
        
    Returns:
        価格履歴のリスト（日次終値）
    """
    try:
        # 楽天証券APIの履歴データエンドポイント（例）
        # 実際のAPIに応じて調整が必要
        res = requests.get(
            f"{RAKUTEN_API_BASE_URL}/primaryexchange",
            headers=HEADERS,
            params={
                "symbol": symbol,
                "exchange": exchange
            }
        )
        
        if res.status_code == 200:
            # ここではサンプルデータを返す（実際の実装では過去データを取得）
            # リアルタイムAPIからは直近のデータのみ取得可能な場合が多いため、
            # 実際の運用では別途履歴データの蓄積が必要
            
            # 仮想的な価格履歴生成（実際の運用では削除）
            current_price = get_price(symbol, exchange)
            if current_price:
                import random
                random.seed(hash(symbol) % 1000)  # 銘柄ごとに一定のシード
                history = []
                price = current_price * 0.95  # 少し前の価格から開始
                
                for i in range(days):
                    change = random.uniform(-0.03, 0.03)  # ±3%の変動
                    price *= (1 + change)
                    history.append(max(price, 1))  # 負の価格を防ぐ
                
                return history
            
        logging.error(f"株価履歴取得失敗: {symbol}")
        return []
        
    except Exception as e:
        logging.error(f"株価履歴取得エラー: {str(e)}")
        return []

# 取引量ランキング取得
def get_volume_ranking():
    try:
        res = requests.get(
            f"{RAKUTEN_API_BASE_URL}/ranking",
            headers=HEADERS,
            params={"type": "1", "exchange": "1"}  # 取引量ランキング
        )
        if res.status_code == 200:
            return res.json().get("ranking", [])
        else:
            logging.error(f"ランキング取得失敗: {res.text}")
            return []
    except Exception as e:
        logging.error(f"ランキング取得エラー: {str(e)}")
        return []

# 値上がり率ランキング取得
def get_price_change_ranking():
    try:
        res = requests.get(
            f"{RAKUTEN_API_BASE_URL}/ranking",
            headers=HEADERS,
            params={"type": "2", "exchange": "1"}  # 値上がり率ランキング
        )
        if res.status_code == 200:
            return res.json().get("ranking", [])
        else:
            logging.error(f"値上がり率ランキング取得失敗: {res.text}")
            return []
    except Exception as e:
        logging.error(f"値上がり率ランキング取得エラー: {str(e)}")
        return []

# 銘柄詳細情報取得
def get_stock_info(symbol: str):
    try:
        res = requests.get(
            f"{RAKUTEN_API_BASE_URL}/symbol",
            headers=HEADERS,
            params={"symbol": symbol, "exchange": "1"}
        )
        if res.status_code == 200:
            return res.json()
        else:
            logging.error(f"銘柄情報取得失敗: {res.text}")
            return None
    except Exception as e:
        logging.error(f"銘柄情報取得エラー: {str(e)}")
        return None

# 銘柄スコア計算（トレンド分析を含む）
def calculate_stock_score(stock_data):
    score = 0
    try:
        symbol = stock_data.get("Symbol", "")
        
        # 基本的なスコア計算
        # 取引量スコア（25点満点）
        volume = stock_data.get("Volume", 0)
        if volume > STOCK_SELECTION_CONFIG["min_volume"]:
            volume_score = min(25, (volume / STOCK_SELECTION_CONFIG["min_volume"]) * 8)
            score += volume_score
        
        # 値動きスコア（25点満点）
        price_change_rate = abs(stock_data.get("ChangePreviousClose", 0))
        if price_change_rate > STOCK_SELECTION_CONFIG["min_price_change"]:
            change_score = min(25, (price_change_rate / STOCK_SELECTION_CONFIG["min_price_change"]) * 12)
            score += change_score
        
        # 価格レンジスコア（15点満点）
        current_price = stock_data.get("CurrentPrice", 0)
        if STOCK_SELECTION_CONFIG["min_price"] <= current_price <= STOCK_SELECTION_CONFIG["max_price"]:
            score += 15
        
        # 流動性スコア（15点満点）
        bid_qty = stock_data.get("BidQty", 0)
        ask_qty = stock_data.get("AskQty", 0)
        if bid_qty > 0 and ask_qty > 0:
            liquidity_score = min(15, (bid_qty + ask_qty) / 1000)
            score += liquidity_score
        
        # トレンド分析スコア（20点満点）
        if symbol:
            try:
                price_history = get_price_history(symbol, days=30)
                if len(price_history) >= 30:
                    trend_result = trend_analyzer.get_comprehensive_signal(price_history)
                    
                    # 買いシグナルの場合はボーナス
                    if trend_result['signal'] in ['buy', 'hold_buy']:
                        trend_score = min(20, trend_result['strength'] / 100 * 20)
                        score += trend_score * STOCK_SELECTION_CONFIG["trend_signal_weight"]
                        
                        logging.info(f"銘柄 {symbol} トレンドスコア: {trend_score:.1f} (シグナル: {trend_result['signal']}, 強度: {trend_result['strength']:.1f})")
                    
                    # 最小トレンド強度チェック
                    if (trend_result['signal'] in ['buy', 'hold_buy'] and 
                        trend_result['strength'] >= STOCK_SELECTION_CONFIG["min_trend_strength"]):
                        score += 10  # 追加ボーナス
                        
            except Exception as trend_error:
                logging.warning(f"銘柄 {symbol} のトレンド分析エラー: {str(trend_error)}")
        
        return score
    except Exception as e:
        logging.error(f"スコア計算エラー: {str(e)}")
        return 0

# 銘柄自動選択
def auto_select_stock():
    try:
        logging.info("銘柄自動選択開始")
        
        # 取引量ランキング取得
        volume_ranking = get_volume_ranking()
        price_change_ranking = get_price_change_ranking()
        
        # 候補銘柄リスト作成
        candidates = {}
        
        # 取引量ランキングから候補追加
        for stock in volume_ranking[:20]:  # 上位20銘柄
            symbol = stock.get("Symbol")
            if symbol:
                candidates[symbol] = stock
        
        # 値上がり率ランキングから候補追加
        for stock in price_change_ranking[:20]:  # 上位20銘柄
            symbol = stock.get("Symbol")
            if symbol:
                if symbol in candidates:
                    # 既存データとマージ
                    candidates[symbol].update(stock)
                else:
                    candidates[symbol] = stock
        
        # 各候補銘柄のスコア計算
        scored_stocks = []
        for symbol, stock_data in candidates.items():
            # 詳細情報取得
            detailed_info = get_stock_info(symbol)
            if detailed_info:
                stock_data.update(detailed_info)
            
            score = calculate_stock_score(stock_data)
            if score > 60:  # 閾値スコアを上げる（トレンド分析を考慮）
                scored_stocks.append({
                    "symbol": symbol,
                    "score": score,
                    "data": stock_data
                })
        
        # スコア順にソート
        scored_stocks.sort(key=lambda x: x["score"], reverse=True)
        
        if scored_stocks:
            best_stock = scored_stocks[0]
            logging.info(f"選択された銘柄: {best_stock['symbol']}, スコア: {best_stock['score']}")
            
            # 銘柄選択ログ
            selection_log = {
                "timestamp": datetime.now().isoformat(),
                "selected_symbol": best_stock['symbol'],
                "score": best_stock['score'],
                "candidates_count": len(candidates),
                "qualified_count": len(scored_stocks)
            }
            logging.info(f"銘柄選択ログ: {json.dumps(selection_log, ensure_ascii=False)}")
            
            return best_stock['symbol']
        else:
            logging.warning("適切な銘柄が見つかりませんでした。デフォルト銘柄を使用します。")
            return "7203"  # デフォルト：トヨタ
            
    except Exception as e:
        logging.error(f"銘柄自動選択エラー: {str(e)}")
        return "7203"  # エラー時はデフォルト銘柄

# 銘柄選択が必要かチェック
def should_select_new_stock():
    global last_selection_time
    current_time = time.time()
    return (current_time - last_selection_time) >= STOCK_SELECTION_CONFIG["selection_interval"]

# アルゴリズム売買ロジック（移動平均クロス + ランダムウォーク）
def decide_order(prices: list):
    if len(prices) < 25:
        return None
    
    # 従来の移動平均クロス
    short_ma = pd.Series(prices[-5:]).mean()
    long_ma = pd.Series(prices[-25:]).mean()
    logging.info(f"短期MA: {short_ma}, 長期MA: {long_ma}")
    
    # 移動平均による基本シグナル
    ma_signal = None
    if short_ma > long_ma:
        ma_signal = "buy"
    elif short_ma < long_ma:
        ma_signal = "sell"
    
    # ランダムウォークモデルによるシグナル
    rw_signal = random_walk_model.generate_trading_signal(
        buy_threshold=0.55,  # 55%以上の確率で買い
        sell_threshold=0.55  # 55%以上の確率で売り
    )
    
    # 両方のシグナルが一致した場合のみ注文を出す
    if ma_signal and rw_signal and ma_signal == rw_signal:
        logging.info(f"シグナル一致: MA={ma_signal}, RW={rw_signal}")
        return ma_signal
    elif ma_signal and rw_signal:
        logging.info(f"シグナル不一致: MA={ma_signal}, RW={rw_signal}")
        return None
    else:
        logging.info(f"シグナル不十分: MA={ma_signal}, RW={rw_signal}")
        return None

# 注文発注
def send_order(symbol: str, side: str, quantity: int):
    order_data = {
        "Password": os.getenv("RAKUTEN_LOGIN_PASSWORD", "<ログインパスワード>"),
        "Symbol": symbol,
        "Exchange": 1,
        "SecurityType": 1,
        "Side": "1" if side == "buy" else "2",
        "CashMargin": 1,
        "DelivType": 0,
        "FundType": "AA",
        "AccountType": 4,
        "Qty": quantity,
        "Price": 0,
        "ExpireDay": 0,
        "FrontOrderType": 10
    }
    res = requests.post(f"{RAKUTEN_API_BASE_URL}/sendorder", headers=HEADERS, json=order_data)
    logging.info(f"注文結果: {res.status_code}, {res.text}")
    return res.status_code == 200

# メイン処理
if __name__ == '__main__':
    logging.info("株式自動売買システム開始")
    
    while True:
        try:
            # 銘柄選択が必要かチェック
            if should_select_new_stock():
                new_symbol = auto_select_stock()
                if new_symbol != current_symbol:
                    logging.info(f"銘柄切り替え: {current_symbol} -> {new_symbol}")
                    current_symbol = new_symbol
                    price_history.clear()
                    # ランダムウォークモデルもリセット
                    random_walk_model = RandomWalkModel(window_size=50)
                last_selection_time = time.time()
            
            # 現在の銘柄の価格取得
            price = get_price(current_symbol)
            if not price:
                logging.warning(f"価格取得失敗: {current_symbol}")
                continue
            
            price_history.append(price)
            # ランダムウォークモデルにも価格を追加
            random_walk_model.add_price(price)
            logging.info(f"現在価格 ({current_symbol}): {price}")
            
            # 売買判断
            action = decide_order(list(price_history))
            if not action:
                continue
            
            success = send_order(current_symbol, action, 100)
            if success:
                logging.info(f"{action.upper()} 注文成功: {current_symbol} @ {price}")
                trade_history.append((len(price_history)-1, action, price))
                
                # 100取引ごとに可視化を実行
                if len(trade_history) % 100 == 0:
                    try:
                        visualizer.plot_random_walk_analysis(random_walk_model, forecast_steps=10)
                        visualizer.plot_performance_metrics(list(price_history), trade_history)
                    except Exception as viz_error:
                        logging.error(f"可視化エラー: {str(viz_error)}")
                
        except Exception as e:
            logging.error(f"メイン処理エラー: {str(e)}")
            
        time.sleep(60)
