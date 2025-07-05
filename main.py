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

RAKUTEN_API_BASE_URL = "http://localhost:18080/kabusapi"
TOKEN = "<あなたのトークン>"
HEADERS = {
    "Content-Type": "application/json",
    "X-API-KEY": TOKEN
}

# ログ設定
logging.basicConfig(filename='trading.log', level=logging.INFO)

# 株価履歴
price_history = deque(maxlen=100)

# 銘柄選択設定
STOCK_SELECTION_CONFIG = {
    "min_volume": 1000000,      # 最小取引量
    "min_price_change": 0.05,   # 最小価格変動率（5%）
    "max_price": 10000,         # 最大価格
    "min_price": 100,           # 最小価格
    "market_cap_min": 1000000000,  # 最小時価総額（10億円）
    "selection_interval": 3600,  # 銘柄選択間隔（秒）
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

# 銘柄スコア計算
def calculate_stock_score(stock_data):
    score = 0
    try:
        # 取引量スコア（30点満点）
        volume = stock_data.get("Volume", 0)
        if volume > STOCK_SELECTION_CONFIG["min_volume"]:
            volume_score = min(30, (volume / STOCK_SELECTION_CONFIG["min_volume"]) * 10)
            score += volume_score
        
        # 値動きスコア（30点満点）
        price_change_rate = abs(stock_data.get("ChangePreviousClose", 0))
        if price_change_rate > STOCK_SELECTION_CONFIG["min_price_change"]:
            change_score = min(30, (price_change_rate / STOCK_SELECTION_CONFIG["min_price_change"]) * 15)
            score += change_score
        
        # 価格レンジスコア（20点満点）
        current_price = stock_data.get("CurrentPrice", 0)
        if STOCK_SELECTION_CONFIG["min_price"] <= current_price <= STOCK_SELECTION_CONFIG["max_price"]:
            score += 20
        
        # 流動性スコア（20点満点）
        bid_qty = stock_data.get("BidQty", 0)
        ask_qty = stock_data.get("AskQty", 0)
        if bid_qty > 0 and ask_qty > 0:
            liquidity_score = min(20, (bid_qty + ask_qty) / 1000)
            score += liquidity_score
        
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
            if score > 50:  # 閾値スコア
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

# アルゴリズム売買ロジック（移動平均クロス）
def decide_order(prices: list):
    if len(prices) < 25:
        return None
    short_ma = pd.Series(prices[-5:]).mean()
    long_ma = pd.Series(prices[-25:]).mean()
    logging.info(f"短期MA: {short_ma}, 長期MA: {long_ma}")
    if short_ma > long_ma:
        return "buy"
    elif short_ma < long_ma:
        return "sell"
    else:
        return None

# 注文発注
def send_order(symbol: str, side: str, quantity: int):
    order_data = {
        "Password": "<ログインパスワード>",
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
    global current_symbol, last_selection_time
    
    logging.info("株式自動売買システム開始")
    
    while True:
        try:
            # 銘柄選択が必要かチェック
            if should_select_new_stock():
                new_symbol = auto_select_stock()
                if new_symbol != current_symbol:
                    logging.info(f"銘柄切り替え: {current_symbol} -> {new_symbol}")
                    current_symbol = new_symbol
                    # 新しい銘柄の場合は価格履歴をリセット
                    price_history.clear()
                last_selection_time = time.time()
            
            # 現在の銘柄の価格取得
            price = get_price(current_symbol)
            if price:
                price_history.append(price)
                logging.info(f"現在価格 ({current_symbol}): {price}")
                
                # 売買判断
                action = decide_order(list(price_history))
                if action:
                    success = send_order(current_symbol, action, 100)
                    if success:
                        logging.info(f"{action.upper()} 注文成功: {current_symbol} @ {price}")
            else:
                logging.warning(f"価格取得失敗: {current_symbol}")
                
        except Exception as e:
            logging.error(f"メイン処理エラー: {str(e)}")
            
        time.sleep(60)
