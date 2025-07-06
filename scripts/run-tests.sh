#!/bin/bash

# テスト実行スクリプト
set -e

echo "🧪 ランダムウォークモデルのテストを実行中..."

# テスト結果ディレクトリを作成
mkdir -p test-results

# Dockerコンテナでテストを実行
docker-compose --profile test build kabu-test
docker-compose --profile test run --rm kabu-test pytest src/test_random_walk.py src/test_trend_indicators.py -v --tb=short --junitxml=/app/test-results/test-results.xml

echo "✅ テスト完了"

# テスト結果を表示
if [ -f "test-results/test-results.xml" ]; then
    echo "📊 テスト結果は test-results/test-results.xml に保存されました"
else
    echo "⚠️  テスト結果ファイルが見つかりません"
fi