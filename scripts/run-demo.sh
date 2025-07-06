#!/bin/bash

# ランダムウォークモデルのデモ実行スクリプト
set -e

echo "🎯 ランダムウォークモデルのデモを実行中..."

# Dockerコンテナでデモを実行
docker-compose --profile test build kabu-test
docker-compose --profile test run --rm kabu-test python src/random_walk.py

echo "📊 可視化デモを実行中..."

# 可視化デモも実行
docker-compose --profile test run --rm kabu-test python src/visualizer.py

echo "✅ デモ完了"