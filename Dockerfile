# Python 3.9をベースイメージとして使用
FROM python:3.9-slim

# 作業ディレクトリを設定
WORKDIR /app

# システムパッケージの更新とPoetryのインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        build-essential && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    apt-get purge -y --auto-remove curl build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Poetryのパスを設定
ENV PATH="/root/.local/bin:$PATH"

# Poetry設定（仮想環境を作成しない）
RUN poetry config virtualenvs.create false

# pyproject.tomlとpoetry.lockをコピー
COPY pyproject.toml poetry.lock* ./

# 依存関係をインストール（開発依存関係も含む）
RUN poetry install --no-interaction --no-ansi

# アプリケーションコードをコピー
COPY . .

# ログディレクトリを作成
RUN mkdir -p /app/logs

# 非rootユーザーを作成
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app

# 非rootユーザーに切り替え
USER app

# ヘルスチェック用のスクリプト
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:18080/kabusapi/board?symbol=7203&exchange=1', timeout=5)" || exit 1

# アプリケーションを実行
CMD ["python", "main.py"]