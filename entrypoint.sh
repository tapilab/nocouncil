#!/usr/bin/env bash
set -e

# Load .env only when running locally (i.e., not on Fly)
if [[ -z "${FLY_APP_NAME:-}" ]]; then
  echo "Local run detected (no FLY_APP_NAME) -> loading .env"
  set -a
  source .env
  set +a
else
  echo "Fly.io detected (FLY_APP_NAME=${FLY_APP_NAME}) -> skipping .env"
fi

CHROMA_DB_DIR="${CHROMA_DB_DIR:-./chroma_db}"
FLY_DATA="${FLY_DATA:-.}"

mkdir -p "$CHROMA_DB_DIR"
mkdir -p "$FLY_DATA"

if [[ -z "$(ls -A "$CHROMA_DB_DIR" 2>/dev/null)" ]]; then
  echo "Seeding ChromaDB from remote archive into $CHROMA_DB_DIR ..."
  curl -fsSL "$CHROMA_URL" -o /tmp/chroma_db.tar.gz
  tar xzf /tmp/chroma_db.tar.gz -C "$CHROMA_DB_DIR"
  rm /tmp/chroma_db.tar.gz
else
  echo "Using existing ChromaDB at $CHROMA_DB_DIR"
fi

if [[ ! -f "$FLY_DATA/data.jsonl" ]]; then
  echo "Seeding council metadata into $FLY_DATA/data.jsonl ..."
  curl -fsSL "$DATA_URL" -o "$FLY_DATA/data.jsonl"
else
  echo "Using existing council metadata at $FLY_DATA/data.jsonl"
fi

# 3) Launch Flask via Gunicorn (or flask run)
echo "→ Starting Flask app…"
exec gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --timeout 180
