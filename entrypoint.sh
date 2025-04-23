#!/usr/bin/env bash
set -e

# if /models is empty, seed it
# if [ -z "$(ls -A /models/chroma_db)" ]; then
echo "Seeding ChromaDB from remote archive…"
curl -fsSL $CHROMA_URL \
-o /tmp/chroma_db.tar.gz
mkdir -p $CHROMA_DB_DIR
tar xzf /tmp/chroma_db.tar.gz -C $CHROMA_DB_DIR
rm /tmp/chroma_db.tar.gz
curl -fsSL DATA_URL \
-o $FLY_DATA/data.jsonl
# fi

# dropping ollama...
# 1) Start Ollama in the background
# echo "→ Starting Ollama server…"
# ollama serve &

# 2) Wait a moment for Ollama to bind
# sleep 2

# 3) Pull model for emebdding
# ollama pull snowflake-arctic-embed2

# 3) Launch Flask via Gunicorn (or flask run)
echo "→ Starting Flask app…"
exec gunicorn app:app --bind 0.0.0.0:${PORT:-5000}
