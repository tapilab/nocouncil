#!/usr/bin/env bash
set -e

# if /models is empty, seed it
# if [ -z "$(ls -A /models/chroma_db)" ]; then
echo "Seeding ChromaDB from remote archive…"
curl -fsSL https://tulane.box.com/shared/static/zluxr3m4i9myyct7sot1ynhxbfe0q9jc.gz \
-o /tmp/chroma_db.tar.gz
mkdir -p /models/chroma_db
tar xzf /tmp/chroma_db.tar.gz -C /models/chroma_db
rm /tmp/chroma_db.tar.gz
curl -fsSL https://tulane.box.com/shared/static/so39zx2l1tuxtlg23vze2cs6xezx3c93.jsonl \
-o /models/data.jsonl
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
