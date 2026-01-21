#!/usr/bin/env bash
# if running locally
# Load .env only when running locally (i.e., not on Fly)
if [[ -z "${FLY_APP_NAME:-}" ]]; then
	set -e
  echo "Local run detected (no FLY_APP_NAME) → loading .env"
  set -a
  source .env
  set +a
else
  echo "Fly.io detected (FLY_APP_NAME=${FLY_APP_NAME}) → skipping .env"
fi
# set -a           # auto-export all variables
# source .env      # load .env file
# set +a
# if /models is empty, seed it
# if [ -z "$(ls -A /models/chroma_db)" ]; then
echo "Seeding ChromaDB from remote archive…"
curl -fsSL $CHROMA_URL \
-o /tmp/chroma_db.tar.gz
mkdir -p $CHROMA_DB_DIR
tar xzf /tmp/chroma_db.tar.gz -C $CHROMA_DB_DIR
rm /tmp/chroma_db.tar.gz
curl -fsSL $DATA_URL \
-o $FLY_DATA/data.jsonl


# 3) Launch Flask via Gunicorn (or flask run)
echo "→ Starting Flask app…"
exec gunicorn app:app --bind 0.0.0.0:${PORT:-5000}
