## Push docker image

1. Create Github token
  - GitHub UI: Settings → Developer settings → Personal access tokens
  - write:packages
  - read:packages

2. Login

```
export GHCR_PAT="PASTE_YOUR_TOKEN_HERE"
echo "$GHCR_PAT" | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

3. Create and push image; corss-platform compatible. (update version as needed)
```
docker buildx create --use --name nocouncilbuilder 2>/dev/null || true
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t ghcr.io/tapilab/nocouncil:0.1.0 \
  -t ghcr.io/tapilab/nocouncil:latest \
  --push .
```