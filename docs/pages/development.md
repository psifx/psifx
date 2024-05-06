# Development

## Build & Push

```bash
export PSIFX_VERSION="X.Y.Z" # Major.Minor.Patch
export HF_TOKEN="write-your-hf-token-here"

docker buildx build \
   --build-arg PSIFX_VERSION=$PSIFX_VERSION \
   --build-arg HF_TOKEN=$HF_TOKEN \
   --tag "psifx:$PSIFX_VERSION" \
   --push .
```
