#! /usr/bin/bash
set -e

LOCAL_DIR="./data/droid_1.0.1"

hf download \
    --repo-type dataset \
    --local-dir "$LOCAL_DIR" \
    lerobot/droid_1.0.1

hf download \
    --repo-type dataset \
    --local-dir "$LOCAL_DIR" \
    griffinlabs/droid_1.0.1-griffin-alpha-overlay
