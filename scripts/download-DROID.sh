#! /usr/bin/bash
set -e

hf download \
    --repo-type dataset \
    --local-dir ./data/droid_1.0.1 \
    lerobot/droid_1.0.1