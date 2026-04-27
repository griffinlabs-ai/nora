#! /usr/bin/bash
set -e

hf download \
    --repo-type dataset \
    --local-dir ./data/egodex \
    --revision simplified \
    yixuan-tan/EgoDex-LeRobot-v3.0
