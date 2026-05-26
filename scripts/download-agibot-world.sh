#! /usr/bin/bash
set -e

hf download \
    --repo-type dataset \
    --local-dir ./data/agibot-world \
    griffinlabs/AgiBot-World-Beta-no-torso-movement
