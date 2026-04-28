#! /usr/bin/bash
set -e

# AgiBot World
hf download \
    --repo-type dataset \
    --local-dir ./data/agibot-world \
    griffinlabs/AgiBot-World-Beta-arm-and-gripper-only \
    tasks/task_327/ \
    tasks/delta_norm_stats.json

# Galaxea Open-World Dataset
hf download \
    --repo-type dataset \
    --local-dir ./data/galaxea-open-world-dataset \
    griffinlabs/Galaxea-Open-World-Dataset-LeRobot-v3.0 \
    subsets/Adjust_The_Air_Conditioner_Temperature_20250711_006/ \
    arm_and_gripper_only_delta_norm_stats.json
mv \
    ./data/galaxea-open-world-dataset/arm_and_gripper_only_delta_norm_stats.json \
    ./data/galaxea-open-world-dataset/delta_norm_stats.json

# InternData-A1
hf download \
    --repo-type dataset \
    --local-dir ./data/interndata-a1 \
    --revision simplified \
    yixuan-tan/InternData-A1-LeRobot-v3.0-by-embodiment \
    franka-1/articulation_tasks/close_the_electriccooker/ \
    franka-2/basic_tasks/pick_the_object_into_trashcan/ \
    genie1/basic_tasks/take_shelf_items_to_cart_left_arm_part1/ \
    lift2/long_horizon_tasks/clean_the_dirt_with_brown_cloth_left_arm/ \
    split_aloha/pick_and_place_tasks/parallel_pick_and_place/basket_top/ \
    '**/delta_norm_stats.json'
