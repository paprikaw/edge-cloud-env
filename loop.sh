#!/bin/bash

# 运行指令10次，并检查输出中是否包含 "detection"
for i in {1..100}
do
    echo "Run #$i"
    output=$(python testingmask.py)
    
    # 检查输出中是否包含 "detection"
    if [[ $output == *"detection"* ]]; then
        echo "Detection found in run #$i"
        exit 0
    else
        echo "No detection found in run #$i"
    firandom.uniform(50, 200)

    echo "------------------------"
done
