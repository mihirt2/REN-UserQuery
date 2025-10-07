#!/bin/bash

N=1200

for ((RUN_ID=0; RUN_ID<=N; RUN_ID++)); do
    echo "Running with RUN_ID=$RUN_ID"
    python run.py "$RUN_ID"
done