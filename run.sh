#!/bin/bash
# Helper script — always runs scripts inside the correct virtual environment
# Usage:
#   ./run.sh train_yolo.py
#   ./run.sh predict.py path/to/image.jpg
#   ./run.sh predict_yolo.py path/to/image.jpg
#   ./run.sh train_classifier.py

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/ai_env/bin/activate"
python3 "$@"
