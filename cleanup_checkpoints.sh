#!/bin/bash
# Keep only the latest checkpoint to save disk space
CHECKPOINT_DIR="./wikitext103_signalllm_fixed"

# Remove debug directory entirely
rm -rf "$CHECKPOINT_DIR/debug"
echo "Removed debug directory to save disk space"

# Find the latest checkpoint by modification time
LATEST=$(find "$CHECKPOINT_DIR" -name "checkpoint_step_*" -type f | sort -r | head -n 1)

if [ -n "$LATEST" ]; then
  echo "Keeping latest checkpoint: $LATEST"
  # Remove all checkpoints except the latest one
  find "$CHECKPOINT_DIR" -name "checkpoint_step_*" -type f | grep -v "$LATEST" | xargs rm -rf
  echo "Removed old checkpoints to save disk space"
fi
