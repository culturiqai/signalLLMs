#!/bin/bash

# run_parallel_comparison.sh - Clean parallel comparison with fixed HRFEvo

echo "Starting Clean Parallel Comparison"
echo "=================================="
echo "SignalLLM: Fixed HRFEvo MPS compatibility"
echo "Baseline: Standard transformer"
echo "Both models: lr=1e-6, seq_len=512, batch_size=8"
echo ""

# Create logs directory
mkdir -p logs

# Set timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Starting SignalLLM (Fixed) at $(date)"
python train_wikitext103.py \
    --seq_length 512 \
    --batch_size 8 \
    --learning_rate 1e-6 \
    --epochs 2 \
    --use_signal_embed \
    --use_wavelet_attention \
    --use_mps \
    --enable_evolution \
    --evolution_population 50 \
    --evolution_generations 10 \
    > ./logs/signalllm_fixed_$TIMESTAMP.log 2>&1 &

SIGNALLLM_PID=$!
echo "SignalLLM started with PID: $SIGNALLLM_PID"

# Wait a moment before starting baseline
sleep 5

echo "Starting Baseline Transformer at $(date)"
python baseline_transformer.py \
    --seq_length 512 \
    --batch_size 8 \
    --learning_rate 1e-6 \
    --epochs 2 \
    > ./logs/baseline_fixed_$TIMESTAMP.log 2>&1 &

BASELINE_PID=$!
echo "Baseline started with PID: $BASELINE_PID"

echo ""
echo "Both models running in parallel!"
echo "Monitor with:"
echo "  SignalLLM: tail -f ./logs/signalllm_fixed_$TIMESTAMP.log"
echo "  Baseline:  tail -f ./logs/baseline_fixed_$TIMESTAMP.log"
echo ""
echo "Stop with: kill $SIGNALLLM_PID $BASELINE_PID"
echo ""
echo "Training started at: $(date)"

# Optional: wait for both to complete
# wait $SIGNALLLM_PID $BASELINE_PID
# echo "Both training runs completed at: $(date)" 