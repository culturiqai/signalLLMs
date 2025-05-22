#!/bin/bash

# run_fair_parallel_comparison.sh - Fair parallel comparison without HRFEvo
# Core SignalLLM architecture vs Baseline with identical training configs

echo "Starting FAIR Parallel Comparison"
echo "================================="
echo "SignalLLM: Core architecture only (NO HRFEvo)"
echo "  - Signal embeddings: ENABLED"
echo "  - Wavelet attention: ENABLED" 
echo "  - HRFEvo evolution: DISABLED"
echo "Baseline: Standard transformer"
echo ""
echo "IDENTICAL CONFIGS:"
echo "  - Learning rate: 1e-6"
echo "  - Warmup steps: 100"
echo "  - Sequence length: 512"
echo "  - Batch size: 8"
echo "  - Epochs: 2"
echo ""

# Create logs directory
mkdir -p logs

# Set timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Starting SignalLLM (Core Architecture + Fixed MPS) at $(date)"
python train_wikitext103.py \
    --seq_length 512 \
    --batch_size 8 \
    --learning_rate 1e-6 \
    --epochs 2 \
    --warmup_steps 100 \
    --use_signal_embed \
    --use_wavelet_attention \
    --use_mps \
    --checkpoint_every 500 \
    --output_dir ./outputs/signalllm_core_fixed_$TIMESTAMP \
    --num_workers 0 \
    --log_level INFO \
    > ./logs/signalllm_core_fixed_$TIMESTAMP.log 2>&1 &

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
    --warmup_steps 100 \
    --output_dir ./outputs/baseline_fair_$TIMESTAMP \
    --num_workers 0 \
    > ./logs/baseline_fair_$TIMESTAMP.log 2>&1 &

BASELINE_PID=$!
echo "Baseline started with PID: $BASELINE_PID"

echo ""
echo "===== FAIR COMPARISON RUNNING ====="
echo "Both models use IDENTICAL configs:"
echo "  ✓ Same learning rate: 1e-6"
echo "  ✓ Same warmup: 100 steps"
echo "  ✓ Same architecture size"
echo "  ✓ Same dataset processing"
echo ""
echo "SignalLLM advantages being tested:"
echo "  → Spectral/frequency embeddings"
echo "  → Wavelet-based attention"
echo "  → Multi-resolution processing"
echo ""
echo "Monitor progress:"
echo "  SignalLLM: tail -f ./logs/signalllm_core_fixed_$TIMESTAMP.log"
echo "  Baseline:  tail -f ./logs/baseline_fair_$TIMESTAMP.log"
echo ""
echo "Stop both: kill $SIGNALLLM_PID $BASELINE_PID"
echo ""
echo "Training started at: $(date)"
echo "This will give us the TRUE architectural benefits!" 