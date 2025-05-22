#!/bin/bash
# run_comparison.sh - Compare SignalLLM vs Baseline Transformer

echo "🔥 CRITICAL LEARNING RATE FIX COMPARISON"
echo "========================================"
echo "Learning Rate: 1e-6 (reduced from 3e-5)"
echo "Log Level: WARNING (reduced from DEBUG)"
echo "Both models: identical parameters"
echo "========================================"

# Activate virtual environment
source signalllm_venv/bin/activate

# Disable tokenizer parallelism
export TOKENIZERS_PARALLELISM=false

# Create output directories
mkdir -p ./wikitext103_baseline
mkdir -p ./wikitext103_signalllm_fixed
mkdir -p ./logs

echo "🔵 Starting BASELINE TRANSFORMER (standard architecture)"
echo "Expected: Slow, steady convergence"
echo "Target: PPL should decrease gradually, not hit <1.4"

python baseline_transformer.py \
  --seq_length 512 \
  --embed_dim 256 \
  --hidden_dim 512 \
  --num_heads 8 \
  --num_layers 4 \
  --batch_size 8 \
  --epochs 1 \
  --learning_rate 1e-6 \
  --warmup_steps 100 \
  --use_mps \
  --output_dir ./wikitext103_baseline \
  --num_workers 0 2>&1 | tee ./logs/baseline_comparison.log

echo ""
echo "⭐ BASELINE COMPLETE! Now starting SignalLLM..."
echo ""
echo "🟠 Starting SIGNALLLM (wavelets + spectral)"
echo "Question: Will it still converge suspiciously fast?"

python train_wikitext103.py \
  --vocab_size 50000 \
  --seq_length 512 \
  --embed_dim 256 \
  --num_heads 8 \
  --num_layers 4 \
  --batch_size 8 \
  --epochs 1 \
  --learning_rate 1e-6 \
  --warmup_steps 100 \
  --use_signal_embed \
  --use_wavelet_attention \
  --use_mps \
  --checkpoint_every 14380 \
  --output_dir ./wikitext103_signalllm_fixed \
  --num_workers 0 \
  --log_level WARNING 2>&1 | tee ./logs/signalllm_comparison.log

echo ""
echo "🎯 COMPARISON COMPLETE!"
echo "========================================"
echo "Results:"
echo "  Baseline: ./wikitext103_baseline/baseline_results.json"
echo "  SignalLLM: ./wikitext103_signalllm_fixed/results.json"
echo ""
echo "📊 Quick analysis:"

# Quick analysis of results
python -c "
import json
import os

print('\\n🔍 PERPLEXITY COMPARISON:')

# Check baseline results
if os.path.exists('./wikitext103_baseline/baseline_results.json'):
    with open('./wikitext103_baseline/baseline_results.json', 'r') as f:
        baseline = json.load(f)
    if baseline['history']:
        final_baseline_ppl = baseline['history'][-1]['val_ppl']
        print(f'📊 Baseline final PPL: {final_baseline_ppl:.2f}')
    else:
        print('❌ Baseline incomplete')
else:
    print('❌ Baseline results not found')

# Check SignalLLM results  
if os.path.exists('./wikitext103_signalllm_fixed/results.json'):
    with open('./wikitext103_signalllm_fixed/results.json', 'r') as f:
        signalllm = json.load(f)
    if 'test_ppl' in signalllm:
        final_signalllm_ppl = signalllm['test_ppl']
        print(f'🌟 SignalLLM final PPL: {final_signalllm_ppl:.2f}')
    else:
        print('❌ SignalLLM incomplete')
else:
    print('❌ SignalLLM results not found')

print('\\n🎯 Analysis:')
print('- If baseline PPL > 100 and SignalLLM PPL < 10: suspicious')
print('- If both have similar convergence: fixed!')
print('- If SignalLLM still converges 10x faster: architecture issue')
"

echo ""
echo "✅ Use this comparison to determine if the issue is:"
echo "   1. Learning rate (fixed) ✅"
echo "   2. Data leakage (should be fixed) ✅"
echo "   3. SignalLLM architecture overfitting (TBD)"
echo "========================================" 