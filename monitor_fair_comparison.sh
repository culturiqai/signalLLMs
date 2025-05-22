#!/bin/bash

# monitor_fair_comparison.sh - Monitor the fair comparison progress

echo "===== FAIR COMPARISON MONITOR ====="
echo ""

# Function to get latest log files
get_latest_logs() {
    SIGNALLLM_LOG=$(ls -t logs/signalllm_core_*.log 2>/dev/null | head -1)
    BASELINE_LOG=$(ls -t logs/baseline_fair_*.log 2>/dev/null | head -1)
}

# Function to extract current metrics
extract_metrics() {
    if [[ -f "$SIGNALLLM_LOG" ]]; then
        SIGNAL_LATEST=$(tail -n 5 "$SIGNALLLM_LOG" | grep -E "step.*ppl.*lr" | tail -1)
        if [[ -n "$SIGNAL_LATEST" ]]; then
            SIGNAL_STEP=$(echo "$SIGNAL_LATEST" | grep -o '[0-9]\+/28760' | cut -d'/' -f1)
            SIGNAL_PPL=$(echo "$SIGNAL_LATEST" | grep -o 'ppl=[0-9\.]*' | cut -d'=' -f2)
            SIGNAL_LR=$(echo "$SIGNAL_LATEST" | grep -o 'lr=[0-9\.e-]*' | cut -d'=' -f2)
        fi
    fi
    
    if [[ -f "$BASELINE_LOG" ]]; then
        BASE_LATEST=$(tail -n 5 "$BASELINE_LOG" | grep -E "step.*ppl.*lr" | tail -1)
        if [[ -n "$BASE_LATEST" ]]; then
            BASE_STEP=$(echo "$BASE_LATEST" | grep -o '[0-9]\+/28760' | cut -d'/' -f1)
            BASE_PPL=$(echo "$BASE_LATEST" | grep -o 'ppl=[0-9\.]*' | cut -d'=' -f2)
            BASE_LR=$(echo "$BASE_LATEST" | grep -o 'lr=[0-9\.e-]*' | cut -d'=' -f2)
        fi
    fi
}

# Function to calculate improvement
calculate_improvement() {
    if [[ -n "$SIGNAL_PPL" && -n "$BASE_PPL" ]]; then
        IMPROVEMENT=$(python3 -c "
signal_ppl = float('$SIGNAL_PPL')
base_ppl = float('$BASE_PPL')
if base_ppl > 0:
    improvement = ((base_ppl - signal_ppl) / base_ppl) * 100
    print(f'{improvement:.1f}%')
else:
    print('N/A')
" 2>/dev/null)
    fi
}

# Main monitoring loop
while true; do
    clear
    echo "===== FAIR COMPARISON PROGRESS ====="
    echo "$(date)"
    echo ""
    
    # Get latest logs
    get_latest_logs
    
    if [[ -z "$SIGNALLLM_LOG" && -z "$BASELINE_LOG" ]]; then
        echo "❌ No training logs found yet..."
        echo "Run: ./run_fair_parallel_comparison.sh"
        sleep 5
        continue
    fi
    
    echo "📊 CURRENT STATUS:"
    echo ""
    
    # Extract and display metrics
    extract_metrics
    
    # SignalLLM status
    if [[ -n "$SIGNAL_STEP" ]]; then
        echo "🔬 SignalLLM (Core Architecture):"
        echo "   Step: $SIGNAL_STEP/28760"
        echo "   Perplexity: $SIGNAL_PPL"
        echo "   Learning Rate: $SIGNAL_LR"
        echo ""
    else
        echo "🔬 SignalLLM: Starting up..."
        echo ""
    fi
    
    # Baseline status  
    if [[ -n "$BASE_STEP" ]]; then
        echo "📈 Baseline Transformer:"
        echo "   Step: $BASE_STEP/28760"
        echo "   Perplexity: $BASE_PPL"
        echo "   Learning Rate: $BASE_LR"
        echo ""
    else
        echo "📈 Baseline: Starting up..."
        echo ""
    fi
    
    # Comparison
    if [[ -n "$SIGNAL_PPL" && -n "$BASE_PPL" ]]; then
        calculate_improvement
        echo "⚡ PERFORMANCE COMPARISON:"
        echo "   SignalLLM PPL: $SIGNAL_PPL"
        echo "   Baseline PPL:  $BASE_PPL"
        if [[ -n "$IMPROVEMENT" ]]; then
            if [[ "$IMPROVEMENT" != "N/A" ]]; then
                if (( $(echo "$IMPROVEMENT" | sed 's/%//') > 0 )); then
                    echo "   🎯 SignalLLM Better: $IMPROVEMENT"
                else
                    echo "   📊 Baseline Better: ${IMPROVEMENT#-}"
                fi
            fi
        fi
        echo ""
        
        # Learning rate check
        if [[ "$SIGNAL_LR" == "$BASE_LR" ]]; then
            echo "✅ Learning rates MATCH: $SIGNAL_LR"
        else
            echo "⚠️  Learning rate MISMATCH:"
            echo "   SignalLLM: $SIGNAL_LR"
            echo "   Baseline:  $BASE_LR"
        fi
    fi
    
    echo ""
    echo "📁 Log files:"
    echo "   SignalLLM: $SIGNALLLM_LOG"
    echo "   Baseline:  $BASELINE_LOG"
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    echo "Refreshing in 10 seconds..."
    
    sleep 10
done 