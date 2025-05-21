#!/usr/bin/env python3
# monitor_training.py - Monitor the progress of WikiText training

import os
import json
import time
import sys
import glob
from datetime import datetime

def get_checkpoints(output_dir):
    """Get a list of checkpoint files in the directory"""
    if not os.path.exists(output_dir):
        return []
    
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*.pt"))
    return sorted(checkpoints)

def get_epoch_from_checkpoint(checkpoint_path):
    """Extract epoch number from checkpoint filename"""
    filename = os.path.basename(checkpoint_path)
    try:
        # Format: checkpoint-{epoch}.pt or checkpoint-{epoch}-{step}.pt
        parts = filename.replace(".pt", "").split("-")
        return int(parts[1])
    except:
        return 0

def get_latest_epoch(output_dir):
    """Get the latest epoch number based on checkpoint files"""
    checkpoints = get_checkpoints(output_dir)
    if not checkpoints:
        return 0
    
    return get_epoch_from_checkpoint(checkpoints[-1])

def has_results(output_dir):
    """Check if results.json exists"""
    return os.path.exists(os.path.join(output_dir, "results.json"))

def get_results(output_dir):
    """Get contents of results.json if it exists"""
    results_path = os.path.join(output_dir, "results.json")
    if not os.path.exists(results_path):
        return None
    
    with open(results_path, 'r') as f:
        return json.load(f)

def get_model_status(output_dir, model_name):
    """Get status of a model's training"""
    if not os.path.exists(output_dir):
        return f"{model_name}: Not started"
    
    # Check for completion
    if has_results(output_dir):
        results = get_results(output_dir)
        return f"{model_name}: Complete - Test perplexity: {results['test_perplexity']:.4f}"
    
    # Check for progress
    latest_epoch = get_latest_epoch(output_dir)
    
    if latest_epoch > 0:
        return f"{model_name}: In progress - Completed {latest_epoch} epochs"
    
    if os.path.exists(os.path.join(output_dir, "tokenizer.json")):
        return f"{model_name}: Initializing - Tokenizer created"
    
    return f"{model_name}: Created directory, not started"

def monitor_progress(standard_dir, signalllm_dir, interval=5):
    """Monitor training progress with periodic updates"""
    try:
        while True:
            os.system('clear')  # Clear screen
            print(f"=== SignalLLM Training Progress - {datetime.now().strftime('%H:%M:%S')} ===\n")
            
            # Check standard model
            print(get_model_status(standard_dir, "Standard Transformer"))
            
            # Check SignalLLM model
            print(get_model_status(signalllm_dir, "SignalLLM (MPS Optimized)"))
            
            # Check for running processes
            print("\nRunning processes:")
            os.system('ps -ef | grep python | grep train_wikitext | grep -v grep')
            
            # Wait for next check
            print(f"\nUpdating every {interval} seconds. Press Ctrl+C to stop.")
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    standard_dir = "./wikitext_standard"
    signalllm_dir = "./wikitext_signalllm"
    
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        # Single status check
        print(get_model_status(standard_dir, "Standard Transformer"))
        print(get_model_status(signalllm_dir, "SignalLLM (MPS Optimized)"))
    else:
        # Continuous monitoring
        interval = 5  # seconds
        if len(sys.argv) > 1:
            try:
                interval = int(sys.argv[1])
            except:
                pass
        
        monitor_progress(standard_dir, signalllm_dir, interval) 