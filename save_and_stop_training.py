#!/usr/bin/env python3
"""
Script to safely stop SignalLLM training and save current model state
"""

import os
import sys
import json
import time
import signal
import argparse
import torch
from pathlib import Path

def find_running_process(pattern="train_wikitext103.py"):
    """Find process IDs for running training scripts"""
    import subprocess
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    lines = result.stdout.split('\n')
    pids = []
    
    for line in lines:
        if pattern in line and 'python' in line:
            parts = line.split()
            if len(parts) > 1:
                pids.append(parts[1])
    
    return pids

def signal_handler(signum, frame):
    print(f"Received signal {signum}, exiting gracefully...")
    sys.exit(0)

def save_model_state(pid=None, output_dir="./saved_models"):
    """Save the current model state by creating a flagfile for the running process"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get timestamp for unique filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create a flag file that the training process will detect
    flag_path = os.path.join(output_dir, f"save_and_stop_{timestamp}.flag")
    
    with open(flag_path, "w") as f:
        status = {
            "command": "save_and_stop",
            "timestamp": timestamp,
            "requested_by": os.environ.get("USER", "unknown"),
            "pid_to_stop": pid
        }
        json.dump(status, f)
    
    print(f"Created flag file at {flag_path}")
    print("Waiting for training process to detect the flag and save...")
    
    return flag_path, timestamp

def wait_for_completion(flag_path, timeout=300):
    """Wait for the flag file to be processed by the training script"""
    completion_indicator = f"{flag_path}.completed"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if os.path.exists(completion_indicator):
            with open(completion_indicator, "r") as f:
                try:
                    data = json.load(f)
                    print(f"Model saved successfully at: {data.get('model_path', 'unknown')}")
                    print(f"Training statistics: {data.get('stats', {})}")
                    return True
                except json.JSONDecodeError:
                    print("Invalid completion file format")
                    return False
        
        print("Waiting for training process to complete saving (10s)...")
        time.sleep(10)
    
    print(f"Timeout after {timeout} seconds. The process may still be saving.")
    return False

def force_stop_process(pid):
    """Force stop the training process if needed"""
    import subprocess
    try:
        subprocess.run(["kill", pid], check=True)
        print(f"Sent SIGTERM to process {pid}")
        time.sleep(5)
        
        # Check if process still exists
        result = subprocess.run(["ps", "-p", pid], capture_output=True, text=True)
        if pid in result.stdout:
            print(f"Process {pid} still running, sending SIGKILL...")
            subprocess.run(["kill", "-9", pid], check=True)
    except subprocess.CalledProcessError:
        print(f"Process {pid} may have already stopped")

def main():
    parser = argparse.ArgumentParser(description='Safely stop SignalLLM training and save model state')
    parser.add_argument('--output_dir', type=str, default='./saved_models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--pid', type=str, default=None, 
                        help='Process ID to stop (if known, otherwise auto-detected)')
    parser.add_argument('--force', action='store_true',
                        help='Force stop the process if gentle stop fails')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout in seconds to wait for save completion')
    
    args = parser.parse_args()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Find running training processes if PID not provided
    if args.pid is None:
        pids = find_running_process()
        if not pids:
            print("No running training processes found")
            return 1
        elif len(pids) > 1:
            print(f"Multiple training processes found: {', '.join(pids)}")
            pid = input("Enter PID to stop [first one]: ").strip() or pids[0]
        else:
            pid = pids[0]
            print(f"Found training process with PID: {pid}")
    else:
        pid = args.pid
    
    # Create flag file for graceful shutdown
    flag_path, timestamp = save_model_state(pid, args.output_dir)
    
    # Wait for completion
    completed = wait_for_completion(flag_path, args.timeout)
    
    # Force stop if requested and didn't complete normally
    if args.force and not completed:
        print("Forcing process to stop...")
        force_stop_process(pid)
    
    if completed:
        print("Model saved successfully. Training stopped gracefully.")
        return 0
    else:
        print("Model save may not have completed. Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 