import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer
from train_wikitext103 import WikiText103Dataset

print("Verifying non-overlapping sequences in WikiText103Dataset")

# Create a small dataset for verification
tokenizer = AutoTokenizer.from_pretrained("gpt2")
seq_length = int(os.environ.get('SEQ_LENGTH', 512))

# Create the dataset
dataset = WikiText103Dataset(split='train', seq_length=seq_length, tokenizer=tokenizer, cache_dir='./data')

# Check sample count
total_tokens = len(dataset.all_tokens)
expected_samples = (total_tokens - seq_length - 1) // seq_length
actual_samples = len(dataset.samples)

print(f"Total tokens: {total_tokens}")
print(f"Expected samples with non-overlapping sequences: ~{expected_samples}")
print(f"Actual samples in dataset: {actual_samples}")

# Verify actual implementation by checking stride
if abs(expected_samples - actual_samples) > expected_samples * 0.1:  # Allow 10% margin for edge cases
    print("❌ VERIFICATION FAILED: Dataset is NOT using non-overlapping sequences!")
    print(f"Expected ~{expected_samples} samples, but got {actual_samples}")
    print("Fix the implementation to ensure stride = seq_length")
    sys.exit(1)
else:
    print("✅ VERIFICATION PASSED: Dataset is using non-overlapping sequences")
    print(f"Expected ~{expected_samples} samples, got {actual_samples}")

# Verify consecutive samples have correct stride
if len(dataset.samples) >= 2:
    stride = dataset.samples[1] - dataset.samples[0]
    if stride != seq_length:
        print(f"❌ VERIFICATION FAILED: Incorrect stride! Expected {seq_length}, got {stride}")
        sys.exit(1)
    else:
        print(f"✅ Stride verification passed: {stride} == {seq_length}")

print("Dataset verification complete!")
