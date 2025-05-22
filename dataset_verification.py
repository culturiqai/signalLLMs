import torch
import logging
import sys
from train_wikitext103 import WikiText103Dataset
from transformers import AutoTokenizer
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def analyze_sample_indices(dataset):
    """Analyze the sample indices to ensure proper non-overlapping sequences"""
    if len(dataset.samples) < 2:
        logger.warning("Not enough samples to analyze indices")
        return True
    
    # Check the first few samples
    indices = dataset.samples[:10]
    diffs = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
    
    # All diffs should equal the sequence length
    expected_diff = dataset.seq_length
    correct = all(diff == expected_diff for diff in diffs)
    
    logger.info(f"Sample index differences: {diffs}")
    logger.info(f"Expected difference: {expected_diff}")
    logger.info(f"Samples have correct stride: {correct}")
    
    return correct

def verify_positional_non_overlap(dataset):
    """Verify that consecutive samples use different token positions (the correct check)"""
    if len(dataset.samples) < 2:
        logger.warning("Not enough samples to analyze sequences")
        return True
    
    # Check positional overlap for consecutive samples
    overlaps_detected = 0
    for i in range(min(5, len(dataset.samples)-1)):
        start_pos_1 = dataset.samples[i]
        end_pos_1 = start_pos_1 + dataset.seq_length
        
        start_pos_2 = dataset.samples[i+1]
        end_pos_2 = start_pos_2 + dataset.seq_length
        
        # Check if position ranges overlap
        positional_overlap = not (end_pos_1 <= start_pos_2 or end_pos_2 <= start_pos_1)
        
        if positional_overlap:
            overlaps_detected += 1
            logger.error(f"❌ Sample {i} positions [{start_pos_1}:{end_pos_1}] OVERLAPS with Sample {i+1} positions [{start_pos_2}:{end_pos_2}]")
        else:
            logger.info(f"✅ Sample {i} positions [{start_pos_1}:{end_pos_1}] and Sample {i+1} positions [{start_pos_2}:{end_pos_2}] are non-overlapping")
    
    return overlaps_detected == 0

def verify_token_diversity(dataset):
    """Optional: Check that consecutive samples have reasonable token diversity (for information only)"""
    if len(dataset.samples) < 2:
        return True
    
    diversities = []
    for i in range(min(3, len(dataset.samples)-1)):
        x1, _ = dataset[i]
        x2, _ = dataset[i+1]
        
        # Convert to sets for unique token comparison
        unique_tokens_1 = set(x1.tolist())
        unique_tokens_2 = set(x2.tolist())
        
        # Calculate Jaccard similarity (intersection over union)
        intersection = len(unique_tokens_1 & unique_tokens_2)
        union = len(unique_tokens_1 | unique_tokens_2)
        similarity = intersection / union if union > 0 else 0
        
        diversities.append(similarity)
        logger.info(f"Sample {i} and {i+1} unique token similarity: {similarity:.2%}")
    
    avg_similarity = sum(diversities)/len(diversities) if diversities else 0
    logger.info(f"Average unique token similarity: {avg_similarity:.2%}")
    logger.info(f"(Token similarity is normal for natural language - this is just for information)")
    
    return True

# Verify with multiple sequence lengths
all_tests_passed = True
for seq_length in [128, 256, 512]:
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing with sequence length: {seq_length}")
    logger.info(f"{'='*60}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Create train dataset
    try:
        train_dataset = WikiText103Dataset(
            split='train', 
            seq_length=seq_length, 
            tokenizer=tokenizer
        )
        
        # Calculate expected sample counts
        total_tokens = len(train_dataset.all_tokens)
        expected_samples = (total_tokens - seq_length - 1) // seq_length
        actual_samples = len(train_dataset)
        
        logger.info(f"Total tokens: {total_tokens}")
        logger.info(f"Expected samples: ~{expected_samples}")
        logger.info(f"Actual samples: {actual_samples}")
        
        # Verify sample indices
        indices_ok = analyze_sample_indices(train_dataset)
        positions_ok = verify_positional_non_overlap(train_dataset)
        verify_token_diversity(train_dataset)
        
        if not indices_ok or not positions_ok:
            logger.error("❌ Verification failed for sequence length %d", seq_length)
            all_tests_passed = False
        else:
            logger.info("✅ All checks passed for sequence length %d", seq_length)
        
    except Exception as e:
        logger.error(f"Error testing sequence length {seq_length}: {str(e)}")
        all_tests_passed = False

if all_tests_passed:
    logger.info("\n🎉 SUCCESS: All dataset verification tests passed!")
    logger.info("The non-overlapping sequence fix is working correctly.")
else:
    logger.error("\n❌ FAILURE: Dataset verification failed!")
    logger.error("The non-overlapping sequence implementation needs further fixes.")

# Exit with appropriate status
sys.exit(0 if all_tests_passed else 1)
