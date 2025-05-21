# SignalLLM: Research Preview

This repository contains a research preview of SignalLLM with HRFEvo, a novel approach to language modeling that leverages techniques from signal processing to achieve significant computational and parameter efficiency.

## Key Innovations

1. **O(n log n) Attention**: Frequency domain attention with O(n log n) complexity instead of the standard O(n²)
2. **Spectral Embeddings**: Parameter-efficient embeddings that achieve 6x reduction in parameters
3. **HRFEvo Framework**: Evolutionary optimization of harmonic representation functions

## Mathematical Foundations

SignalLLM reconceptualizes language as signals in a frequency domain, enabling several theoretical advantages:

### Frequency Domain Attention

Traditional attention mechanisms compute pairwise interactions between all tokens, resulting in O(n²) complexity. By reformulating attention as a convolution operation in the frequency domain, we leverage the Fast Fourier Transform (FFT) to achieve O(n log n) complexity:

```
Attention(Q, K, V) = softmax(QK^T)V                    # Standard attention: O(n²)
Attention(Q, K, V) = IFFT(FFT(Q) · FFT(K^T) · FFT(V))  # Frequency domain: O(n log n)
```

### Spectral Embeddings

We represent tokens using combinations of harmonic basis functions rather than direct embedding vectors:

```
E(t) = ∑ αᵢ(t) · φᵢ
```

Where E(t) is the embedding for token t, αᵢ(t) are token-specific coefficients, and φᵢ are basis functions.

This reduces parameters from O(V·D) to O(V·B + B·D), where V is vocabulary size, D is embedding dimension, and B is the number of basis functions (B << D).

### Wavelet Multi-Resolution Analysis

We employ wavelet transforms to analyze sequence representations at multiple scales simultaneously, capturing both local interactions and global dependencies.

## Repository Contents

- `signalllm_hrfevo_poc.py`: Single-file proof-of-concept implementation
- `signalllm_demo.py`: Demonstration script for testing and benchmarking
- `SignalLLM_Technical_Paper.tex`: Technical paper describing the approach and results
- `report_assets/`: Visualizations and benchmarking results

## Getting Started

```bash
# Set up environment
python -m venv signalllm_env
source signalllm_env/bin/activate  # On Windows: signalllm_env\Scripts\activate
pip install torch numpy matplotlib tqdm pywavelets scipy scikit-learn tensorboard

# Run the attention complexity demo
python signalllm_demo.py --demo_mode attention --seq_lengths "64,256,512,1024,2048" 

# Run the embedding efficiency demo
python signalllm_demo.py --demo_mode embedding --vocab_sizes "1000,10000,50000,100000" --embed_dim 768 --harmonic_bases 64
```

## Current Status and Limitations

This is an early research preview with the following current limitations:

1. **HRFEvo Implementation**: The evolutionary optimization component has tensor dimension issues that are being resolved. We welcome community contributions to this component.
2. **Performance Validation**: While theoretical advantages are demonstrated, comprehensive benchmarking against standard transformer models on established NLP tasks is still in progress.
3. **Hardware Optimization**: The current implementation is not yet optimized for specific hardware architectures (CUDA, MPS, etc.).

## Benchmarking Results

Initial benchmarking shows the following advantages:

- **Attention Complexity**: Confirmed O(n log n) scaling vs O(n²) for traditional attention
- **Parameter Efficiency**: Consistent 6x reduction in embedding parameters
- **Computational Efficiency**: Significantly lower FLOPS at longer sequence lengths

## Technical Paper

The repository includes a technical paper (`SignalLLM_Technical_Paper.tex`) describing the mathematical foundations, implementation details, and empirical results. To compile the paper:

```bash
pdflatex SignalLLM_Technical_Paper.tex
bibtex SignalLLM_Technical_Paper
pdflatex SignalLLM_Technical_Paper.tex
pdflatex SignalLLM_Technical_Paper.tex
```

## Roadmap

We are working on the following enhancements:

1. Fixing tensor dimension issues in the HRFEvo component
2. Adding comprehensive benchmarks against standard transformers
3. Implementing hardware-specific optimizations
4. Expanding wavelet family implementations
5. Adding visualization tools for spectral representations

## Author

Aditya Tiwari (Jamshedpur, India)

## License

This research preview is provided for academic and research purposes under the MIT License. 