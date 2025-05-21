# SignalLLM: Research Preview

This repository contains a research preview of SignalLLM with HRFEvo, a novel approach to language modeling that leverages techniques from signal processing to achieve significant computational and parameter efficiency.

## Key Innovations

1. **O(n log n) Attention**: Frequency domain attention with O(n log n) complexity instead of the standard O(n²)
2. **Spectral Embeddings**: Parameter-efficient embeddings that achieve 6x reduction in parameters
3. **HRFEvo Framework**: Evolutionary optimization of harmonic representation functions

## Repository Contents

- `signalllm_hrfevo_poc.py`: Single-file proof-of-concept implementation
- `signalllm_demo.py`: Demonstration script for testing and benchmarking
- `SignalLLM_Technical_Paper.tex`: Technical paper describing the approach and results
- `report_assets/`: Visualizations and benchmarking results

## Getting Started

```bash
# Set up environment
python -m venv signalllm_env
source signalllm_env/bin/activate
pip install torch numpy matplotlib tqdm pywavelets scipy scikit-learn tensorboard

# Run the attention complexity demo
python signalllm_demo.py --demo_mode attention --seq_lengths "64,256,512,1024,2048" 

# Run the embedding efficiency demo
python signalllm_demo.py --demo_mode embedding --vocab_sizes "1000,10000,50000,100000" --embed_dim 768 --harmonic_bases 64
```

## Technical Paper

The repository includes a technical paper (`SignalLLM_Technical_Paper.tex`) describing the mathematical foundations, implementation details, and empirical results. To compile the paper:

```bash
pdflatex SignalLLM_Technical_Paper.tex
bibtex SignalLLM_Technical_Paper
pdflatex SignalLLM_Technical_Paper.tex
pdflatex SignalLLM_Technical_Paper.tex
```

## Research Status

This is an early research preview. Some components (particularly the evolutionary optimization) have tensor dimension issues that are being resolved. The code demonstrates the core concepts and provides empirical evidence for the theoretical advantages.

## Author

Aditya Tiwari (Jamshedpur, India)

## License

This research preview is provided for academic and research purposes under the MIT License. 