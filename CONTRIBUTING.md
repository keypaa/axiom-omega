# AXIOM-Ω

Latent space intervention system for transformer-based language models.

## Status

✅ All phases complete (1-3)

## Quick Start

```bash
# Install dependencies
pip install torch transformers datasets numpy

# Run benchmark
python phase1_benchmark.py \
    --target_layers "10,31" \
    --lambda_scale 60.0 \
    --magnitude_cap 0.4 \
    --threshold 0.55
```

## Results

- **Refusal bypass**: 100%
- **False positives**: 0%
- **Speed overhead**: 0.6%

## Documentation

See [README.md](./README.md) for full documentation.