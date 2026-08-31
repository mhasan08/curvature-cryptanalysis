# Trained-checkpoint format

The `extract` workflow expects a PyTorch checkpoint containing at least:

```python
{
    "config": {
        "num_classes": 10,
        "patch_size": 4,
        "d_model": 64,
        "ffn_hidden": 128,
        "depth": 4,
        "heads": 4,
        "activation": "gelu",  # or "silu"
        "dropout": 0.05,
        "class_names": [
            "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
        ],
    },
    "model": model.state_dict(),
}
```

The embedded architecture is a CIFAR-10 transformer with 4×4 patches, four transformer blocks, model dimension 64, four attention heads, and FFN width 128. Older checkpoints whose serialized argument namespace refers to `__main__.train` or `__main__.run_extract` are supported by compatibility shims in the pipeline.

Large checkpoints are intentionally excluded from Git. Store them locally under `checkpoints/`, which is ignored by `.gitignore`, or publish them as a versioned release asset with a checksum.
