# Trained checkpoint format

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

The six sanitized victim checkpoints used for the independent-model evaluation are included under checkpoints/.
