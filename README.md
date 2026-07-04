# QuarkLM

QuarkLM is a small Python workspace for experimenting with LLM-related topics, especially text processing and embeddings.

## Repository Layout

```text
.
├── LICENSE
├── examples/
├── main.py
├── pyproject.toml
├── README.md
├── notebooks/
│   └── 01-Working-With-Text-Data/
│       ├── 01-text-tokenisation.ipynb
│       └── 02-word-embeddings.ipynb
├── project/
│   └── README.md
└── uv.lock
```

## Notebooks

The notebooks are organized as a small learning path:

1. `01-text-tokenisation.ipynb` covers breaking text into tokens and related preprocessing steps.
2. `02-word-embeddings.ipynb` builds on tokenization with word vector representations.

## Examples

The `examples/` folder contains code that shows how the individual pieces of the project work together in practice.

## Environment

The project currently depends on `gensim`, `numpy`, and `tiktoken`, with Jupyter support available for notebook work.

## Entry Point

`main.py` is a minimal placeholder script at the moment and can be used as the starting point for future command-line or application code.
