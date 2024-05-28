# CauseMe Methods

A collection of advanced causal discovery methods for time series analysis. This repository contains implementations of various techniques in deep learning and machine learning for detecting causal relationships. All methods are compatible with the CauseMe platform, facilitating seamless integration and evaluation.



## Features

- **Deep Learning & Machine Learning Techniques:** Implementations of state-of-the-art methods for causal discovery.
- **Time Series Analysis:** Specialized in detecting causal relationships within time series data.
- **CauseMe Platform Compatibility:** Methods are designed to work smoothly with the CauseMe platform for easy evaluation and benchmarking.

## Available Methods

- **LSTM_method:** Utilizes Long Short-Term Memory (LSTM) networks for capturing temporal dependencies and discovering causal relationships.
- **NN_method:** Applies neural networks for causal inference, leveraging their powerful pattern recognition capabilities.
- **Elasnet_method:** Implements Elastic Net regression for causal discovery, combining the strengths of both LASSO and Ridge regression techniques.


## Template

This repository gives also a fully working template or skeleton
for new [CauseMe](https://causeme.uv.es) python method(s).

```
├── .pre-commit-config.yaml
├── .gitignore
├── tfmmethods
│   ├── __init__.py
│   ├── LSTM_method.py
│   ├── NN_method.py
|   └── elasnet_method.py
├── dev_requirements.txt
├── LICENSE.txt
├── methods.json
├── pyproject.toml
├── README.md
├── requirements.txt
├── setup.cfg
└── tests
    ├── __init__.py
    └── test_method.py
```


