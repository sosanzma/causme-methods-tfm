# CauseMe Methods

A collection of  causal discovery methods for time series.
Methods implementation is compatible with CauseMe platform.


## Available Methods

- LSTM_method
- NN_method
- Elasnet_method

## Template

This repository gives also a fully working template or skeleton
for new [CauseMe](https://causeme.uv.es) python method(s).

```
├── .pre-commit-config.yaml
├── .gitignore
├── causemesplmthds
│   ├── __init__.py
│   ├── LSTM_method.py
│   └── NN_method.py
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


