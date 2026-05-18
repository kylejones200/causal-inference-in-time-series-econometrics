# Causal Inference in Time Series Econometrics

This project demonstrates causal inference methods for time series econometrics.

## Business context

Econometric data is full of correlations that mean nothing. Ice cream sales and drowning rates move together — because both respond to summer. Interest rates and bond prices move together — because one mechanically determines the other. The challenge is not finding relationships in time series. It is finding ones that reflect a real causal mechanism.

Causal inference in econometrics is a toolkit of methods for making that distinction. Each method makes different assumptions about what you know and does not know about the data-generating process.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Causal inference functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize analysis parameters and output settings.

## Causal Inference Methods

- Granger Causality: Tests if one time series helps predict another
- Difference-in-Differences: Estimates treatment effects using control groups

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).