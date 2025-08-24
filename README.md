# Gold Price Prediction (Deep Learning & Regime Switching)

A reproducible research notebook for forecasting gold prices using multiple sequence modeling architectures (CNN→LSTM, LSTM→CNN, Parallel CNN+LSTM, Hybrid Attention) and a regime‐switching baseline (Gaussian HMM). The workflow automates data acquisition (gold futures + macro factors), technical indicator engineering, feature scaling, model training, evaluation (RMSE/MAE), residual diagnostics, attention visualization, and uncertainty estimation (MC Dropout).

## ✨ Key Features
- Automated market & macro data download (yfinance + FRED via pandas-datareader)
- Rich technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic)
- Macro factors: DXY, Treasury yields, CPI, Oil, Silver, S&P 500, ETF flows
- Sliding window sequence construction with configurable horizon
- Multiple neural architectures (Keras / TensorFlow)
- Hybrid dual-branch attention (technical vs macro signals)
- Regime-switching baseline (Hidden Markov Model) for interpretability
- Metrics table (RMSE / MAE) on original price scale
- Extensive visualization: predictions, residuals, rolling RMSE, attention weights, uncertainty bands

## 📁 Repository Structure
```
├── test_gold_price.ipynb   # Main end-to-end notebook
├── README.md               # Project documentation (this file)
├── LICENSE                 # Open-source license (MIT)
└── .gitignore              # Ignore build, cache & environment artifacts
```

## 🔧 Requirements
Core Python >= 3.9.
Install dependencies (CPU example):
```
pip install -U \
  yfinance pandas numpy pandas-datareader scikit-learn matplotlib seaborn tensorflow hmmlearn
```
Optional / Notes:
- `hmmlearn` (for regime baseline). If unavailable, a lightweight fallback EM routine runs.
- For GPU acceleration install the appropriate `tensorflow` build (e.g. `pip install tensorflow==2.15.*`).

## ▶️ Quick Start
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Gold_Price_Prediction.git
   cd Gold_Price_Prediction
   ```
2. (Optional) Create & activate a virtual environment.
3. Install dependencies (see above).
4. Open `test_gold_price.ipynb` in Jupyter / VS Code / Colab.
5. Run cells top to bottom. Adjust `START_DATE`, `END_DATE`, `DATA_INTERVAL` in the config cell.

## ⚙️ Configuration
Change these in the notebook:
- `START_DATE` / `END_DATE` – historical window
- `DATA_INTERVAL` – e.g. `1d`, `1h`, `1wk`
- `seq_length` – lookback window (default 60)
- `horizon` – forecast step ahead (default 1)
- Early stopping & learning rate reduction callbacks are pre-configured.

## 🧠 Models Included
| Model | Concept |
|-------|---------|
| CNN→LSTM | Local temporal pattern extraction then sequence memory |
| LSTM→CNN | Recurrent feature extraction then convolutional summarization |
| Parallel CNN + LSTM | Dual pathway fusion (convolutional + recurrent) |
| Hybrid Attention | Separate technical & macro branches with temporal attention + gating |
| RegimeSwitch (HMM) | Probabilistic 2-state return regime baseline |

## 📊 Evaluation & Diagnostics
Generated automatically:
- Original scale RMSE / MAE comparison
- Prediction overlay (full + zoomed window)
- Residual scatter, distribution, density, rolling RMSE
- Calibration scatter (Actual vs Predicted)
- Attention weight curves & gate weight distribution
- MC Dropout uncertainty band (approximate predictive uncertainty)

## 🛡️ Disclaimer
This project is for educational & research purposes. It is **not** financial advice. Historical performance does not guarantee future results. Use responsibly.

## 🤝 Contributing
Issues & pull requests welcome. Suggested enhancements:
- Add hyperparameter search
- Multi-step forecasting (horizon > 1)
- Transformer / Temporal Fusion Transformer baseline
- Model packaging as a Python module + CLI

## 📄 License
Released under the MIT License (see `LICENSE`).

## 📬 Contact
Feel free to open an issue or reach out via GitHub if you have questions or suggestions.

---
If you use or extend this work, a citation / link back is appreciated.
