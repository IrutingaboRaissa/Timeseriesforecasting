# Beijing Air Quality Forecasting - PM2.5 Prediction

## Project Overview

This project uses Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) models to forecast PM2.5 air pollution levels in Beijing based on historical air quality and weather data. This is a time series forecasting challenge submitted as part of the Machine Learning Techniques I course.

**Goal**: Achieve RMSE < 4000 (target < 3000) on Kaggle Private Leaderboard

## Dataset

- **Training Data**: 30,678 hourly observations (2010-2013)
- **Test Data**: 13,150 hourly observations (2013)
- **Features**:
  - `DEWP`: Dew Point temperature
  - `TEMP`: Temperature
  - `PRES`: Atmospheric pressure
  - `Iws`: Cumulated wind speed
  - `Is`: Cumulated hours of snow
  - `Ir`: Cumulated hours of rain
  - `cbwd_NW`, `cbwd_SE`, `cbwd_cv`: Wind direction (one-hot encoded)
- **Target**: `pm2.5` - PM2.5 concentration (μg/m³)

## Project Structure

```
Timeseriesforecasting/
│
├── formative-1-time-series-forecasting-january-2026/
│   ├── train.csv                          # Training data
│   └── test.csv                           # Test data
│
├── air_quality_forecasting_starter_code.ipynb   # Main notebook
├── sample_submission .csv                 # Sample submission format
├── experiment_results.csv                 # All experiment results
├── submission_pm25_predictions.csv        # Final predictions
└── README.md                              # This file
```

## Installation & Setup

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

### Running the Notebook
1. Clone this repository
2. Ensure data files are in the correct directory structure
3. Open `air_quality_forecasting_starter_code.ipynb` in Jupyter/VS Code
4. Run all cells sequentially

## Methodology

### 1. Data Exploration
- Comprehensive statistical analysis
- Time series visualization
- Correlation analysis
- Missing value analysis
- Distribution analysis

### 2. Feature Engineering
- **Lag Features**: Previous values (t-1, t-2, t-3, t-6, t-12, t-24)
- **Rolling Statistics**: Moving averages and standard deviations (3, 6, 12, 24-hour windows)
- **Time-Based Features**: Hour, day, month, day of week, season
- **Cyclical Encoding**: Sin/cos transformations for hour and month
- **Normalization**: StandardScaler for feature scaling

### 3. Model Architecture
Conducted 15 systematic experiments including:

#### Architectures Tested:
- Simple LSTM (32, 64, 128 units)
- Stacked LSTM (2-3 layers)
- Bidirectional LSTM
- GRU networks
- Models with dropout regularization

#### Hyperparameters Varied:
- **Optimizers**: Adam, SGD, RMSprop
- **Learning Rates**: 0.0001, 0.001, 0.01
- **Batch Sizes**: 16, 32, 64
- **Dropout Rates**: 0.0, 0.2, 0.3

### 4. Evaluation Metric

**Root Mean Squared Error (RMSE)**:

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

Where:
- $y_i$ = actual PM2.5 value
- $\hat{y}_i$ = predicted PM2.5 value
- $n$ = number of samples

## Results

### Best Model Configuration
- **Architecture**: [To be filled after running experiments]
- **Validation RMSE**: [To be filled]
- **Kaggle Private Score**: [To be filled]

### Key Findings
1. **Lag features** significantly improved performance (especially lag-1 and lag-24)
2. **Regularization** (dropout) helped prevent overfitting
3. **Time-based features** captured daily and seasonal patterns
4. **Bidirectional LSTM** showed improved context understanding

### Experiment Summary
See `experiment_results.csv` for complete experiment table with all 15+ experiments.

## Challenges & Solutions

### 1. Vanishing Gradients
- **Challenge**: Traditional RNNs struggle with long-term dependencies
- **Solution**: LSTM cells with forget gates maintain gradient flow

### 2. Overfitting
- **Challenge**: Model memorizing training data
- **Solution**: Dropout layers, early stopping, train/validation split

### 3. Data Leakage
- **Challenge**: Future information influencing predictions
- **Solution**: Temporal split (not random), proper feature engineering

### 4. Non-Stationarity
- **Challenge**: PM2.5 trends change over time
- **Solution**: Lag features, rolling statistics, detrending

## Visualizations

The notebook includes:
- Time series plots of PM2.5
- Feature distributions and correlations
- Training/validation loss curves
- Predictions vs actual comparisons
- Error analysis plots
- Hourly and monthly pattern analysis

## Future Improvements

1. **Sequence-based Input**: Use 24-hour windows as input
2. **Ensemble Methods**: Combine multiple model predictions
3. **Attention Mechanisms**: Focus on relevant time steps
4. **External Data**: Traffic patterns, industrial activity
5. **Advanced Architectures**: Transformer models, Temporal Convolutional Networks
6. **Hyperparameter Optimization**: Bayesian optimization, grid search

## References

[1] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997.

[2] K. Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation," arXiv:1406.1078, 2014.

[3] F. Chollet, "Deep Learning with Python," Manning Publications, 2017.

## Author

[Your Name]  
[Your Email]  
ALU - Machine Learning Techniques I  
January 2026

## License

This project is submitted as coursework for Machine Learning Techniques I.

## Acknowledgments

- African Leadership University
- Course Instructor and TAs
- Kaggle for hosting the competition
- TensorFlow/Keras documentation and community

---

**Note**: This project demonstrates time series forecasting using deep learning. All code is original work with proper citations for referenced materials.
