# Comparing Deep Learning and Statistical Models for Stock Price Prediction

This project explores stock price prediction using time series analysis, comparing the performance of Long Short-Term Memory (LSTM), a deep learning model, with AutoRegressive Integrated Moving Average (ARIMA), a statistical model. The primary focus is on forecasting HP Inc.'s future stock prices based on historical data. The objective is to highlight the strengths and weaknesses of both approaches in predicting stock trends.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [ARIMA Model](#arima-model)
  - [LSTM Model](#lstm-model)
- [Evaluation Metrics](#evaluation-metrics)
- [Results and Discussion](#results-and-discussion)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [License](#license)

## Introduction

Predicting stock prices is a challenging task due to the inherent complexity and volatility of financial markets. Traditional statistical methods like ARIMA have been widely used for time series forecasting. However, with the advent of deep learning, models like LSTM have shown promise in capturing long-term dependencies and non-linear patterns in data.

In this project, we compare these two models—ARIMA and LSTM—to determine which approach offers better performance in predicting the stock prices of HP Inc. The analysis provides insights into the predictive power and limitations of each model.

## Dataset

The dataset used in this project is sourced from HP Inc.'s historical stock prices and is available in the `Dataset` folder of this repository under the file name `HPQ.csv`.

**Dataset Features:**
- Date
- Open
- High
- Low
- Close
- Volume

## Methodology

### Data Preprocessing

1. **Loading Data:**
   The dataset is read using Pandas and checked for missing values and anomalies.

2. **Data Cleaning:**
   - Handled missing values.
   - Converted date columns to datetime objects.

3. **Feature Selection:**
   Focused on the 'Close' price for prediction as it represents the final price at which the stock is traded on a given day.

4. **Normalization:**
   - For LSTM, data was normalized using MinMaxScaler to improve model performance.
   - ARIMA, being a statistical model, required stationarity checks and differencing to stabilize variance.

### ARIMA Model

1. **Stationarity Check:**
   - Used Augmented Dickey-Fuller (ADF) test to check for stationarity.
   - Applied differencing to achieve stationarity if needed.

2. **Model Fitting:**
   - Determined optimal p, d, q parameters using ACF and PACF plots.
   - Fitted the ARIMA model on the training data.

3. **Forecasting:**
   - Generated forecasts and compared them against the actual values.

### LSTM Model

1. **Data Preparation:**
   - Created sequences of past stock prices to predict the next price.
   - Split the data into training and testing sets.

2. **Model Architecture:**
   - Used an LSTM layer followed by Dense layers.
   - Configured with appropriate loss functions and optimizers.

3. **Training:**
   - Trained the model on the training set and validated on the testing set.

4. **Prediction:**
   - Made predictions and inverse transformed the results to the original scale.

## Evaluation Metrics

Both models were evaluated using the following metrics:

- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in a set of predictions.
- **Root Mean Squared Error (RMSE)**: Penalizes larger errors more significantly than MAE.
- **R-squared (R2 Score)**: Indicates the proportion of variance in the dependent variable predictable from the independent variables.

## Results and Discussion

- **ARIMA Model:**
  - Strengths: Simplicity, interpretability, and good performance on linear data.
  - Weaknesses: Struggles with non-linear patterns and requires stationarity.

- **LSTM Model:**
  - Strengths: Captures complex, non-linear relationships and long-term dependencies.
  - Weaknesses: Requires more computational resources and time to train.

**Performance Comparison:**
- The LSTM model showed superior performance in capturing non-linear trends and provided better predictive accuracy compared to the ARIMA model.
- However, ARIMA's simplicity and faster computation make it suitable for quick, linear trend analysis.

## Conclusion

This project demonstrates the comparative strengths and weaknesses of LSTM and ARIMA models in stock price prediction. While LSTM outperforms ARIMA in terms of predictive accuracy, ARIMA's simplicity and interpretability make it a viable option for certain applications. The choice of model depends on the specific requirements of the forecasting task.

## How to Run

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/mohiuddin-khan-shiam/Comparing-Deep-Learning-and-Statistical-Models-for-Stock-Price-Prediction.git
   cd Comparing-Deep-Learning-and-Statistical-Models-for-Stock-Price-Prediction
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook:**
   Open `LSTM_vs_Arima.ipynb` in Jupyter Notebook or Google Colab and run all cells.

## Dependencies

- Python 3.x
- numpy
- pandas
- matplotlib
- statsmodels
- scikit-learn
- tensorflow/keras

## License

This project is licensed under the MIT License.

---

For any questions or feedback, feel free to reach out via [GitHub Issues](https://github.com/mohiuddin-khan-shiam/Comparing-Deep-Learning-and-Statistical-Models-for-Stock-Price-Prediction/issues).

