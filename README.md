# Stock Market Forecasting with ARIMA

## Project Overview

This project focuses on forecasting stock market values using Time
Series Analysis and ARIMA modeling. The dataset is processed,
visualized, and used to build forecasting models while evaluating
prediction performance using RMSE.

## Features

-   Data preprocessing\
-   Exploratory data analysis\
-   Time series decomposition\
-   ACF & PACF visualization\
-   ARIMA model training\
-   Train--Test split forecasting\
-   RMSE calculation\
-   Prediction visualization

## Technologies Used

-   Python\
-   NumPy\
-   Pandas\
-   Matplotlib\
-   Statsmodels\
-   Scikit-learn

## Installation

``` bash
pip install numpy pandas matplotlib scikit-learn statsmodels pmdarima
```

## ARIMA Model

Autoregressive Integrated Moving Average\
Model order defined as ARIMA(p, d, q).

## Model Training

``` python
from statsmodels.tsa.arima.model import ARIMA
model3 = ARIMA(train, order=(2,2,2))
model_fit3 = model3.fit()
```

## Forecasting

``` python
predictions3 = model_fit3.predict(
    start=len(train),
    end=len(train) + len(test) - 1
)
```

## Model Evaluation

``` python
import numpy as np
from sklearn.metrics import mean_squared_error
rmse2 = np.sqrt(mean_squared_error(test, predictions3))
print("RMSE:", rmse2)
```

## Common Warnings

You may see: - Non-stationary starting autoregressive parameters\
- Non-invertible starting MA parameters

These are normal and automatically handled.

## Author

Md. Miskatul Masabi\
Daffodil International University
