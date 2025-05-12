# Airbnb-Price-Prediction

## Introduction:
As a data enthusiast passionate about real-world applications of machine learning, I developed this Airbnb Price Prediction model to help hosts optimize their pricing strategies and travelers make informed booking decisions. Leveraging Python's data science stack (Pandas, NumPy, Scikit-learn) and exploratory data analysis (EDA) techniques, I analyzed 12,805 property listings across 23 features to identify key price drivers.

## Dataset Overview
- 12,805 Airbnb listings
- 23 features including:
  - Property metadata (e.g. bathrooms, bedrooms, beds)
  - Host and location information
  - Amenities and safety rules
  - Price (target variable)

## Features
- **Exploratory Data Analysis (EDA)**: 
  - Analyzed feature distributions, relationships, and correlations.
  - Visualized categorical and numerical variables.
  - Identified trends and patterns affecting price
- **Data Preprocessing**: 
  - Handled missing values and outliers
  - Encoded categorical variables (OneHot & Label Encoding).
  - Scaled numerical features using MinMaxScaler & StandardScalerHandle Missing Values, Outlier Detection and Removal, Convert Categorical Features, Feature Scaling.
- **Model Development**:
  - Built a custom ANN using TensorFlow & Keras.
  - Applied Dropout, BatchNormalization, L2 Regularization.
  - Used log-transformed target variable to reduce skewness
- **Model Optimization**: 
  - Implemented EarlyStopping to prevent overfitting.
  - Performed learning rate tuning, batch size experiments.
  - Evaluated with MAE, RMSE, R² ScoreModel Evaluation, Hyperparameter Tuning, Cross-Validation, Ensemble Methods.
- **Advanced Techniques**: 
  - Compared performance against Gradient Boosting & Random Forest.
  - Used inverse transform (expm1) for accurate real-world evaluation.
  - Prepared final predictions for deployment/readiness.



## Model Performance:
| **Metric**                               | **Value** |
|------------------------------------------|-----------|
| **MAE (Mean Absolute Error)**            | ~0.68     |
| **MSE (Mean Squared Error)**       | ~0.82     |
| **R² Score (Coefficient of Determination)** | ~0.17     |

_NOTE:These results are on the original price scale (after inverse transforming the predictions from log scale)._


## Tech Stack:
- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
- TensorFlow / Keras

## Project Structure:

```
airbnb-price-prediction
├── data/
│   └── airbnb_listings.csv
├── airbnb_price_prediction.ipynb
└── README.md
```


## Conclusion
This project demonstrates the end-to-end application of machine learning and deep learning techniques for a real-world regression problem: predicting Airbnb listing prices. Through rigorous exploratory data analysis, effective feature engineering, and careful model design using an Artificial Neural Network (ANN), we explored the factors that drive pricing decisions.

While the final model achieved reasonable performance, especially after log transformation and regularization, the relatively low R² score suggests that price is influenced by many complex, possibly unobserved factors such as seasonality, location desirability, and host reputation—features that may not be fully captured in the dataset.

Future improvements could involve optimizing model performance, incorporating more datasets, or fine-tuning hyperparameters properly to achieve even better results. This experimentation lays the groundwork for future research and practical implementations in this field.