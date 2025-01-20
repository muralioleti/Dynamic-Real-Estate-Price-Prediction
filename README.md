# **_Dynamic Real Estate Price Prediction_**

_This project predicts the prices of houses in Bengaluru based on various features such as area type, availability, location, size, and other related factors. The goal is to create a predictive model using machine learning algorithms that can estimate house prices based on the provided dataset._

# _Table of Contents_

_1._ _Project Overview_

_2. Data Preprocessing_

_3. Feature Engineering_

_4. Modeling_

_5. Model Evaluation_

_6. Hyperparameter Tuning_

_7. Feature Importance_

_8. Residual Analysis_

_9. Results_

_10. Conclusion_

# _Project Overview_
_In this project, we use a dataset of house listings in Bengaluru, India, with the objective of predicting house prices based on various features. The dataset includes columns such as_ _`area_type`, `availability`, `location`, `size`, `bath`, `balcony`, `price`, etc._

_We applied several data preprocessing techniques, followed by the development of machine learning models, including Linear Regression, Random Forest, and XGBoost. The goal is to train a model that can accurately predict the price of a house based on the features provided._
# _Data Preprocessing_
_**1. Loading the Dataset:**_
_The dataset is loaded from a CSV file named `Bengaluru_House_Data.csv`. It contains detailed information about various properties for sale in Bengaluru._

_`data = pd.read_csv('/content/drive/MyDrive/RESUMES/Bengaluru_House_Data.csv')`_

_**2. Handling Missing Data:**_
_The `total_sqft` column, which represents the size of the house in square feet, sometimes contains invalid or missing values. We converted these values into numeric, coercing errors to NaN. Missing values were filled with the mean value of the respective column._

_`data['total_sqft'] = pd.to_numeric(data['total_sqft'], errors='coerce')`_
_`data['total_sqft'].fillna(data['total_sqft'].mean(), inplace=True)`_

_**3. Removing Duplicates:**_
_Duplicate rows were removed to ensure that the model isn't biased by repetitive data._

_`data.drop_duplicates(inplace=True)`_
# _Feature Engineering_
_**1. Price Per Square Foot:**_
_A new feature, `price_per_sqft`, was created by dividing the price by the total square feet of the house._

_`data['price_per_sqft'] = data['price'] / data['total_sqft']`_

_**2. Ratios:**_
_We also created ratios such as `bath_to_sqft_ratio` and `balcony_to_sqft_ratio` to give the model additional features that could be indicative of house quality._

_`data['bath_to_sqft_ratio'] = data['bath'] / data['total_sqft']`_
_`data['balcony_to_sqft_ratio'] = data['balcony'] / data['total_sqft']`_

_**3. Size Conversion:**_
_The `size` column, which contains the number of bedrooms (e.g., '2 BHK', '3 BHK'), was converted into a numeric format by extracting the number of bedrooms._

_`data['size'] = data['size'].str.split(' ').str[0].astype(int)`_

_**4. One-Hot Encoding:**_
_Categorical variables such as `area_type`, `availability`, `location`, and `society` were converted to numeric features using one-hot encoding to make them suitable for model input._

_`data = pd.get_dummies(data, columns=['area_type', 'availability', 'location', 'society'], drop_first=True)`_
# _Modeling_
_We used three different models to predict house prices: **Linear Regression**, **Random Forest Regressor**, and **XGBoost Regressor**._

_**1. Linear Regression:**_
_A baseline model was built using Linear Regression._

_`lr_model = LinearRegression()`_
_`lr_model.fit(X_train, y_train)`_

_**2. Random Forest Regressor:**_
_A more powerful model, **Random Forest Regressor**, was used, leveraging parallel processing._

_`rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)`_
_`rf_model.fit(X_train, y_train)`_

_**3. XGBoost Regressor:**_
_XGBoost, a highly efficient gradient boosting model, was also implemented._

_`xgb_model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=1)`_
_`xgb_model.fit(X_train, y_train)`_
# _Model Evaluation_
_After training the models, we evaluated their performance using Mean Absolute Error (MAE) and R² score._

_**Model Performance**_

- _**Linear Regression MAE:** 42.85_
- _**Random Forest MAE:** 3.37_
- _**XGBoost MAE:** 5.74_
- _**Linear Regression R²:** -12.94_
- _**Random Forest R²:** 0.95_
- _**XGBoost R²:** 0.92_

_The Random Forest Regressor outperformed the other models with a low MAE and high R² score._
# _Hyperparameter Tuning_
_We performed hyperparameter tuning for the XGBoost model to find the best combination of hyperparameters, such as `n_estimators`, `learning_rate`, and `max_depth`. The best parameters were:_

- _**n_estimators:** 100_
- _**learning_rate:** 0.1_
- _**max_depth:** 10_

_These parameters resulted in the lowest Mean Absolute Error._
# _Feature Importance_
_We used Random Forest to evaluate the importance of each feature in predicting house prices. The most important features included `total_sqft`, `size`, and `price_per_sqft`._

_**Feature Importance Bar Plot** illustrates which features had the most significant impact on the model's predictions._
# _Residual Analysis_
_A residual plot was created to assess the prediction errors of the Random Forest model. The residuals are small and scattered around zero, indicating that the model is well-calibrated._

_**Residual Plot** visualizes the residuals from the Random Forest model._

<img width="325" alt="7" src="https://github.com/user-attachments/assets/f68a376d-d733-41bb-94d6-328b49481f35" />

# _Results_
_The **Random Forest Regressor** proved to be the most accurate model for predicting house prices in Bengaluru, with the following evaluation metrics:_

- _**MAE:** 3.37_
- _**MSE:** 1222.44_
- _**RMSE:** 34.96_
# _Conclusion_
_This project demonstrates the process of predicting house prices using machine learning models. By preprocessing the data, creating new features, and training models, we successfully predicted the prices of houses in Bengaluru with high accuracy. The Random Forest model outperformed Linear Regression and XGBoost, showing its effectiveness in this task._
