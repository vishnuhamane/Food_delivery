import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib
import pandas as pd

Path to your CSV file
# Sample historical data for training
df1 = pd.read_csv('data/zomato.csv', encoding='latin1')

# Columns for the generated data
columns = [
    'original price',
    'discount_required',
    'sales_volume',
    'inventory_level',
    'discount_last_week',
    'day_of_week',
    'month',
    'lag_sales',
    'food_items'
]

# Create an empty DataFrame with these columns
df2 = pd.DataFrame(columns=columns)

# Generate and add rows with indexes from 1 to 9951
for i in range(1, 9951):
    df2.loc[i] = [
        np.random.randint(200, 500),  # original price
        np.random.randint(0, 50),  # discount_required
        np.random.randint(1, 500),  # sales_volume
        np.random.randint(0, 500),  # inventory_level
        np.random.randint(0, 50),  # discount_last_week
        np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),  # day_of_week
        np.random.choice(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']),  # month
        np.random.randint(0, 500),  # lag_sales
        np.random.choice(['Apples', 'Bananas', 'Oranges', 'Pears', 'Grapes', 'Pizza', 'Burger'])  # food_items
    ]

# Merge df1 and df2
df = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
print(df)
df.info()

# Encode categorical columns
df_encoded = df.apply(lambda col: pd.Categorical(col).codes if col.dtype == 'object' else col)

# Create 'optimal price' column
df_encoded['optimal price'] = df_encoded['original price'] - df_encoded['discount_required']
print(df_encoded['optimal price'])

# Define features (X) and target (y)
X = df_encoded[['original price', 'sales_volume', 'inventory_level', 'discount_last_week', 'day_of_week', 'month', 'lag_sales']]
y = df_encoded['optimal price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost model
xgb_model3 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model3.fit(X_train, y_train)

# Save the model
joblib.dump(xgb_model3, 'xgb_model3 (1).pkl')