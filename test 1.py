import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# --- Correct File Path ---
file_path = 'D:/urea creatinine app/urea_creatinine_extended.xlsx'

# --- Load data ---
data = pd.read_excel(file_path)

# --- Feature Engineering ---
data['Urea/Creatinine Ratio'] = data['Urea'] / data['Creatinine']

# --- Outlier Removal using IQR ---
def remove_outliers_iqr(df, columns):
    clean_df = df.copy()
    for col in columns:
        Q1 = clean_df[col].quantile(0.25)
        Q3 = clean_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
    return clean_df

columns_to_check = ['Urea', 'Creatinine', 'Urea/Creatinine Ratio']
data_cleaned = remove_outliers_iqr(data, columns_to_check)

# --- Features and Labels ---
X = data_cleaned[['Creatinine']]  # Input: Creatinine
y = data_cleaned['Urea']           # Output: Urea

# --- Feature Transformation: Add Polynomial Features ---
poly = PolynomialFeatures(degree=3, include_bias=False)  # Creatinine, Creatinine^2, Creatinine^3
X_poly = poly.fit_transform(X)

# --- Split into Train and Test ---
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# --- Train Gradient Boosting Regressor ---
model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# --- Predict ---
y_pred = model.predict(X_test)

# --- Evaluate Model ---
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"✅ Model Accuracy (R² Score): {r2*100:.2f}%")
print(f"✅ Root Mean Squared Error: {rmse:.2f}")

# --- Plot Actual vs Predicted ---
plt.scatter(X_test[:, 0], y_test, color='blue', label='Actual')  # X_test[:, 0] = original creatinine
plt.scatter(X_test[:, 0], y_pred, color='red', label='Predicted')
plt.xlabel('Creatinine')
plt.ylabel('Urea')
plt.title('Actual vs Predicted Urea Levels (Boosted & Polynomial Model)')
plt.legend()
plt.grid(True)
plt.show()

# --- Predict for New Input ---
try:
    new_creatinine_value = float(input("Enter Creatinine value to predict Urea: "))
    new_creatinine_poly = poly.transform(np.array([[new_creatinine_value]]))
    predicted_urea = model.predict(new_creatinine_poly)
    print(f"🔮 Predicted Urea: {predicted_urea[0]:.2f} mg/dL")
except ValueError:
    print("⚠️ Please enter a valid number.")
