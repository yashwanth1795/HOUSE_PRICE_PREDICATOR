import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(
    "data/enhanced_house_price_dataset.csv"
)


print("\n========== DATASET ==========")
print(df.head())


print("\nColumns:")
print(df.columns)


print("\nShape:")
print(df.shape)


print("\nDataset Information:")
df.info()


print("\nStatistics:")
print(df.describe())


print("\nMissing Values:")
print(df.isnull().sum())


print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 2. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates()


print("\nShape After Removing Duplicates:")
print(df.shape)


print("\nMissing Values After Removing Duplicates:")
print(df.isnull().sum())


# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

# Numerical columns

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns


for column in numeric_columns:

    df[column] = df[column].fillna(
        df[column].median()
    )


# Categorical columns

categorical_columns = df.select_dtypes(
    include=["str", "object"]
).columns


for column in categorical_columns:

    df[column] = df[column].fillna(
        df[column].mode()[0]
    )


print("\nMissing Values After Filling:")
print(df.isnull().sum())


# ============================================================
# 4. TARGET COLUMN
# ============================================================

target = "Price"


print("\n========== TARGET COLUMN ==========")
print("Target:", target)


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[target]
)

y = df[target]


print("\n========== FEATURES ==========")
print(X.columns.tolist())


print("\n========== TARGET ==========")
print(y.head())


# ============================================================
# 6. REMOVE ID COLUMNS IF PRESENT
# ============================================================

if "ID" in X.columns:

    X = X.drop(
        columns=["ID"]
    )


# ============================================================
# 7. CONVERT CATEGORICAL DATA TO NUMBERS
# ============================================================

X = pd.get_dummies(
    X,
    drop_first=True
)


print("\n========== FEATURES AFTER ENCODING ==========")
print(X.head())


print("\nNumber of features after encoding:")
print(X.shape[1])


# ============================================================
# 8. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\n========== TRAIN TEST SPLIT ==========")

print("Total data:", X.shape)

print("Training data:", X_train.shape)

print("Testing data:", X_test.shape)


# ============================================================
# 9. CREATE LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\n========== MODEL TRAINING ==========")

model.fit(
    X_train,
    y_train
)


print("Linear Regression model trained successfully!")


# ============================================================
# 11. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


print("\nPredictions completed!")


# ============================================================
# 12. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)


r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# 13. DISPLAY PERFORMANCE
# ============================================================

print("\n========================================")
print("         MODEL PERFORMANCE")
print("========================================")


print(
    f"MAE  : {mae:,.2f}"
)


print(
    f"RMSE : {rmse:,.2f}"
)


print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# 14. ACTUAL VS PREDICTED
# ============================================================

comparison = pd.DataFrame({

    "Actual Price": y_test.values,

    "Predicted Price": y_pred

})


print("\n========== ACTUAL VS PREDICTED ==========")

print(
    comparison.head(10)
)


# ============================================================
# 15. SAVE MODEL
# ============================================================

os.makedirs(
    "model",
    exist_ok=True
)


model_data = {

    "model": model,

    "features": X.columns.tolist()

}


joblib.dump(
    model_data,
    "model/house_price_model.pkl"
)


print("\n========================================")
print("       MODEL SAVED SUCCESSFULLY")
print("========================================")


print(
    "Location: model/house_price_model.pkl"
)


print("\n========== PROJECT COMPLETED ==========")