# training_script.py (Optimized Final Version)

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_selection import RFECV
from imblearn.over_sampling import SMOTE

# 1️⃣ Load and clean dataset
df = pd.read_csv("Loan.csv")
df.dropna(inplace=True)

# 2️⃣ Feature Engineering
df['DebtToIncomeRatio'] = df['MonthlyDebtPayments'] / (df['AnnualIncome'] + 1)

# Selected features for final model (based on analysis)
features = [
    'Age',
    'AnnualIncome',
    'MonthlyDebtPayments',
    'DebtToIncomeRatio',
    'PaymentHistory',
    'UtilityBillsPaymentHistory',
    'PreviousLoanDefaults',
    'CreditScore',
    'BankruptcyHistory',
    'InterestRate',
    'LoanAmount'
]
X = df[features]
y = df['LoanApproved']

# 3️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5️⃣ Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

# 6️⃣ Feature selection using RFECV with RandomForest
rfecv = RFECV(
    estimator=RandomForestClassifier(random_state=42),
    step=1,
    cv=StratifiedKFold(3),
    scoring='f1',
    n_jobs=1
)
rfecv.fit(X_train_bal, y_train_bal)

selected_features_mask = rfecv.support_
selected_features = X.columns[selected_features_mask]
X_train_selected = rfecv.transform(X_train_bal)
X_test_selected = rfecv.transform(X_test_scaled)

print(f"\n✅ Selected Features ({len(selected_features)}): {selected_features.tolist()}")

# 7️⃣ Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    n_jobs=1
)
grid_search.fit(X_train_selected, y_train_bal)
best_model = grid_search.best_estimator_
print("\n📌 Best Random Forest Params:", grid_search.best_params_)

# 8️⃣ Final model evaluation
y_pred = best_model.predict(X_test_selected)
y_prob = best_model.predict_proba(X_test_selected)[:, 1]

print("\n📊 Evaluation Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# 9️⃣ Save all required artifacts
joblib.dump(best_model, "credit_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selected_features.tolist(), "feature_names.pkl")
default_values = df[selected_features].median().to_dict()
joblib.dump(default_values, "default_values.pkl")

print("\n✅ All artifacts saved successfully.")
