# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = r"C:\Users\kateg\OneDrive\Desktop\KUMC\BIOS 835\Final Project\healthcare-dataset-stroke-data.csv"
df = pd.read_csv(file_path)

#print(df.head())

# Drop NA values (remove rows with missing BMI)
df = df.dropna(subset=['bmi'])
# Drop 'id' column (not useful for modeling)
df = df.drop(columns=['id'])
#print(df.head())

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define X and y
X = df.drop(columns=['stroke'])
y = df['stroke']
# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Logistic regression model
log_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_model.fit(X_train, y_train)

# Random Forest and XGBoost models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions
log_preds = log_model.predict_proba(X_test)[:, 1]
rf_preds = rf_model.predict_proba(X_test)[:, 1]
xgb_preds = xgb_model.predict_proba(X_test)[:, 1]

# AUC Scores

print("Logistic Regression AUC:", roc_auc_score(y_test, log_preds))
print("Random Forest AUC:", roc_auc_score(y_test, rf_preds))
print("XGBoost AUC:", roc_auc_score(y_test, xgb_preds))

# ROC Curve
fpr_log, tpr_log, _ = roc_curve(y_test, log_preds)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_preds)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_preds)

plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label='Logistic Regression')
plt.plot(fpr_rf, tpr_rf, label='Random Forest')
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost')
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()
