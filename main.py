import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("BankChurners1.csv")

# ---------------- TARGET VARIABLE ----------------
df['Churn'] = df['Attrition_Flag'].map({
    'Existing Customer': 0,
    'Attrited Customer': 1
})

df = df.drop(columns=[
    'Attrition_Flag',
    'CLIENTNUM',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
])

# ---------------- FEATURE ENGINEERING ----------------
df['Avg_Transaction_Value'] = df['Total_Trans_Amt'] / (df['Total_Trans_Ct'] + 1)

df['Engagement_Score'] = (
    df['Total_Relationship_Count'] +
    df['Contacts_Count_12_mon'] -
    df['Months_Inactive_12_mon']
)

# ---------------- SPLIT ----------------
X = df.drop('Churn', axis=1)
y = df['Churn']

categorical_cols = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ],
    remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- MODEL COMPARISON ----------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

for name, model in models.items():
    pipe = ImbPipeline([
        ('preprocess', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])
    
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))

# ---------------- FINAL MODEL ----------------
final_pipeline = ImbPipeline([
    ('preprocess', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    ))
])

final_pipeline.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
probs = final_pipeline.predict_proba(X_test)[:, 1]

threshold = 0.40
y_pred = (probs >= threshold).astype(int)

print("\nFINAL MODEL PERFORMANCE")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, probs))

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# ---------------- ROC CURVE ----------------
fpr, tpr, _ = roc_curve(y_test, probs)

plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve")
plt.savefig("roc_curve.png")
plt.close()

# ---------------- RISK SEGMENTATION ----------------
risk_df = X_test.copy()
risk_df['Churn_Probability'] = probs

def segment_risk(p):
    if p >= 0.70:
        return "High Risk"
    elif p >= 0.30:
        return "Medium Risk"
    else:
        return "Low Risk"

risk_df['Risk_Group'] = risk_df['Churn_Probability'].apply(segment_risk)

# ---------------- VaR ----------------
risk_df['VaR'] = risk_df['Churn_Probability'] * risk_df['Credit_Limit']

print("\nTop Risk Customers:")
print(risk_df[['Churn_Probability','Risk_Group','VaR']].head())

import joblib

joblib.dump(final_pipeline, "model.pkl")

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "auc": roc_auc_score(y_test, probs)
}

joblib.dump(metrics, "metrics.pkl")