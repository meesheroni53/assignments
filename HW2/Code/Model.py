import subprocess
subprocess.check_call(["pip", "install", "numpy"])
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# File paths 
script_dir = Path(__file__).resolve().parent
rawdata_path = script_dir.parent / "RawData"
train_path = rawdata_path / "train.csv"
test_path = rawdata_path / "test.csv"
output_dir = script_dir.parent / "Output"
output_dir.mkdir(exist_ok=True)
conf_path = output_dir / "confusion_matrix_logistic.jpg"
plot_path = output_dir / "Logistic Regression Plot"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Feature engineering
def bmi_category(bmi):
    if pd.isna(bmi): return 'Unknown'
    elif bmi < 18.5: return 'Underweight'
    elif bmi < 25: return 'Normal'
    elif bmi < 30: return 'Overweight'
    else: return 'Obese'

def categorize_fasting_glucose(glucose):
    if pd.isna(glucose): return 'Unknown'
    elif glucose < 100: return 'Normal'
    elif glucose < 126: return 'Prediabetes'
    else: return 'Diabetes'

def categorize_age(age):
    if age < 18: return 'Child'
    elif age < 35: return 'Young Adult'
    elif age < 50: return 'Middle-Aged'
    elif age < 65: return 'Older Adult'
    else: return 'Senior'

train_df['bmi_category'] = train_df['bmi'].apply(bmi_category)
test_df['bmi_category'] = test_df['bmi'].apply(bmi_category)
train_df['glucose_category'] = train_df['avg_glucose_level'].apply(categorize_fasting_glucose)
test_df['glucose_category'] = test_df['avg_glucose_level'].apply(categorize_fasting_glucose)
train_df['age_category'] = train_df['age'].apply(categorize_age)
test_df['age_category'] = test_df['age'].apply(categorize_age)

# Drop original continuous features
columns_to_drop = ['age', 'bmi', 'avg_glucose_level']
train_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
test_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Drop the stroke (since target variable not input feature) and id column from the training set
y_train = train_df['stroke']
X_train_raw = train_df.drop(columns=['stroke', 'id'])
X_test_raw = test_df.copy()

categorical_cols = X_train_raw.select_dtypes(include=['object', 'category']).columns.tolist()

# Save the processed data (before encoding)
processed_data_dir = script_dir.parent / "ProcessedData"
processed_data_dir.mkdir(exist_ok=True)

train_df.to_csv(processed_data_dir / "train_processed.csv", index=False)
test_df.to_csv(processed_data_dir / "test_processed.csv", index=False)
print(f"Processed train and test CSVs saved to: {processed_data_dir}")

# Check whether there are categories in the test or train that is not in the other and vice versa
print("\n=== Checking for category mismatches between training and test sets ===")
for col in categorical_cols:
    train_values = set(X_train_raw[col].dropna().astype(str).str.strip().str.lower().unique())
    test_values = set(X_test_raw[col].dropna().astype(str).str.strip().str.lower().unique())
    
    missing_in_test = train_values - test_values
    missing_in_train = test_values - train_values

    if missing_in_test or missing_in_train:
        print(f"\n🔸 Column: '{col}'")
        if missing_in_test:
            print(f"  - Values in TRAIN but not in TEST: {sorted(missing_in_test)}")
        if missing_in_train:
            print(f"  - Values in TEST but not in TRAIN: {sorted(missing_in_train)}")
    else:
        print(f"✓ Column '{col}': All categories match.")

# Encode categorical features
encoder = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)],
    remainder='passthrough'
)

X_train = encoder.fit_transform(X_train_raw)
X_test = encoder.transform(X_test_raw)

# Get encoded categorical feature names
encoded_cat_cols = encoder.named_transformers_['cat'].get_feature_names_out(categorical_cols)

# Columns passed through without encoding
remainder_cols = [col for col in X_train_raw.columns if col not in categorical_cols]

# Combine all column names in correct order (encoded categorical + remainder)
encoded_columns = list(encoded_cat_cols) + remainder_cols

# Convert encoded arrays to DataFrames
train_encoded_df = pd.DataFrame(X_train, columns=encoded_columns)
test_encoded_df = pd.DataFrame(X_test, columns=encoded_columns)

# Add stroke and id back for reference
train_encoded_df['stroke'] = y_train.values
test_encoded_df['id'] = test_df['id'].values

# Save to CSV
train_encoded_df.to_csv(processed_data_dir / "train_encoded.csv", index=False)
test_encoded_df.to_csv(processed_data_dir / "test_encoded.csv", index=False)

print(f"Encoded train and test data saved to: {processed_data_dir}")

# Train-test split
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# Logistic Regression with class weighting
model = LogisticRegression(
    penalty='l2',  # Ridge regularization
    solver='liblinear',
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_split, y_train_split)

# Threshold adjustments to reduce false negatives
thresholds = np.arange(0.1, 0.9, 0.02)
best_f1 = 0
best_threshold = 0.5
metrics = []

y_val_proba = model.predict_proba(X_val)[:, 1]
for t in thresholds:
    preds = (y_val_proba >= t).astype(int)
    r = recall_score(y_val, preds)
    p = precision_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    metrics.append((t, r, p, f1))
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

# Plot threshold vs metrics
thresholds_plot, recalls, precisions, f1_scores = zip(*metrics)
plt.figure(figsize=(10,6))
plt.plot(thresholds_plot, recalls, label='Recall')
plt.plot(thresholds_plot, precisions, label='Precision')
plt.plot(thresholds_plot, f1_scores, label='F1 Score')
plt.axvline(x=best_threshold, color='gray', linestyle='--', label=f'Best Threshold = {best_threshold:.2f}')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Metrics vs Classification Threshold')
plt.legend()
plt.grid(True)
plot_path = output_dir / "metrics_vs_threshold.jpg"
plt.savefig(plot_path, dpi=300, format='jpg')
plt.show()
print(f"Saved plot to: {plot_path}")

# Final evaluation
y_val_pred_final = (y_val_proba >= best_threshold).astype(int)
print(f"\n=== Logistic Regression Evaluation (Threshold = {best_threshold:.2f}) ===")
print(classification_report(y_val, y_val_pred_final))
print("Confusion Matrix:")
disp = ConfusionMatrixDisplay(confusion_matrix(y_val, y_val_pred_final), display_labels=["No Stroke", "Stroke"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
conf_path = output_dir / "confusion_matrix_logistic.jpg"
plt.savefig(conf_path, dpi=300, format='jpg')
plt.show()
print(f"Saved confusion matrix to: {conf_path}")

# KNN Model for Comparison
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_split, y_train_split)
y_knn_pred = knn_model.predict(X_val)
print("\n=== KNN Evaluation ===")
print(classification_report(y_val, y_knn_pred))

# Predict on test set using final logistic regression model
y_test_proba = model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= best_threshold).astype(int)
test_df['stroke'] = y_test_pred
test_df['stroke_probability'] = y_test_proba

# Compute final evaluation metrics for logistic regression model
final_accuracy = accuracy_score(y_val, y_val_pred_final)
final_precision = precision_score(y_val, y_val_pred_final)
final_recall = recall_score(y_val, y_val_pred_final)
final_f1 = f1_score(y_val, y_val_pred_final)
final_auc = roc_auc_score(y_val, y_val_proba)

print("\n=== Final Evaluation Metrics (Logistic Regression) ===")
print(f"Best threshold used: {best_threshold:.2f}")
print(f"Accuracy:  {final_accuracy:.4f}")
print(f"Precision: {final_precision:.4f}")
print(f"Recall:    {final_recall:.4f}")
print(f"F1 Score:  {final_f1:.4f}")
print(f"AUC:       {final_auc:.4f}")

# Save predictions
output_dir = script_dir.parent / "Output"
output_dir.mkdir(exist_ok=True)
test_df.to_csv(output_dir / "test_with_predictions.csv", index=False)
print(f"Test predictions saved to: {output_dir / 'test_with_predictions.csv'}")

# Create a DataFrame with only 'id' and 'stroke'
export_df = test_df[['id', 'stroke']]

# Save it as a tab-separated .txt file
output_file_path = output_dir / "predicted_strokes.csv"
export_df.to_csv(output_file_path, index=False)

print(f"Predictions saved to: {output_file_path}")

# Path for the README file, save to the HW2 folder
current_file_path = Path(__file__).resolve()
hw2_folder = current_file_path.parent.parent
readme_path = hw2_folder / "README.md"

readme_content = """\
# Stroke Prediction Model

This project predicts stroke occurrences using Logistic Regression and KNN.

1. Data Collection 
The training and testing data came from a Kaggle public competition called the spring-2025-classification-competition. 

2. Data Preprocessing
I encoded the categorical data so that the categories appeared as 0 or 1 which works more accurately in a model. 
I dropped the stroke and id columns from the preprocessed file. Further, I looked to see whether there were variables
in a category that were part of the training dataset and not part of the testing dataset and vice versa.

3. Feature Engineering
I created the following new features:
- BMI Category - if bmi < 18.5 then Underweight; if bmi <25 then Normal; if bmi < 30 then Overweight else Obese.
- Glucose Category - assuming this is the fasting glucose, if <100 then Normal; if < 126 then Prediabetes else Diabetes.
- Age Category - if age is less than 18 then Child; if < 35 then Young Adult; if < 50 then Middle-Aged; if <65 then Older Adult else Senior.

4. Model Selection
I tried logistic regression and KNN, noting that the logistic regression was far more accurate at predicting stroke, though both did poorly. 

5. Training and Tuning
I performed the following steps to help train/ tune the model:
- Split the data (divided the training data as 80% training and 20% validation)
- In "class_weight = 'balanced', I adjusted the weights of classes based on their frequency in the data. As positive cases of stroke are rare, this piece ensures the model doesn't ignore the minority class (where stroke = 1)
- In "penalty='l2", a penalty term is given to prevent overfitting (ridge regularization)
- Tuned the threshold to improve recall/ F1 score. Specifically, I looked at a variety of thresholds and calculated recall, precision and the F1 score for each. Then the threshold with the highest F1 score was chosen. This can help minimize false negatives which is more important (not missing strokes).

6. Evaluation metrics on training/validation split
Using the best possible threshold (0.78), where if the predicted probability of stroke is greater or equal to 0.78 then it is stroke (1) otherwise, not stroke. With this threshold, I got the following results:
- Accuracy: 0.8930 (89.3% of predictions are correct)
- Precision: 0.1985 (19.85% of predicted stroke were actual strokes so many false positives)
- Recall: 0.5248 (52.5% of true strokes were identified)
- F1: 0.2880 (balances false positives and false negatives)
- AUC: 0.8625 (since near 1, indicates strong ability to identify stroke vs. non-stroke)

As we can see above, since AUC is relatively high, the model performs well. Precision is lower, indicating patients may be predicted as having a stroke incorrectly.

Screenshot of Kaggle leaderboard ranking:

Kaggle Leaderboard Screenshot
![Leaderboard](Output/Leaderboard_Screenshot.png)

Short reflections/ lessons learned:
I learned that it is a balancing act. For example, to reduce the number of false negatives (where stroke is not identified and should have been), it has resulted in more false positives (patients inaccurately being labeled as a stroke risk).

"""

with open(readme_path, "w") as f:
    f.write(readme_content)

print(f"README saved to: {readme_path}")

#  https://stackoverflow.com/questions/52404971/get-a-list-of-categories-of-categorical-variable,
#https://medium.com/@darbinrawal10/data-preprocessing-101-cleaning-and-preparing-your-data-for-ai-models-e4925ca09ae2,
#https://forums.fast.ai/t/what-happens-when-train-and-test-categorical-classes-are-different/19884/4,
#https://markaicode.com/dataset-quality-clean-training-data/