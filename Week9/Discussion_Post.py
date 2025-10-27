import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

# Set random seed 
np.random.seed(12)

# 1) Create two predictor variables (x1 and x2) and generate 3000 random observations
n=3000

# Select 3000 samples between the ages of 18 and 65 as is the working age
age = np.random.randint(18, 66, size=n)        

# Select 3000 samples with income between 20,000 and 150,000
income = np.random.randint(20000, 150000, size=n)  

# 2) Define if-else rule for target variable y (if age > 42 and income > 70000 → y = 1, else → y = 0)
y = np.where((age > 42) & (income > 70000), 1, 0)

# 3) Add random noise to make the problem realistic. 
# For example, you might flip 5–10% of the labels (turn some 0s into 1s, or vice versa), or add random variation to the predictors.
noise = 0.05 
num_flip = int(noise * n) 
flip_y = np.random.choice(n, size=num_flip, replace=False) 

# Flip 0↔1
y_noise = y.copy() 
y_noise[flip_y] = 1 - y_noise[flip_y] 

# Add small random variation to age (+/- years) and income (+/- $2k)
age_noise = age + np.random.normal(0, 5, size=n) 
income_noise = income + np.random.normal(0, 2000, size=n) 

# STEP 2
#  1. Split your data into training and testing sets (e.g., 70/30 split).
df = pd.DataFrame({
    'x1': age_noise,
    'x2': income_noise,
    'y': y_noise
})
X= df[['x1','x2']]
y= df['y']

# Display summary
print(df.head())
print("\nNumber of observations:", len(df))

# Split 70% train / 30% test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

# Display the results
print("Training set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])

# 2. Train a decision tree using your preferred software: DecisionTreeClassifier or DecisionTreeRegressor from scikit-learn
clf = DecisionTreeClassifier(
    criterion='gini',    
    max_depth=None,      
    random_state=12
)

clf.fit(X_train, y_train)

#  Evaluate on test data
y_pred = clf.predict(X_test)

# Evaluate the training data
y_train_pred = clf.predict(X_train)

#3. Visualize your tree 
plt.figure(figsize=(14, 8))
tree.plot_tree(
    clf,
    feature_names=['x1', 'x2'],       # your predictor names
    class_names=['0', '1'],           # target class labels
    filled=True,                      # color the boxes by class
    rounded=True                      # round the corners for readability
)
plt.title("Decision Tree Visualization")
plt.show()

#4. Evaluate performance using your test set. For classification: accuracy, precision, recall, or confusion matrix.
print("Accuracy Score of testing data:", accuracy_score(y_test, y_pred))
print("Accuracy Score of training data:", accuracy_score(y_train, y_train_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# https://www.geeksforgeeks.org/python/how-to-split-the-dataset-with-scikit-learns-train_test_split-function/
# https://www.geeksforgeeks.org/machine-learning/building-and-implementing-decision-tree-classifiers-with-scikit-learn-a-comprehensive-guide/
# https://medium.com/@ms_somanna/guide-to-adding-noise-to-your-data-using-python-and-numpy-c8be815df524
# https://www.geeksforgeeks.org/python/numpy-random-choice-in-python/
# https://www.geeksforgeeks.org/numpy/numpy-where-in-python/