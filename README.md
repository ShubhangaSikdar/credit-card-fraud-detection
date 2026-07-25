# credit-card-fraud-detection
## 📌 Overview of Credit card fraud detection:
Financial fraud is costing billions each year. The objective of this project is to create a complete machine learning pipeline to detect fraud transactions in real-time, one of the hardest applications of machine learning due to extremely imbalanced classes (0.17%).

## 📊 Dataset

| Property | Details |
| :--- | :--- |
| Source | Kaggle|
| Size | 284,807 transactions |
| Features | 30 (V1–V28 PCA-transformed + Time + Amount) |
| Target | Class — 0 (Legitimate), 1 (Fraud) |
| Fraud Rate | 0.172% (492 fraud cases) |


## ⚙️ Methodology
1. Exploratory Data Analysis
• Class imbalance visualization (bar chart and pie chart)
• Feature distribution graphs for all 30 features
• PCA feature comparison between fraud and legitimate classes (V1, V3, V4, V10, V12, V14)
• Analysis of Transaction Amount (on log scale) by class (box plot, mean comparison)
• Heatmap of correlation coefficients — top 14 features most correlated with fraud
• Transaction fraud rate analysis based on hours

2. Data Preprocessing
• Removal of duplicate transactions
• 80-20 stratified split of the data
• Time and Amount standardization (other features already normalized during PCA)
• Balancing the class using SMOTE

3. Modelling and Evaluation
• Training of multiple classifiers and comparison between them
• Selection of suitable performance metrics (ROC-AUC, F1, Precision, Recall, Accuracy)
• Analysis of Precision-Recall Curves (preferred over ROC for imbalanced cases)
• Confusion Matrix visualization

## 🛠️ Tech Stack
• Language: Python 3.8+\
• ML: scikit-learn, tensorflow, imbalanced-learn\
• Data: pandas, NumPy\
• Visualisation: matplotlib, seaborn\
• Imbalance Handling: SMOTE (imblearn)

## 🔑 Skills
• Work with highly unbalanced data in a real-world setting\
• Importance of features using PCA\
• Choose the right evaluation metrics for class imbalances (precision, recall, F1-score)\
• Whole ML flow process: EDA + Preprocessing + Modeling + Evaluation

