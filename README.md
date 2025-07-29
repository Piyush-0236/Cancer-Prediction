## 🧬 Cancer Prediction using Machine Learning
This project showcases a supervised machine learning pipeline developed to predict whether a tumor is malignant or benign, using a real-world cancer dataset. Leveraging classification algorithms, we process and analyze diagnostic data to assist early detection and improve treatment outcomes.

## 📌 Problem Statement
Timely and accurate detection of cancer is critical in healthcare. In this project, we use machine learning to classify tumors based on diagnostic measurements. The target is to predict the diagnosis label (Malignant or Benign) from features computed from digitized images of breast mass cell nuclei.

##📊 Dataset
Source: Breast Cancer Wisconsin (Diagnostic) Dataset

Samples: 569

Features: 30 numerical input features (mean, standard error, and worst of various cell measurements)

Target: Diagnosis (M = Malignant, B = Benign)

## 🧰 Technologies Used
Python

Jupyter Notebook

Pandas, NumPy – Data manipulation

Matplotlib, Seaborn – Visualization

Scikit-learn – Preprocessing, Modeling, Evaluation

## 🔍 Workflow
1. Data Preprocessing
Dropping unnecessary ID columns

Converting categorical diagnosis labels to numeric

Checking and handling null or missing values

Feature scaling using StandardScaler

2. Exploratory Data Analysis (EDA)
Correlation heatmaps

Class distribution visualization

Feature importance analysis

3. Model Building
Several classification algorithms were evaluated:

Logistic Regression

Support Vector Machine (SVM)

Decision Tree

Random Forest

K-Nearest Neighbors (KNN)

4. Model Evaluation
Confusion Matrix

Accuracy, Precision, Recall, F1-Score

ROC Curve and AUC Score

## 🧠 Best Performing Model
The Random Forest Classifier gave the highest accuracy and balanced performance on unseen data.

Evaluation metrics showed high sensitivity (recall) for malignant class, critical for medical diagnosis.

## 🔮 Future Enhancements
Hyperparameter tuning with GridSearchCV

Model deployment using Flask or Streamlit

Deep Learning implementation using Keras/TensorFlow

Integration with medical image data for better prediction

## 🤝 Contribution
Pull requests are welcome. For major changes, open an issue first to discuss what you would like to change.
