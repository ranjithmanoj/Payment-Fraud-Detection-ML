# 💳 Payment Fraud Detection Using Machine Learning

A Machine Learning-based application that identifies potentially fraudulent financial transactions using trained classification models and a Streamlit interface.

## 📌 Project Overview

Financial fraud is a major challenge in digital payment systems. This project aims to identify suspicious transactions and classify them as fraudulent or legitimate using Machine Learning techniques.

The project includes data extraction, preprocessing, model training, evaluation, and an interactive Streamlit application for fraud prediction.

## 🎯 Objectives

- Detect potentially fraudulent financial transactions.
- Perform data preprocessing and feature engineering.
- Handle class imbalance in transaction data.
- Train and compare multiple Machine Learning algorithms.
- Develop an interactive fraud detection application using Streamlit.

## 📂 Project Structure

```text
Payment-Fraud-Detection-ML/
│
├── app.py
├── Dataset_Extraction.ipynb
├── Payment_Fraud_Detection.ipynb
├── ML_Synthetic_payments.csv
├── Payments_fraud_detection_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

## 📊 Dataset

The project uses a large-scale financial transaction dataset containing approximately 6.5 million original records.

A separate dataset extraction process was performed to prepare the synthetic payment dataset used in the subsequent model development workflow.

Dataset preparation and extraction are documented in `Dataset_Extraction.ipynb`.

## ⚙️ Project Workflow

1. **Data Collection:** Obtaining the original transaction dataset.
2. **Data Extraction:** Preparing the synthetic payment dataset.
3. **Data Preprocessing:** Handling missing values, data cleaning, and feature preparation.
4. **Exploratory Data Analysis:** Understanding transaction patterns and class distribution.
5. **Feature Engineering:** Preparing relevant features for model training.
6. **Class Imbalance Handling:** Applying suitable techniques to address imbalanced classes.
7. **Model Training:** Training and comparing classification algorithms.
8. **Model Evaluation:** Evaluating model performance using classification metrics.
9. **Deployment:** Integrating the trained model into a Streamlit application.

## 🤖 Machine Learning Algorithms

The project explores and compares the following classification algorithms:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Gaussian Naive Bayes
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- Decision Tree
- Support Vector Classifier (SVC)

## 📈 Model Evaluation

The models are evaluated using performance metrics such as:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

**Note:** Add your final selected model's actual accuracy, precision, recall, and F1-score here after confirming the results from your notebook.

## 🖥️ Application

The project includes an interactive web application developed using Streamlit.

The application uses the trained Machine Learning model and scaler to generate fraud predictions.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Class Imbalance | Imbalanced-learn |
| Model Serialization | Joblib / Pickle |
| Web Application | Streamlit |
| Development Environment | Jupyter Notebook |

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Payment-Fraud-Detection-ML.git
```

### 2. Navigate to the project directory

```bash
cd Payment-Fraud-Detection-ML
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

### 5. Open the application

Streamlit will provide a local URL in your terminal. Open it in your browser.

## 📁 Project Files

- `Dataset_Extraction.ipynb` – Dataset extraction and preparation.
- `Payment_Fraud_Detection.ipynb` – Data analysis, model training, and evaluation.
- `app.py` – Streamlit application.
- `Payments_fraud_detection_model.pkl` – Saved trained classification model.
- `scaler.pkl` – Saved feature scaler.
- `ML_Synthetic_payments.csv` – Prepared synthetic payment dataset.

## 👨‍💻 Author

**Ranjith Manojkumar**

Computer Science and Engineering

GitHub: [YOUR_GITHUB_USERNAME](https://github.com/YOUR_USERNAME)

---

*This project was developed as a Machine Learning project focused on identifying potentially fraudulent financial transactions.*
