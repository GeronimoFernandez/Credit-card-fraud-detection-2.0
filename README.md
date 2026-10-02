# Credit Card Fraud Detection 2.0

End-to-end Machine Learning project focused on detecting fraudulent credit card transactions and evolving a traditional ML workflow into a production-oriented application.

The original objective of the project was to build a machine learning model capable of detecting fraudulent transactions in a highly imbalanced dataset.

Version 2.0 extends that work beyond model training by integrating the trained pipeline into an application and progressively introducing production-oriented technologies such as Streamlit, FastAPI, Docker, and MLflow.

---

## 🎯 Project Objective

Credit card fraud detection is a highly imbalanced classification problem: fraudulent transactions represent only a small fraction of all transactions.

The objective of this project is to develop a machine learning system capable of identifying potentially fraudulent transactions while maintaining a reasonable balance between detecting fraud and limiting false positives.

The project also explores how a traditional machine learning workflow can evolve into a more production-oriented system.

### Project Evolution

```text
Traditional ML workflow
        ↓
Data analysis & feature engineering
        ↓
Model experimentation
        ↓
Threshold optimization
        ↓
Reusable model artifact
        ↓
Streamlit application
        ↓
FastAPI inference API
        ↓
Docker containerization
        ↓
MLflow experiment & model tracking


📊 Dataset
The project uses the Credit Card Fraud Detection dataset available on Kaggle.
The cleaned dataset contains approximately 1.3 million transactions, with fraudulent transactions representing approximately 0.58% of the dataset.
The original dataset is not included in this repository.
Dataset source:
https://www.kaggle.com/datasets/kartik2112/fraud-detection


🧠 Machine Learning Approach

The machine learning workflow includes:
Data inspection and cleaning
Exploratory Data Analysis (EDA)
Feature engineering
Stratified train/validation/test split
Categorical feature encoding
XGBoost model training
Feature-set comparison
Decision threshold optimization
Final model training
Error analysis
Model artifact creation
Application integration
🔧 Feature Engineering

Several features were derived from the original transaction data:
age
hour
distance_km

The geographic distance between the customer and merchant was calculated using the Haversine formula.

The final model uses the following eight features:
amt
category
merchant
hour
age
gender
job
distance_km

Categorical variables are transformed using a sparse OneHotEncoder, while numerical variables are passed directly to the model.

A dense one-hot encoding approach initially caused a memory allocation error due to the size and dimensionality of the dataset.

The preprocessing pipeline was therefore redesigned using sparse matrices, allowing the model to be trained without materializing the entire encoded dataset as a dense matrix.

🌲 Model

The final classification model is an XGBoost classifier.

Several feature configurations were evaluated during the modeling process.

The selected pipeline combines:

Transaction information
Customer profile information
Transaction hour
Customer-to-merchant geographic distance

The model also uses scale_pos_weight to account for the strong class imbalance.


🎚️ Decision Threshold Optimization

The default classification threshold of 0.5 was not treated as a fixed decision boundary.

Different thresholds were evaluated on the validation set to analyze the trade-off between precision and recall.

The selected threshold was: 0.75

This threshold produced the best F1-score among the tested validation thresholds.

The test set remained isolated from this optimization process and was only used for the final evaluation.


📈 Final Test Results
The final model was evaluated on a previously unseen test set containing 194,502 transactions.

Metric	Test Result
ROC-AUC	0.9958
PR-AUC	0.8791
Precision	87.39%
Recall	78.15%
F1-Score	82.51%

Confusion Matrix
                 Predicted
              Legitimate  Fraud

Actual
Legitimate       193249    127
Fraud               246    880
The test set contained 1,126 fraudulent transactions.
Because the dataset is highly imbalanced, accuracy is not used as the primary evaluation metric.

🖥️ Streamlit Application

The trained machine learning pipeline is integrated into a Streamlit application.

The application allows a user to enter transaction information and obtain a fraud probability.

The application performs the following steps:

User input
    ↓
Input validation
    ↓
Feature engineering
    ↓
Model preprocessing
    ↓
XGBoost prediction
    ↓
Fraud probability
    ↓
Threshold 0.75
    ↓
Final classification

The application calculates derived features such as:

Age
Transaction hour
Customer-to-merchant distance

The same preprocessing and model pipeline generated during training is reused during inference.

This avoids duplicating the preprocessing logic between the notebook and the application.


📦 Model Artifact
The trained model is stored as a serialized artifact: models/fraud_detection_pipeline.joblib
The artifact contains:
model
threshold
features
category_values
This allows the application to load the complete inference configuration without retraining the model.


🏗️ Production-Oriented Evolution

Version 2.0 is being developed progressively, with each technology addressing a specific engineering concern.

Current — Streamlit

Provides an interactive interface for testing the trained model and simulating an end-to-end inference workflow.

Next — FastAPI

The prediction logic will be exposed through an API, separating the user interface from the model inference service.

Next — Docker

The application and its dependencies will be containerized to improve reproducibility and simplify deployment.

Next — MLflow

Experiment tracking and model management will be introduced to make model experimentation and versioning more reproducible.

🛠️ Technologies
Machine Learning
Python 3.12
Pandas
NumPy
Scikit-learn
XGBoost
Application
Streamlit
Joblib
Planned Production Stack
FastAPI
Docker
MLflow
Data Analysis & Visualization
Matplotlib
Seaborn


📁 Project Structure
Credit-card-fraud-detection/
│
├── data/
│   └── fraud_clean.csv
│
├── notebooks/
│   └── modeling.ipynb
│
├── models/
│   └── fraud_detection_pipeline.joblib
│
├── app/
│   ├── app.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── .gitignore

The structure will evolve as FastAPI, Docker, and MLflow are introduced.

🚀 How to Run

Clone the repository:

git clone https://github.com/GeronimoFernandez/Credit-card-fraud-detection.git

Move into the project directory:

cd Credit-card-fraud-detection

Install the dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app/app.py

The application will be available locally at: http://localhost:8501

🔮 Roadmap
 Exploratory Data Analysis
 Feature Engineering
 Model experimentation
 Feature-set comparison
 Threshold optimization
 Final model training
 Error analysis
 Model artifact creation
 Streamlit application
 FastAPI inference API
 Docker containerization
 MLflow experiment tracking
 Production-oriented architecture
 Deployment


👤 Author
Geronimo Fernandez
Data Scientist | Machine Learning Enthusiast

📄 License
This project is licensed under the MIT License.       

