# customer-churn-prediction
Customer churn prediction using a Keras-based ANN with preprocessing, feature scaling, and hyperparameter tuning via GridSearchCV. Includes model evaluation with early stopping and TensorBoard, and a Streamlit app for deployment.
# 🧠 Customer Churn Prediction

This project builds an end-to-end deep learning pipeline to predict customer churn using an Artificial Neural Network (ANN). It includes preprocessing, feature engineering, hyperparameter tuning, model evaluation with TensorBoard, and deployment using Streamlit.

---

## 🚀 Problem Statement

Customer churn is a major problem in industries like telecom, where retaining existing customers is more cost-effective than acquiring new ones. This project aims to:

- Identify patterns leading to customer churn.
- Predict churn using historical data.
- Provide actionable insights for business decision-making.

---

## 🧭 How It Works

<p align="center">
  <img src="https://github.com/soham-kar/customer-churn-prediction/blob/main/workflow_diagram.png" width="1400 "/>
</p>

## 📊 Project Workflow

## 🔄 Project Workflow

### 1️⃣ Data Exploration & Visualization (EDA)
- 📊 **Univariate & Bivariate Analysis**  
  → Explored features like gender, tenure, and contract type  
- 🔥 **Churn Distribution & Correlation Analysis**  
  → Used heatmaps to identify feature relationships and potential churn indicators

---

### 2️⃣ Preprocessing
- 🧩 **Categorical Encoding**  
  → Applied Label Encoding & One-Hot Encoding where appropriate  
- 📏 **Feature Scaling**  
  → Used `StandardScaler` to normalize numerical inputs  
- 💾 **Saving Pipelines**  
  → Persisted encoders & scalers using `pickle` for reuse in app deployment

---

### 3️⃣ Modeling
- 🔍 **Hyperparameter Tuning**  
  → Optimized with `GridSearchCV` to find best batch size, epochs, and optimizer  
- 🧠 **Model Architecture**  
  → Built an Artificial Neural Network (ANN) using TensorFlow/Keras  
- 📉 **Evaluation & Monitoring**  
  → Integrated `EarlyStopping` and **TensorBoard** for training visualization and diagnostics

### 4️⃣ 📈 Model Training Metrics (TensorBoard)

Below are the training metrics visualized using **TensorBoard** during model training:

<p align="center">
  <img src="https://github.com/soham-kar/customer-churn-prediction/blob/main/tensorboard_metrics.png" width="800"/>
</p>

### 📊 Key Observations

- ✅ **Validation Accuracy** steadily improved, reaching **~86.9%**
- 📉 **Validation Loss** consistently decreased, indicating effective learning
- 🧠 **Training vs. Validation curves** show **no overfitting**, thanks to `EarlyStopping`

---

### 5️⃣ 🌐 Deployment

- 🚀 Deployed via a **Streamlit app** for real-time, interactive churn prediction  
- 🧩 Integrated with saved preprocessing pipeline and trained ANN model

---

## 🛠️ Tech Stack

- **Languages & Libraries:** Python, Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, TensorFlow, Keras
- **Hyperparameter Tuning:** GridSearchCV
- **Visualization & Monitoring:** TensorBoard
- **Deployment:** Streamlit
- **Platform:** Google Colab + GitHub

---

## 📂 Project Structure
customer-churn-prediction/
├── models/ # 🔐 Contains saved encoders, scalers, and trained ANN model (.pkl/.h5 files)
│
├── Churn_Model_Training.ipynb # 📒 Jupyter notebook with full pipeline:
│ # - EDA
│ # - Preprocessing
│ # - GridSearchCV tuning
│ # - ANN model training and evaluation
│
├── app.py # 🌐 Streamlit script for deploying the model as a web app
│ # - Loads saved model and encoders
│ # - Takes user input and predicts churn
│
├── requirements.txt # 📦 Python package dependencies for the project
│
├── runtime.txt # ⚙️ Runtime environment file for Streamlit Cloud (e.g., python-3.10)
│
├── README.md # 📘 Project documentation — what you're reading now!
│
├── tensorboard_metrics.png # 📊 Training metrics screenshot from TensorBoard
│
└── workflow_diagram.png # 🧭 Visual diagram of the ML pipeline (EDA → Model → Deployment)
