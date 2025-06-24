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

1. **Data Exploration & Visualization (EDA):**
   - Univariate & bivariate analysis (e.g., gender, tenure, contract type).
   - Correlation heatmaps & churn distribution.
2. **Preprocessing:**
   - Categorical encoding (Label, One-Hot).
   - Feature scaling.
   - Saving encoders & scalers for reuse.
3. **Modeling:**
   - Hyperparameter tuning via `GridSearchCV`.
   - ANN built using TensorFlow/Keras.
   - EarlyStopping & TensorBoard for evaluation.
4. 📉 Model Training Metrics (TensorBoard)

Below are the training metrics visualized using TensorBoard:

<p align="center">
  <img src="https://github.com/soham-kar/customer-churn-prediction/blob/main/tensorboard_metrics.png" width="800"/>
</p>

- ✅ **Validation Accuracy** reached **~86.9%**
- 📉 **Validation Loss** consistently decreased, indicating effective learning
- 📈 Training and validation curves show **no overfitting**, thanks to early stopping
5. **Deployment:**
   - Deployed via a **Streamlit app** for interactive predictions.

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

