# ⚡ Electric Motor Temperature Prediction

> **Machine Learning-based Rotor Temperature Prediction for Permanent Magnet Synchronous Motors (PMSMs)**

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

Electric motors power everything from **electric vehicles** and **industrial automation** to **robots** and **household appliances**. Among these, **Permanent Magnet Synchronous Machines (PMSMs)** are widely used due to their high efficiency and excellent performance.

One major challenge in PMSMs is **rotor overheating**, which can reduce efficiency, increase maintenance costs, and even damage the motor.

This project leverages **Machine Learning** to accurately predict the **Permanent Magnet (Rotor) Temperature (`pm`)** using multiple motor operating parameters. The workflow includes **Exploratory Data Analysis (EDA), feature engineering, model training, evaluation, and deployment preparation using Flask.**

---

# 🚀 Features

- 📊 Comprehensive Exploratory Data Analysis (EDA)
- 🧹 Data Cleaning & Preprocessing
- 📈 Feature Scaling using MinMaxScaler
- 🤖 Multiple Machine Learning Models
- 📉 Performance Comparison using Evaluation Metrics
- 💾 Model Serialization using Joblib
- 🌐 Flask Integration Ready
- ⚡ Predictive Maintenance Support

---

# 📂 Dataset

Dataset Used:

**measures_v2.csv**

The dataset contains operating measurements collected from several PMSM motor profiles.

### Input Features

- Ambient Temperature
- Coolant Temperature
- Motor Speed
- Current (i_d, i_q)
- Voltage (u_d, u_q)
- Torque
- Stator Yoke Temperature
- Stator Tooth Temperature
- Stator Winding Temperature
- Profile ID

### Target Variable

**pm** → Permanent Magnet (Rotor) Temperature

---

# 🛠 Tech Stack

### Programming

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib

### Deployment

- Flask

---

# 📊 Exploratory Data Analysis

The dataset was analyzed using multiple visualization techniques including:

- Feature Distribution Analysis
- Correlation Heatmap
- Scatter Plots
- Temperature Trend Analysis
- Profile-wise Distribution
- Outlier Detection

EDA helped identify important relationships between operating parameters and rotor temperature.

---

# ⚙ Data Preprocessing

The preprocessing pipeline included:

- Removing unnecessary columns
- Handling missing values (if present)
- Feature Engineering
- Train-Test Split
- MinMax Feature Scaling
- Saving the trained scaler using Joblib

---

# 🤖 Machine Learning Models

The following regression models were trained and evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor (SVR)

---

# 📈 Model Evaluation

Models were evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### 🏆 Best Performing Model

**Random Forest Regressor**

It achieved the highest prediction accuracy while maintaining good generalization performance.

---

# 💾 Model Saving

The trained model was exported using Joblib.

```python
joblib.dump(best_model, "best_model.pkl")
```

The MinMaxScaler was also saved to ensure consistent preprocessing during inference.

```python
joblib.dump(scaler, "transform.save")
```

---

# 🌐 Deployment

The project is deployment-ready using **Flask**.

Prediction Workflow

```
User Input
      │
      ▼
Flask Backend
      │
      ▼
Load Saved Scaler
      │
      ▼
Load Random Forest Model
      │
      ▼
Temperature Prediction
      │
      ▼
Display Predicted Rotor Temperature
```

---

# 📊 Results

✅ Random Forest Regressor achieved the best overall performance.

✅ Feature scaling significantly improved model accuracy.

✅ Exploratory Data Analysis helped identify key feature relationships.

✅ The trained model and scaler were successfully saved for production deployment.

---

# 🔮 Future Improvements

- Real-time IoT Sensor Integration
- LSTM-based Time Series Prediction
- Edge Device Deployment
- Interactive Dashboard
- Live Temperature Monitoring
- Cloud Deployment
- Predictive Maintenance Alerts

---

# 📚 Learning Outcomes

Through this project, I gained hands-on experience with:

- Data Preprocessing
- Exploratory Data Analysis
- Regression Algorithms
- Model Evaluation
- Feature Scaling
- Model Serialization
- Flask Deployment
- Predictive Maintenance Applications

---

# 📸 Project Workflow

```
Motor Sensor Data
        │
        ▼
Data Cleaning
        │
        ▼
EDA & Visualization
        │
        ▼
Feature Scaling
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Random Forest Selected
        │
        ▼
Model Saved (.pkl)
        │
        ▼
Flask Deployment
        │
        ▼
Rotor Temperature Prediction
```

---

# 👨‍💻 Author

**Chirag Garg**

Computer Science Engineering Student

Machine Learning • Backend Development • Artificial Intelligence

---

⭐ If you found this project useful, consider giving it a **Star** on GitHub!
