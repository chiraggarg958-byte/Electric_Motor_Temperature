Electric Motor Temperature Prediction
Category: Machine Learning
Skills Used: Python, Exploratory Data Analysis (EDA), NumPy, Scikit-Learn
1. Introduction
Electric motors are used in almost every modern machine — from electric vehicles and industrial drives to home appliances and robots. Among them, Permanent Magnet Synchronous Machines (PMSMs) are very popular because they offer high efficiency, compact size, and excellent performance.

However, one key challenge is temperature rise in the motor. Excessive heating of the rotor can lead to reduced performance or even permanent damage. Hence, predicting the rotor temperature accurately helps in improving reliability, safety, and energy efficiency.

This project focuses on building a machine learning model that can estimate the rotor temperature (pm) of a PMSM using various operating parameters. We also perform a detailed exploratory data analysis (EDA), train multiple models, compare their results, and finally save the best-performing one for deployment using Flask.

2. Project Objectives
- Explore and visualize the dataset to understand the relationships between variables.
- Apply data cleaning, feature engineering, and normalization.
- Train and test multiple regression models: Linear Regression, Decision Tree, Random Forest, and SVM.
- Evaluate and compare their accuracy.
- Save the best model in .pkl format and prepare it for Flask integration.

3. Dataset Overview
We used the dataset measures_v2.csv, which contains various PMSM operating measurements collected from several motor profiles.

Key features include ambient, coolant, motor speed, i_d, i_q, u_d, u_q, stator_yoke, stator_tooth, stator_winding, torque, and profile_id. The target variable is pm (rotor temperature).

4. Tools and Technologies
Python (Jupyter/Anaconda) - Coding environment
Pandas, NumPy - Data handling
Matplotlib, Seaborn - Visualization
Scikit-Learn - Model building and evaluation
Joblib - Model saving
Flask - Web integration

5. Exploratory Data Analysis (EDA)
Data loading, profile distribution, feature distributions, scatter plots, correlation heatmap, and temperature trends were performed to understand data relationships and prepare for modelling.

6. Data Preprocessing
- Dropped unnecessary columns
- Split the dataset into train and test sets
- Normalized values using MinMaxScaler
- Saved the scaler using joblib

7. Model Building and Evaluation
We trained and compared Linear Regression, Decision Tree Regressor, Random Forest Regressor, and Support Vector Machine models. Each was evaluated using R², MAE, and RMSE.

The Random Forest Regressor gave the best performance overall.

8. Saving the Model
The best model was saved using joblib.dump(best model, "best_model.pkl"). The scaler was saved as transform.save to maintain consistent input scaling.

9. Flask Integration (Next Step)
The next step is to integrate the model into a Flask web app that takes user input and returns the predicted rotor temperature.

10. Results and Insights
- The Random Forest Regressor performed best among all models.
- Proper feature scaling and EDA improved results.
- The model and scaler were successfully saved for deployment.

11. Future Improvements
- Add real-time sensor data for online predictions.
- Use LSTM models for time-series forecasting.
- Deploy on an edge device for on-board temperature monitoring.
- Build a web dashboard for analytics.

12. Conclusion
This project demonstrates how machine learning can predict the temperature of an electric motor’s rotor, improving reliability and efficiency. With data analysis, modelling, and deployment, the system supports predictive maintenance and smarter energy management in PMSM applications.
