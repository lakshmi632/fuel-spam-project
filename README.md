# fuel-spam-project
⛽ Fuel Spam Detection Using Machine Learning

Fuel Spam Detection is a Machine Learning system that identifies abnormal fuel consumption in vehicles to detect misuse, theft, or fraudulent fuel records.

📌 Problem Statement

Fuel misuse and theft cause financial losses, inefficiency, and environmental impact. Manual monitoring is slow and unreliable. This system automates detection using ML models.

🎯 Objectives

Detect abnormal fuel usage automatically
Identify mileage drops
Reduce manual monitoring
Improve detection accuracy

💡 Motivation

Rising fuel prices
Increasing fuel theft
Need for automation
Environmental concerns

📊 Dataset Overview

Synthetic vehicle fuel dataset
12,450 records
Structured and balanced
Used for training & testing

🧾 Dataset Attributes
Feature	Description
Vehicle ID	Unique vehicle identifier
Fuel Type	Petrol / Diesel / CNG / Electric
Fuel Consumed	Fuel used in liters
Distance Travelled	Distance covered in km

⚙️ Feature Engineering
Mileage = Distance Travelled ÷ Fuel Consumed
Average Mileage per vehicle
Current Mileage
Deviation from normal mileage
Threshold (80% of average mileage)
Mileage Drop detection

🧹 Data Preprocessing

Removed noise
Handled missing values
Encoded fuel types
Balanced dataset maintained

🤖 ML Models Used

Logistic Regression
Decision Tree
Random Forest
Support Vector Machine
Final Model: Logistic Regression
Accuracy Achieved: 99.2%

📈 Evaluation Metrics

Accuracy
Precision
Recall
F1 Score
Confusion Matrix

✅ Conclusion

Mileage drop is a strong indicator of fuel misuse. Machine learning improves accuracy and reliability of detection.

🔮 Future Scope

Real-time vehicle data
IoT fuel sensors
Monitoring dashboard
Mobile/Web application

👥 Team Members

Lakshmi Priya P V
Shreya
Salmanul Faris
Malavika
Ananadhakrishnan

## ▶ How to Run the Project

1. Install dependencies  
   pip install -r requirements.txt

2. Run the application  
   python app.py

3. Open browser and access  
   http://127.0.0.1:5000/

