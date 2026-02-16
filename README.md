🏥 Insurance Premium ML API
Production-Ready FastAPI Backend | ML Powered | Dockerized

A machine learning powered REST API that predicts:

📊 Insurance Risk Category (Low / Medium / High)

💰 Estimated Insurance Premium Amount

Built using FastAPI, Scikit-Learn, and Docker, following production-ready backend architecture practices.

🚀 Project Highlights

Clean FastAPI backend architecture

Feature engineering (BMI calculation)

ML classification + regression models

Pydantic input validation

Modular folder structure

Dockerized deployment

Swagger auto documentation

Separation of training & inference

This project demonstrates backend engineering + ML system integration.

🧠 ML Approach
Feature Engineering

BMI calculated from height & weight

Smoker → numerical encoding

Occupation → categorical encoding

Income-based risk weighting

Models Used

RandomForestClassifier → Risk Category

RandomForestRegressor → Premium Prediction





📂 Project Structure
fastAPI_Docker/
│
├── app.py
├── schema/
│   └── user_schema.py
├── model/
│   ├── risk_model.pkl
│   ├── premium_model.pkl
│
├── config/
│   └── settings.py
│
├── insurance.csv
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md

⚙️ API Endpoint
POST /predict
Request Body
{
  "age": 30,
  "weight": 75,
  "height": 170,
  "income_lpa": 8,
  "smoker": false,
  "city": "Delhi",
  "occupation": "private_job"
}

Response
{
  "bmi": 25.95,
  "risk_category": "medium",
  "predicted_premium": 18250.42
}



