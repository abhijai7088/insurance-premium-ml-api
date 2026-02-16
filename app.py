# this is the fastapi endpoints to servel the ml model from the fast api this is basically the insurence premium we will take some features from the user and then we will pass those features to the ml model and then we will return the predicted premium to the user info from user and we will predict 

from fastapi import FastAPI
from pydantic import BaseModel,Field,computed_field
from typing import List, Dict, Optional,Annotated,Literal
from fastapi.responses import JSONResponse
import pandas as pd
import pickle

#import the ml model 

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = FastAPI()
# pydabntic model to validate the incoming data 
tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
] 
class UserInput(BaseModel):
    age: int = Annotated[int, Field(..., description="Age of the User", example=30)]  
    weight:Annotated[float, Field(..., description="Weight of the User in kg", example=70.5)]
    height:Annotated[float, Field(..., description="Height of the User in cm", example=175.0)]
    income_lpa:Annotated[float, Field(..., description="Income of the User in LPA", example=5.0)]
    smoker:Annotated[bool, Field(..., description="Smoking status of the User", example=False)]
    city:Annotated[str, Field(..., description="City of residence of the User", example="New York")]
    occupation:Annotated[Literal['retired', 'freelancer', 'student', 'government_job',
       'business_owner', 'unemployed', 'private_job'], Field(..., description="Occupation of the User", example="private_job")]


    @computed_field
    @property
    def bmi(self) -> float:
        height_m = self.height / 100  # Convert height from cm to meters
        return self.weight / (height_m ** 2)  # BMI formula
    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        
        if self.smoker and self.bmi >30:
            return "high"
        if self.smoker or self.bmi > 27:
            return "medium"
        else:
            return "low"
        
    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 25:
            return "young"
        elif self.age < 45:
            return "adult"
        elif self.age < 65:
            return "middle-aged"
        return "senior"
    
    @computed_field
    @property
    def city_tier(self) -> str:
        if self.city in tier_1_cities:
            return 1
        elif self.city in tier_2_cities:
            return 2
        else:
            return 3
        
@app.post("/predict")
def predict_premium(data: UserInput):
    input_df=pd.DataFrame([{
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation
    }])
    #predict the premium using the ml model
    prediction = model.predict(input_df)[0]
    return JSONResponse(content={"predicted_premium": prediction}, status_code=200)
