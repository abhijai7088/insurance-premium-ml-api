# this is the fastapi endpoints to servel the ml model from the fast api this is basically the insurence premium we will take some features from the user and then we will pass those features to the ml model and then we will return the predicted premium to the user info from user and we will predict 

from fastapi import FastAPI
from pydantic import BaseModel,Field,computed_field,field_validator
from typing import List, Dict, Optional,Annotated,Literal
from fastapi.responses import JSONResponse
import pandas as pd

from schema.user_input import UserInput # 5(a)th improvement so that we can seperate the schema of the app.py
from model.predict import predict_output,model,MODEL_VERSION  # 5(c) improvement to seperate the prediction logic from the app.py and move it to model folder

from schema.prediction_response import PredictionResponse # 8 improvement to seperate the prediction response schema from the app.py and move it to schema folder



#import the ml model now this is improved in model folder

'''with open('models/model.pkl', 'rb') as file:
    model = pickle.load(file)'''

# 4th improvement model version #####

#MODEL_VERSION = "1.0.0"

#--------------------------------------------------------------------------------------

app = FastAPI()
# pydabntic model to validate the incoming data 
# tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
# tier_2_cities = [
#     "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
#     "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
#     "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
#     "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
#     "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
#     "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
# ] 
# class UserInput(BaseModel):
#     age: int = Annotated[int, Field(..., description="Age of the User", example=30)]  
#     weight:Annotated[float, Field(..., description="Weight of the User in kg", example=70.5)]
#     height:Annotated[float, Field(..., description="Height of the User in cm", example=175.0)]
#     income_lpa:Annotated[float, Field(..., description="Income of the User in LPA", example=5.0)]
#     smoker:Annotated[bool, Field(..., description="Smoking status of the User", example=False)]
#     city:Annotated[str, Field(..., description="City of residence of the User", example="New York")]
#     occupation:Annotated[Literal['retired', 'freelancer', 'student', 'government_job',
#        'business_owner', 'unemployed', 'private_job'], Field(..., description="Occupation of the User", example="private_job")]
    
# #####  1st improvements in the API  #########
#     '''this field validator is added to avoid
#     the user to enter the city any mode 
#     it will convery that into title like --->  Mumbai '''

#     @field_validator('city')
#     @classmethod
#     def normalize_city(cls, v:str)-> str:
#         v=v.strip().title()  # Remove leading/trailing whitespace and convert to title case
#         return v
# # ---------------------------------------------------------------------------------------------------------------------

#     @computed_field
#     @property
#     def bmi(self) -> float:
#         height_m = self.height / 100  # Convert height from cm to meters
#         return self.weight / (height_m ** 2)  # BMI formula
#     @computed_field
#     @property
#     def lifestyle_risk(self) -> str:
        
#         if self.smoker and self.bmi >30:
#             return "high"
#         if self.smoker or self.bmi > 27:
#             return "medium"
#         else:
#             return "low"
        
#     @computed_field
#     @property
#     def age_group(self) -> str:
#         if self.age < 25:
#             return "young"
#         elif self.age < 45:
#             return "adult"
#         elif self.age < 65:
#             return "middle-aged"
#         return "senior"
    
#     @computed_field
#     @property
#     def city_tier(self) -> str:
#         if self.city in tier_1_cities:
#             return 1
#         elif self.city in tier_2_cities:
#             return 2
#         else:
#             return 3
        

####   2nd improvements in the API  #########        
# to inhance the user  and let API is woking inserting the home 
@app.get("/")
def home():
    return {"message": "Welcome to the Insurance Premium Prediction API! Please use the /predict endpoint to get your insurance premium category."}    


# -------------------------------------------------------------------------------------


######### 3rd improvements in the API to add healt so that API host requires this #########
# cloud serive providers like AWS, Azure, GCP require the API to have a health check endpoint to check the status of the API and to ensure that the API is working fine and to avoid any downtime issues so we will add a health check endpoint to our API

@app.get("/health")
def health_check():
    return {"status": "OK",
            "model_version": MODEL_VERSION,
            "model_status": "loaded" if model else "not loaded",
            "message": "The API is healthy and ready to serve predictions."}

#------------------------------------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)  # 8 improvement to add the response model to the predict endpoint
def predict_premium(data: UserInput):
    user_input={
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation
   }
    

    #predict the premium using the ml model ---> moved to model folder


    '''in python we need to use the try catch when we are using prediction from another files'''  ## 6th imporvement so code is not get wasted
    try:
        prediction = predict_output(user_input)
        return JSONResponse(content={"Response": prediction}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
