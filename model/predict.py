# the prediction of insurence premium
from fastapi.responses import JSONResponse
import pandas as pd
import pickle


# load model from the model folder
with open('model/model.pkl', 'rb') as file:
    model = pickle.load(file)

MODEL_VERSION = "1.0.0"

# get the class labels from the model
class_labels = model.classes_.tolist() # Get the class labels from the model

def predict_output(user_input:dict):
    ##  this is the 7th improvement to add the richness in the output of the model 
    ##  so basically confidence score will also come since we used the random forest
    df = pd.DataFrame([user_input])

    #ptedict the class
    predicted_class = model.predict(df)[0]

    #get probability of all classes main 7th improvement
    probabilities = model.predict_proba(df)[0]
    confidence = max(probabilities)
    


    class_probs=dict(zip(class_labels, map(lambda p: round(p, 4), probabilities)))
    return {
        "predicted_category": predicted_class,  
        "confidence": round(confidence, 4),
        "class_probabilities": class_probs
    }


   