from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

model = joblib.load("regression.joblib")

class HouseData(BaseModel):
    size: float
    nb_rooms: int
    garden: int

@app.post("/predict")
def predict(house_data: HouseData):
    size = house_data.size
    nb_rooms = house_data.nb_rooms
    garden = house_data.garden


    data = pd.DataFrame([[size, nb_rooms, garden]], columns=['size', 'nb_rooms', 'garden'])
    predicted_price = model.predict(data)[0]

    return { "price_prediction": predicted_price }
