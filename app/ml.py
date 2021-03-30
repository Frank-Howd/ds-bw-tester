"""Machine learning functions."""

import logging
import random

from joblib import load
from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator

log = logging.getLogger(__name__)
router = APIRouter()

# uvicorn app.main:app --reload

classifier = load('app/lr_model.joblib')

class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    category: str = Field(..., example='Poetry')
    goal: float = Field(..., example=1000.0)
    backers: int = Field(..., example=10)
    
    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])

    # @validator('x3')
    # def x3_must_be_positive(cls, value):
    #     """Validate that x3 is a positive number."""
    #     assert value > 0, f'x3 == {value}, must be > 0'
    #     return value


@router.post('/predict')
async def predict(item: Item):
    """
    Make random baseline predictions for classification problem 🔮.

    ### Request Body
    - `category`: string
    - `goal`: float
    - `backers`: int
    
    ### Response
    - `prediction`: success, failed
    - `predict_proba`: float between 0.0 and 1.0, 
    representing the predicted class's probability

    """
    
    X_new = item.to_df()
    # X_new = pd.DataFrame({"category": ["Poetry"],
    #                       "goal": [1000.0],
    #                       "backers": [10]}
    #                       )
    # y_pred = classifier.predict(X_new)
    # # y_pred_proba = classifier.predict_proba(X_new)

    # log.info(X_new)
    # y_pred = random.choice([True, False])
    # y_pred_proba = random.random() / 2 + 0.5
    # return {
    #     'prediction': y_pred
    #     # 'probability': y_pred_proba
    # }
    choice = classifier.predict(X_new)
    probability = classifier.predict_proba(X_new)
    return choice[0], f"{probability[0][1]*100:.2f}% probability"
