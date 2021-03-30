from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app import db, ml, viz

description = """
The Kickstarter-Success-Predictor deploys a Logistic 
Regression model fit for Kickstarter campaigns.

<img src="https://miro.medium.com/max/4638/1*nOdS52xlJh2n8T2Wu0UbKg.jpeg" 
width="40%" />

"""

app = FastAPI(
    title='🏆 Kickstarter-Success-Predictor',
    description=description,
    version=1.0,
    docs_url='/',
)

app.include_router(db.router, tags=['Database'])
app.include_router(ml.router, tags=['Machine Learning'])
app.include_router(viz.router, tags=['Visualization'])

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

if __name__ == '__main__':
    uvicorn.run(app)
