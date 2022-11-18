import orjson
import typing
# from base import *
from datetime import datetime
from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse
import requests

# from conf import Config
from pydantic import BaseModel
from typing import List, Optional, Union
from predict import run_summarizer
from web_scrape import web_scrape


# ========== PREDICTION MODEL INPUTS ==========
class PredictInput(BaseModel):
    """Prediction settings"""
    data: str

EXAMPLE_JSON = {
    'PredictInput': {
        "data": 'Singapore does not rule out bringing back some COVID-19 restrictions, but it will try its "very best" not '
        'to disrupt the normal lives of people, Health Minister Ong Ye Kung said on Saturday (Oct 15). '
        'The ministry is closely monitoring the situation,and it does not rule out reimposing safe management measures (SMM) such as '
        'mask-wearing, but he pointed out that the country is now riding out yet another wave without these measures.'
        '"We have never declared that COVID is an endemic disease, some countries have. We have never declared that it\'s no longer '
        'a social health threat, some countries have. We much prefer to let action and our lives speak for itself," said Mr Ong. '
        '"And what we have done is that with every successive wave that we have gone through, we relax the safe distancing, '
        'safe measurement measures ... to the extent that now almost everything has been dismantled. "So for all, in reality, '
        'having done all that, in effect, we are living with COVID like it is an endemic disease ... look at where we are now '
        '- we\'re going through a wave without SMM." Mr Ong said that Singapore will try its "very, very best" never to go back'
        'to the days of "circuit breaker", "heightened alert" or "anything that severely disrupt our normal lives". '
        'Giving an update on the latest COVID-19 wave of XBB infections, Mr Ong said at a Ministry of Health (MOH) press '
        'conference that the number of cases is rising, driven by the new Omicron strain and by reinfections."'
    },
    }

# ========== API Definition ==========
class ORJSONResponse(JSONResponse):
    """Custom JSONResponse class for returning NaN float values in JSON."""
    media_type = "application/json"


    def render(self, content: typing.Any) -> bytes:
        return orjson.dumps(content)


app = FastAPI(default_response_class=ORJSONResponse)

@app.get('/')
async def home(request: Request):
    return "Welcome"

# ============================== Prediction ==============================
@app.post('/summarize/', tags=['summarize'])
async def api_predict(request: Request, inputs: PredictInput = Body(
        ..., example=EXAMPLE_JSON["PredictInput"]
    )
):

    inputs = inputs.dict()

    assert inputs['data'] is not None, "`data` was not provided for /predict/"

    # check if the input is a valid URL
    try:
        response = requests.get(inputs['data'])
        inputs['data'] = web_scrape(inputs['data']) if response.status_code == 200 else inputs['data']
    except:
        pass

    # Running summarizer
    y_pred = run_summarizer(
        text=inputs['data']
    )

    return y_pred


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host='0.0.0.0', port=8000, debug=False, reload=False)
