from pathlib import Path
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .service import ROOT, available_trips, assets, process

log=logging.getLogger('uvicorn.error')
app=FastAPI(title='PathSnap',version='1.0',description='Directed GPS map matching with server-side scoring')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['GET','POST'],allow_headers=['Content-Type'],allow_credentials=False)
app.mount('/ui',StaticFiles(directory=str(ROOT/'ui')),name='ui')


class PredictionRequest(BaseModel):
    trip_id:str=Field(min_length=1,max_length=64,pattern=r'^[A-Za-z0-9_-]+$')


@app.get('/')
def home(): return FileResponse(ROOT/'ui'/'index.html')


@app.get('/health')
def health():
    try:
        config,_=assets(); choices,mode=available_trips()
        return {'status':'ready','model':config['selected_model'],'mode':mode,'trip_count':len(choices)}
    except (FileNotFoundError,ValueError) as error:
        raise HTTPException(503,'Train the model before starting inference') from error


@app.get('/api/trips')
def trips():
    choices,mode=available_trips()
    return {'trips':choices,'mode':mode,'expected_evaluator_trips':10,
            'evaluator_ready':mode=='evaluator' and len(choices)==10}


@app.post('/api/predict')
def prediction(request:PredictionRequest):
    log.info('PathSnap inference request trip_id=%s',request.trip_id)
    try:
        result=process(request.trip_id)
    except KeyError as error: raise HTTPException(404,'Trip not found') from error
    except FileNotFoundError as error: raise HTTPException(503,'Required data or trained model is missing') from error
    except ValueError as error: raise HTTPException(422,str(error)) from error
    log.info('PathSnap completed trip_id=%s observations=%s seconds=%.3f accuracy=%s',request.trip_id,result['observations'],result['inference_seconds'],result['accuracy'])
    return result
