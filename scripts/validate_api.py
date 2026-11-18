"""Validate synchronous API handlers without network or event-loop socket access."""
from pathlib import Path
import json,sys,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fastapi import HTTPException
from pydantic import ValidationError
from pathsnap.api import app,health,trips,prediction,PredictionRequest


def main():
    state=health();available=trips();assert state['status']=='ready';assert len(available['trips'])==10
    assert available['mode']=='student_demo';assert not available['evaluator_ready']
    evidence=[]
    for trip_id in available['trips']:
        start=time.perf_counter();result=prediction(PredictionRequest(trip_id=trip_id))
        assert result['trip_id']==trip_id
        assert len(result['predicted_arc_ids'])==result['observations']==len(result['gps'])
        assert result['predicted_components'];assert 0<=result['accuracy']<=1
        assert not {'reference_route','reference_arc_ids','labels','truth','reference_geometry'} & set(result)
        expected_path=ROOT/'results'/'predictions'/(trip_id+'.json')
        expected=json.loads(expected_path.read_text())
        assert result['predicted_arc_ids']==expected['predicted_arc_ids'], 'API and experiment predictions differ'
        assert abs(result['accuracy']-expected['accuracy'])<1e-12, 'API and experiment accuracy differ'
        encoded=json.dumps(result,allow_nan=False)
        summary={key:result[key] for key in ['trip_id','observations','accuracy','inference_seconds','components','mode']}
        summary['response_bytes']=len(encoded.encode());evidence.append(summary)
        print(json.dumps(summary),flush=True)
    try: prediction(PredictionRequest(trip_id='unknown_trip'))
    except HTTPException as error:assert error.status_code==404
    else:raise AssertionError('Unknown trip accepted')
    try:PredictionRequest(trip_id='../train')
    except ValidationError:pass
    else:raise AssertionError('Traversal input accepted')
    schema=app.openapi();assert '/api/predict' in schema['paths']
    result={'validation_mode':'direct synchronous FastAPI handler calls',
            'transport_layer_validated':False,'event_loop_validated':False,
            'successful_trips':len(evidence),'health':state,'trips':evidence,
            'unknown_trip_rejected':True,'path_traversal_rejected':True,
            'reference_labels_excluded':True,'openapi_schema_generated':True,
            'saved_experiment_predictions_match':True,
            'model_sha256':hashlib.sha256((ROOT/'artifacts'/'final_model.joblib').read_bytes()).hexdigest(),
            'limitation':'Socket access, including asyncio socketpair, is restricted; HTTP and LAN deployment remain unverified.'}
    (ROOT/'results'/'api_validation.json').write_text(json.dumps(result,indent=2))
    print('Validated all handlers and ten student trips.',flush=True)


if __name__=='__main__':main()
