from functools import lru_cache
from pathlib import Path
import json, os, time
import joblib
import numpy as np
from .data import load_trip, candidates, load_reference
from .decoder import decode, route_geometry

ROOT=Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def assets():
    config=json.loads((ROOT/'artifacts'/'config.json').read_text())
    model=joblib.load(ROOT/'artifacts'/'final_model.joblib')
    return config,model


def dataset():
    evaluator=os.environ.get('PATHSNAP_EVALUATOR_ROOT')
    if evaluator: return Path(evaluator).resolve(),'evaluator'
    development=os.environ.get('PATHSNAP_DATA_ROOT')
    root=Path(development) if development else ROOT.parent/'Student folder'
    return (root/'student_test').resolve(),'student_demo'


def available_trips():
    root,mode=dataset()
    trips=[]
    if root.exists():
        for folder in sorted(root.iterdir()):
            if folder.is_dir() and all((folder/(folder.name+ext)).exists() for ext in ['.track','.nodes','.arcs']):
                trips.append(folder.name)
    return trips,mode


def predict(folder):
    """This prediction function cannot access reference labels."""
    from .experiment import scores
    config,model=assets()
    trip=load_trip(folder); c=candidates(trip,config['candidates'])
    probabilities=scores(model,c['features'])
    ids,chosen,resets=decode(trip,c,probabilities,**config['decoder'])
    geometry,arc_ids=route_geometry(trip,c,chosen,resets)
    return trip,ids,geometry,resets,arc_ids


def process(trip_id):
    choices,mode=available_trips()
    if trip_id not in choices: raise KeyError('Unknown trip ID')
    root,_=dataset(); folder=root/trip_id
    start=time.perf_counter()
    trip,prediction,geometry,resets,arc_ids=predict(folder)
    inference=time.perf_counter()-start
    # The server-side scorer is invoked only after inference has completed.
    reference_path=folder/(trip_id+'.route')
    accuracy=None
    if reference_path.exists():
        truth,_,_=load_reference(folder,trip)
        accuracy=float(np.mean(prediction==truth))
    config,_=assets()
    lonlat=trip.track[:,:2]
    lower=lonlat.min(axis=0)-0.001;upper=lonlat.max(axis=0)+0.001
    starts,ends=trip.nodes[trip.arcs[:,0]],trip.nodes[trip.arcs[:,1]]
    visible=(((starts>=lower)&(starts<=upper)).all(axis=1)|((ends>=lower)&(ends<=upper)).all(axis=1))
    # Small unlabelled network overlay. Never include reference route geometry or IDs.
    edge_indices=np.flatnonzero(visible)[:12000]
    roads=np.stack([starts[edge_indices,::-1],ends[edge_indices,::-1]],axis=1).tolist()
    return {'trip_id':trip_id,'mode':mode,'model':config['selected_model'],
            'observations':len(trip.track),'discarded_observations':trip.issues['invalid_observations'],
            'gps':trip.track[:,:2][:,::-1].tolist(),'original_rows':trip.original_rows.tolist(),'predicted_components':geometry,
            'predicted_arc_ids':arc_ids,'road_network':roads,'reset_indices':resets[1:],
            'inference_seconds':inference,'accuracy':accuracy,
            'accuracy_basis':'Derived monotone geometric reference alignment; not native point annotations' if accuracy is not None else 'Reference unavailable',
            'components':len(geometry)}
