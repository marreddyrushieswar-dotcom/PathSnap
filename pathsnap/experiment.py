"""Reproducible trip-grouped experiments. No evaluator partition is opened here."""
from pathlib import Path
import argparse, csv, json, time, warnings, hashlib
import numpy as np
import joblib
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from scipy.special import ndtr
from .data import load_trip, candidates, load_reference, feature_names
from .decoder import decode, continuity, route_geometry

ROOT = Path(__file__).resolve().parents[1]
SEED = 181126


def write_csv(path, rows):
    if not rows: return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def scores(model, x):
    n, k, features = x.shape
    positive = model.predict_proba(x.reshape(-1, features))[:, list(model.classes_).index(1)]
    positive = np.clip(positive.reshape(n, k), 1e-12, 1)
    return positive/positive.sum(axis=1, keepdims=True)


def metrics(probability, ids, labels, groups):
    winner = probability.argmax(axis=1)
    predicted = ids[np.arange(len(ids)), winner]
    per_trip = [f1_score(labels[groups == g], predicted[groups == g], average='macro', zero_division=0)
                for g in np.unique(groups)]
    truth = (ids == labels[:, None])
    p_true = np.sum(probability * truth, axis=1)
    binary_pred = np.zeros(ids.shape, dtype=int); binary_pred[np.arange(len(ids)), winner] = 1
    return {'top1_accuracy': float(accuracy_score(labels, predicted)),
            'macro_f1': float(np.mean(per_trip)),
            'precision': float(np.mean([precision_score(labels[groups==g], predicted[groups==g], average='macro', zero_division=0) for g in np.unique(groups)])),
            'recall': float(np.mean([recall_score(labels[groups==g], predicted[groups==g], average='macro', zero_division=0) for g in np.unique(groups)])),
            'log_loss': float(-np.log(np.clip(p_true, 1e-12, 1)).mean()),
            'candidate_miss_rate': float((~truth.any(axis=1)).mean()),
            'candidate_macro_f1': float(f1_score(truth.ravel(), binary_pred.ravel(), average='macro', zero_division=0)),
            'confusion_matrix_candidate': confusion_matrix(truth.ravel(), binary_pred.ravel(), labels=[0,1]).tolist()}


def models(feature_version=1):
    choices = {
        'LogisticRegression': make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=400, class_weight='balanced', random_state=SEED)),
        'GaussianNB': make_pipeline(StandardScaler(), GaussianNB(var_smoothing=1e-5)),
        'DecisionTree': DecisionTreeClassifier(max_depth=12, min_samples_leaf=12, class_weight='balanced', random_state=SEED),
        'RandomForest': RandomForestClassifier(n_estimators=60, max_depth=15, min_samples_leaf=5, class_weight='balanced', n_jobs=2, random_state=SEED),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=85, max_leaf_nodes=23, learning_rate=0.08, l2_regularization=1, random_state=SEED),
    }
    if feature_version == 2:
        choices['LogisticRegression'].set_params(logisticregression__max_iter=700)
        choices['DecisionTree'].set_params(max_depth=18, min_samples_leaf=15)
        choices['RandomForest'].set_params(n_estimators=90, max_depth=22, min_samples_leaf=3)
        choices['HistGradientBoosting'].set_params(max_iter=180, max_leaf_nodes=31, learning_rate=.08)
    return choices


def inventory(folder, trip, residual, route, split):
    dt = np.diff(trip.track[:, 2])
    return {'split':split, 'trip_id':trip.trip_id, 'observations':len(trip.track),
            'nodes':len(trip.nodes), 'arcs':len(trip.arcs), 'route_arcs':len(route),
            'min_lon':float(trip.track[:,0].min()), 'max_lon':float(trip.track[:,0].max()),
            'min_lat':float(trip.track[:,1].min()), 'max_lat':float(trip.track[:,1].max()),
            'median_dt_s':float(np.median(dt)) if len(dt) else 0,
            'one_second_steps':int((dt==1).sum()), 'total_steps':len(dt),
            'reference_residual_median_m':float(np.median(residual)),
            'reference_residual_p95_m':float(np.percentile(residual,95)),
            **trip.issues,
            **{ext+'_bytes':(folder/(trip.trip_id+'.'+ext)).stat().st_size for ext in ['track','nodes','arcs','route']}}


def prepare(data, max_observations=350, k=12, feature_version=1):
    cache=ROOT/'artifacts'/'prepared'; cache.mkdir(exist_ok=True,parents=True)
    training=[]; testing=[]; rows=[]
    rng=np.random.default_rng(SEED)
    for split in ['train','student_test']:
        folders=sorted((data/split).iterdir())
        for number,folder in enumerate(folders):
            if not folder.is_dir(): continue
            signature=hashlib.sha256()
            signature.update(str(folder.resolve()).encode())
            for extension in ['.nodes','.arcs','.track','.route']:
                info=(folder/(folder.name+extension)).stat()
                signature.update(f'{extension}:{info.st_size}:{info.st_mtime_ns}'.encode())
            target=cache/(split+'_'+folder.name+f'_k{k}_v{feature_version}_'+signature.hexdigest()[:12]+'.npz')
            trip=load_trip(folder)
            if target.exists():
                with np.load(target) as loaded: c={key:loaded[key] for key in ['ids','features','fractions','distances']}; labels=loaded['labels']; route=loaded['route']; residual=loaded['residual']
            else:
                c=candidates(trip,k,feature_version); labels,route,residual=load_reference(folder,trip)
                np.savez_compressed(target,**c,labels=labels,route=route,residual=residual)
            rows.append(inventory(folder,trip,residual,route,split))
            if split=='train':
                take=np.sort(rng.choice(len(labels),min(max_observations,len(labels)),replace=False))
                training.append((c['features'][take],c['ids'][take],labels[take],np.full(len(take),folder.name)))
            else: testing.append((folder,trip,c,labels,route))
            print(f'Prepared {split} {number+1}/{len(folders)} {folder.name}',flush=True)
    write_csv(ROOT/'results'/'dataset_inventory.csv',rows)
    arrays=tuple(np.concatenate([t[i] for t in training]) for i in range(4))
    return arrays,testing,rows


def fit(model,x,ids,labels):
    target=(ids==labels[:,None]).astype(int).ravel()
    if len(np.unique(target))<2: raise ValueError('Training requires positive and negative candidate labels')
    return model.fit(x.reshape(-1,x.shape[-1]),target)


def cross_validate(model, arrays, splits):
    x,ids,labels,groups=arrays
    results=[]
    for train,val in splits:
        estimator=fit(clone(model),x[train],ids[train],labels[train])
        results.append(metrics(scores(estimator,x[val]),ids[val],labels[val],groups[val])['macro_f1'])
    return float(np.mean(results)),float(np.std(results)),results


def evaluate(model,testing):
    all_p=[]; all_ids=[]; all_labels=[]; all_groups=[]; inference=0
    for folder,trip,c,labels,route in testing:
        start=time.perf_counter(); p=scores(model,c['features']); inference+=time.perf_counter()-start
        all_p.append(p);all_ids.append(c['ids']);all_labels.append(labels);all_groups.append(np.full(len(labels),trip.trip_id))
    result=metrics(np.concatenate(all_p),np.concatenate(all_ids),np.concatenate(all_labels),np.concatenate(all_groups))
    result['inference_seconds']=inference
    return result,all_p


def tune_spec(name, unit, feature_version=1):
    u,v=map(float,unit)
    if name=='LogisticRegression':
        params={'logisticregression__C':10**(-3+6*u), 'logisticregression__tol':10**(-5+3*v)}
    elif name=='GaussianNB': params={'gaussiannb__var_smoothing':10**(-10+8*u)}
    elif name=='DecisionTree': params={'max_depth':int(4+20*u),'min_samples_leaf':int(2+30*v)}
    elif name=='RandomForest': params={'max_depth':int(6+18*u),'min_samples_leaf':int(1+15*v)}
    else: params={'learning_rate':10**(-2+1.4*u),'max_leaf_nodes':int(8+40*v)}
    return clone(models(feature_version)[name]).set_params(**params),params


def bayesian_search(name,arrays,splits,trials=10,feature_version=1):
    rng=np.random.default_rng(SEED); positions=[]; objectives=[]; records=[]
    start=time.perf_counter()
    for trial in range(trials):
        if trial<3: point=rng.random(2)
        else:
            gp=GaussianProcessRegressor(kernel=Matern(length_scale=.3,nu=2.5)+WhiteKernel(1e-5),normalize_y=True,random_state=SEED,n_restarts_optimizer=0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore'); gp.fit(np.array(positions),np.array(objectives))
            pool=rng.random((512,2)); mu,sigma=gp.predict(pool,return_std=True)
            improvement=mu-max(objectives)-0.001
            z=improvement/np.maximum(sigma,1e-12)
            expected=improvement*ndtr(z)+sigma*np.exp(-z*z/2)/np.sqrt(2*np.pi)
            point=pool[np.argmax(expected)]
        estimator,params=tune_spec(name,point,feature_version)
        mean,std,folds=cross_validate(estimator,arrays,splits)
        positions.append(point.tolist());objectives.append(mean)
        records.append({'trial':trial,'method':'random_initialization' if trial<3 else 'GP_expected_improvement',
                        'params':params,'cv_macro_f1':mean,'std':std,'fold_scores':folds,'unit_coordinates':point.tolist()})
        (ROOT/'results'/'bayesian_trials.json').write_text(json.dumps(records,indent=2))
        print(f'Bayesian trial {trial+1}/{trials}: {mean:.4f} {params}',flush=True)
    best=int(np.argmax(objectives)); estimator,params=tune_spec(name,positions[best],feature_version)
    return estimator,params,objectives[best],time.perf_counter()-start


def run(data,trials=10,max_observations=350,feature_version=1):
    for directory in ['results','artifacts','reports']: (ROOT/directory).mkdir(exist_ok=True,parents=True)
    arrays,testing,inventory_rows=prepare(data,max_observations,feature_version=feature_version)
    x,ids,labels,groups=arrays
    splits=list(GroupKFold(n_splits=3).split(x,labels,groups))
    split_record=[{'fold':i,'train_trips':sorted(np.unique(groups[a]).tolist()),'validation_trips':sorted(np.unique(groups[b]).tolist())} for i,(a,b) in enumerate(splits)]
    (ROOT/'results'/'cv_splits.json').write_text(json.dumps(split_record,indent=2))
    rows=[]; detail={}; trained={}; test_scores={}
    for name,model in models(feature_version).items():
        start=time.perf_counter(); mean,std,folds=cross_validate(model,arrays,splits)
        fitted=fit(clone(model),x,ids,labels); elapsed=time.perf_counter()-start
        result,probabilities=evaluate(fitted,testing)
        training=metrics(scores(fitted,x),ids,labels,groups)
        row={'model':name,'cv_macro_f1':mean,'cv_std':std,'train_macro_f1':training['macro_f1'],
             **{key:result[key] for key in ['top1_accuracy','macro_f1','precision','recall','log_loss','candidate_miss_rate','candidate_macro_f1','inference_seconds']},
             'training_and_cv_seconds':elapsed}
        rows.append(row);detail[name]={'parameters':model.get_params(deep=False).__repr__(),'fold_scores':folds,**result}
        trained[name]=fitted;test_scores[name]=probabilities
        joblib.dump(fitted,ROOT/'artifacts'/(name+'.joblib'),compress=3)
        write_csv(ROOT/'results'/'model_comparison.csv',rows)
        print(f'{name}: CV {mean:.4f}, student F1 {result["macro_f1"]:.4f}',flush=True)
    selected=max(rows,key=lambda row:row['cv_macro_f1'])['model']
    tuned,params,cv_score,search_seconds=bayesian_search(selected,arrays,splits,trials,feature_version)
    start=time.perf_counter(); fit(tuned,x,ids,labels); refit_seconds=time.perf_counter()-start
    tuned_result,tuned_probabilities=evaluate(tuned,testing)
    # Keep untuned model if Bayesian CV evidence does not improve it.
    untuned_cv=next(r['cv_macro_f1'] for r in rows if r['model']==selected)
    use_tuned=cv_score>untuned_cv
    final=tuned if use_tuned else trained[selected]
    final_probabilities=tuned_probabilities if use_tuned else test_scores[selected]
    joblib.dump(final,ROOT/'artifacts'/'final_model.joblib',compress=3)
    decoder_rows=[]; prediction_dir=ROOT/'results'/'predictions';prediction_dir.mkdir(exist_ok=True)
    for test_index,(folder,trip,c,truth,route) in enumerate(testing):
        p=final_probabilities[test_index]
        raw=p.argmax(axis=1)
        start=time.perf_counter();predicted,chosen,resets=decode(trip,c,p);seconds=time.perf_counter()-start
        for variant,columns in [('classifier',raw),('hmm',chosen)]:
            pred=c['ids'][np.arange(len(truth)),columns]
            a,b=set(pred.tolist()),set(route.tolist()); set_f1=2*len(a&b)/max(len(a)+len(b),1)
            decoder_rows.append({'trip_id':trip.trip_id,'variant':variant,'accuracy':float(np.mean(pred==truth)),
                                 'macro_f1':float(f1_score(truth,pred,average='macro',zero_division=0)),
                                 'route_continuity':continuity(trip,c,columns),'route_arc_set_f1':set_f1,
                                 'resets':len(resets)-1 if variant=='hmm' else 0,'decode_seconds':seconds if variant=='hmm' else 0})
        geometry,arc_ids=route_geometry(trip,c,chosen,resets)
        output={'trip_id':trip.trip_id,'gps':trip.track[:,:2][:,::-1].tolist(),
                'predicted_components':geometry,'predicted_arc_ids':arc_ids,'original_rows':trip.original_rows.tolist(),
                'resets':resets,'accuracy':float(np.mean(predicted==truth)),'reference_kind':'monotone-geometric proxy alignment'}
        (prediction_dir/(trip.trip_id+'.json')).write_text(json.dumps(output))
        print(f'Decoded student trip {test_index+1}/{len(testing)} {trip.trip_id}',flush=True)
    # Route continuity for every classifier, using the same directed transition matrices.
    for row in rows:
        values=[continuity(trip,c,test_scores[row['model']][i].argmax(axis=1)) for i,(_,trip,c,_,_) in enumerate(testing)]
        row['route_continuity']=float(np.mean(values))
    write_csv(ROOT/'results'/'model_comparison.csv',rows)
    write_csv(ROOT/'results'/'decoder_comparison.csv',decoder_rows)
    config={'seed':SEED,'features':feature_names(feature_version),'feature_version':feature_version,'candidates':12,'selected_model':selected,'use_tuned':use_tuned,
            'bayesian_parameters':params,'bayesian_cv_macro_f1':cv_score,'untuned_cv_macro_f1':untuned_cv,
            'tuned_student_results':tuned_result,'bayesian_search_seconds':search_seconds,'refit_seconds':refit_seconds,
            'trials':trials,'cv_folds':3,'maximum_training_observations_per_trip':max_observations,
            'training_observations':len(x),'training_trips':len(np.unique(groups)),
            'label_method':'monotone geometric alignment; derived proxy labels, not native GPS point annotations',
            'decoder':{'beta':35.0,'gap_seconds':30},'data_root':str(data.resolve()),
            'evaluator_partition_used':False}
    (ROOT/'artifacts'/'config.json').write_text(json.dumps(config,indent=2))
    (ROOT/'results'/'model_details.json').write_text(json.dumps(detail,indent=2))
    print('Experiments complete.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=ROOT.parent/'Student folder')
    parser.add_argument('--trials',type=int,default=10);parser.add_argument('--max-observations',type=int,default=350)
    parser.add_argument('--feature-version',type=int,choices=[1,2],default=1)
    parser.add_argument('--output',type=Path,help='Separate experiment directory; preserves deployed artifacts')
    args=parser.parse_args()
    if args.output: ROOT=args.output.resolve()
    run(args.data,args.trials,args.max_observations,args.feature_version)
