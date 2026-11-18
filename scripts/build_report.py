"""Build the report from executed result files. Requires ReportLab and Pillow."""
from pathlib import Path
import csv,json,math,statistics,sys
from collections import defaultdict
from html import escape
from PIL import Image,ImageDraw,ImageFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image as RLImage,PageBreak,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4,landscape

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports';OUT.mkdir(exist_ok=True)


def rows(file):
    with (ROOT/'results'/file).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def font(size,bold=False):
    path=Path('C:/Windows/Fonts')/('arialbd.ttf' if bold else 'arial.ttf')
    return ImageFont.truetype(str(path),size)


def chart(title,labels,series,names,filename,maximum=None):
    im=Image.new('RGB',(1400,650),'white');d=ImageDraw.Draw(im)
    d.text((55,25),title,fill='#132944',font=font(29,True))
    left,top,width,height=120,105,1200,400
    maximum=maximum or max(max(x) for x in series)*1.15
    for tick in range(6):
        y=top+height-height*tick/5
        d.line([(left,y),(left+width,y)],fill='#dce3ec',width=1)
        d.text((20,y-12),f'{maximum*tick/5:.2f}',fill='#506070',font=font(19))
    palette=['#148c8c','#d68431','#637bc5']
    group=width/len(labels);bar=group*.72/len(series)
    for j,values in enumerate(series):
        for i,value in enumerate(values):
            x=left+group*i+group*.14+bar*j;y=top+height-height*value/maximum
            d.rectangle((x,y,x+bar-5,top+height),fill=palette[j%len(palette)])
            d.text((x,y-23),f'{value:.3f}',fill='#20354b',font=font(18))
    for i,label in enumerate(labels):
        words=label.replace('HistGradientBoosting','Hist gradient\nboosting').replace('LogisticRegression','Logistic\nregression').replace('RandomForest','Random\nforest').replace('DecisionTree','Decision\ntree').replace('GaussianNB','Gaussian\nNB')
        d.multiline_text((left+group*i+12,top+height+14),words,fill='#243b53',font=font(20),spacing=5)
    for j,name in enumerate(names):
        x=left+j*380;d.rectangle((x,603,x+20,623),fill=palette[j%len(palette)]);d.text((x+30,600),name,fill='#243b53',font=font(19))
    im.save(OUT/filename)


def trajectory_image(prediction,trip_nodes,arcs,route):
    im=Image.new('RGB',(1400,850),'#f5f8fb');d=ImageDraw.Draw(im)
    gps=prediction['gps'];points=gps+sum(prediction['predicted_components'],[])
    lat=[p[0] for p in points];lon=[p[1] for p in points]
    minlat,maxlat=min(lat),max(lat);minlon,maxlon=min(lon),max(lon)
    cosine=math.cos((minlat+maxlat)*math.pi/360)
    scale=min(1200/max((maxlon-minlon)*cosine,1e-8),650/max(maxlat-minlat,1e-8))
    def xy(p):return (700+(p[1]-(minlon+maxlon)/2)*cosine*scale,440-(p[0]-(minlat+maxlat)/2)*scale)
    for u,v in arcs:
        a,b=trip_nodes[int(u)],trip_nodes[int(v)]
        if minlon-.001<=a[0]<=maxlon+.001 and minlat-.001<=a[1]<=maxlat+.001:
            d.line([xy(a[::-1]),xy(b[::-1])],fill='#ccd5df',width=1)
    for arc in route:
        u,v=arcs[int(arc)];d.line([xy(trip_nodes[int(u)][::-1]),xy(trip_nodes[int(v)][::-1])],fill='#31a354',width=9)
    for component in prediction['predicted_components']:
        if len(component)>1:d.line([xy(p) for p in component],fill='#ee8d27',width=4)
    for p in gps:
        x,y=xy(p);d.ellipse((x-3,y-3,x+3,y+3),fill='#277ec1')
    d.text((45,22),'Student trip 00000052 | reference and reconstructed route',font=font(29,True),fill='#16324f')
    d.text((45,775),'Blue: GPS     Green: supplied reference     Orange: HMM prediction',font=font(23),fill='#243b53')
    d.text((45,815),'Static geographic plot from actual outputs; interactive Folium view supplied separately.',font=font(18),fill='#506070')
    im.save(OUT/'trajectory.png')


def main():
    comparison=rows('model_comparison.csv');decoder=rows('decoder_comparison.csv');inventory=rows('dataset_inventory.csv')
    config=json.loads((ROOT/'artifacts'/'config.json').read_text());details=json.loads((ROOT/'results'/'model_details.json').read_text())
    trials=json.loads((ROOT/'results'/'bayesian_trials.json').read_text())
    labels=[r['model'] for r in comparison]
    chart('Trip-grouped cross-validation',labels,[[float(r['cv_macro_f1']) for r in comparison]],['Observation-level macro F1'],'cv_comparison.png',1)
    chart('Student-test classification: all 19,160 observations',labels,[[float(r['top1_accuracy']) for r in comparison],[float(r['macro_f1']) for r in comparison]],['Top-1 accuracy','Mean per-trip macro F1'],'student_comparison.png',1)
    chart('Student-test probability quality (lower is better)',labels,[[float(r['log_loss']) for r in comparison]],['Candidate-normalized log loss'],'logloss_comparison.png')
    chart('Bayesian search: measured CV objective',[str(i+1) for i in range(len(trials))],[[t['cv_macro_f1'] for t in trials]],['Macro F1 per trial'],'bayesian_search.png',1)
    import numpy as np
    data=Path(config['data_root'])/'student_test'/'00000052'
    nodes=np.loadtxt(data/'00000052.nodes');arcs=np.loadtxt(data/'00000052.arcs',dtype=int);route=np.loadtxt(data/'00000052.route',dtype=int)
    prediction=json.loads((ROOT/'results'/'predictions'/'00000052.json').read_text());trajectory_image(prediction,nodes,arcs,route)
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='SmallNote',fontName='Helvetica',fontSize=8.5,leading=12,textColor=colors.HexColor('#52667c')))
    styles['BodyText'].fontSize=9.5;styles['BodyText'].leading=14;styles['BodyText'].spaceAfter=8
    styles['Heading1'].textColor=colors.HexColor('#153650');styles['Heading1'].fontSize=21;styles['Heading1'].spaceAfter=15
    styles['Heading2'].textColor=colors.HexColor('#147a7a');styles['Heading2'].spaceBefore=15
    story=[]
    def p(text,style='BodyText'):story.append(Paragraph(text,styles[style]))
    def h(text):p(text,'Heading1')
    def image(name,width=490):
        im=Image.open(OUT/name);story.append(RLImage(str(OUT/name),width=width,height=width*im.height/im.width));story.append(Spacer(1,8))
    def table(data,widths=None,size=8):
        items=[[Paragraph(escape(str(cell)),ParagraphStyle(name='cell',fontSize=size,leading=size+3,textColor=colors.white if i==0 else colors.HexColor('#23384d'))) for cell in row] for i,row in enumerate(data)]
        t=Table(items,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#153650')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f5f8'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),1,colors.HexColor('#148c8c'))]));story.append(t);story.append(Spacer(1,10))
    def page():story.append(PageBreak())
    p('MACHINE LEARNING HACKATHON / DEVELOPMENT BUILD','SmallNote');story.append(Spacer(1,20));h('PathSnap')
    p('Distributed map matching &amp; route reconstruction','Title');story.append(Spacer(1,18))
    p('Technical report | 18 November 2026','Heading2')
    p('This report records experiments actually executed on the supplied 80 training and 10 student-test trips. It includes five trained models, ten Bayesian optimisation trials, and classifier-versus-HMM results for every student-test trip.')
    p('<b>Scope:</b> reserved evaluator trips were not provided or accessed. A second laptop and public GitHub publication have not been demonstrated in this build. Socket access is restricted in the execution environment; API validation calls synchronous FastAPI handlers directly. HTTP transport and event-loop startup remain unverified. These external deployment requirements remain open.')
    p('<b>Reference-label limitation:</b> .route files contain traversed arcs, rather than native per-GPS labels. All pointwise metrics use a monotone geometric alignment derived from those routes. These are proxy scores, not independently verified GPS-level ground truth.')
    image('trajectory.png')
    p('AI assistance was used for implementation and report drafting. All numerical experiment results originate from executed source code and retained result files. The team should review and explain every claim before submission. Team name, members and repository URL must be supplied by the team.','SmallNote')
    page();h('2. Dataset understanding and EDA')
    table([['Partition','Trips','GPS observations','Use'],['Training',80,sum(int(r['observations']) for r in inventory if r['split']=='train'),'Feature preparation, grouped CV and fitting'],['Student testing',10,sum(int(r['observations']) for r in inventory if r['split']=='student_test'),'Development comparison and decoder evaluation'],['Evaluator testing','Not supplied','Not accessed','Final evaluation only']],[100,55,110,225])
    p('Each trip is independently indexed. .nodes rows contain longitude/latitude; .arcs rows contain directed from/to node IDs, and row index is arc ID. .track rows contain longitude, latitude and timestamp. .route rows are an ordered sequence of traversed arc IDs. Identical numerical arc IDs in different trips do not represent the same semantic class.')
    p('Preprocessing','Heading2')
    p('The loader validates file shapes, node coordinates and arc endpoints; removes invalid GPS coordinates; preserves original observation indices; and stable-sorts timestamps. Duplicate timestamps are retained with guarded time denominators. Road IDs are never renumbered. Coordinates are projected into local metres using the network mean latitude. This local approximation is appropriate for the small network extents, but is not a globally accurate projection.')
    for split in ['train','student_test']:
        subset=[r for r in inventory if r['split']==split];count=sum(int(r['observations']) for r in subset)
        steps=sum(int(r['total_steps']) for r in subset);regular=sum(int(r['one_second_steps']) for r in subset)
        p(f'<b>{split}:</b> {count:,} valid observations; trip lengths {min(int(r["observations"]) for r in subset):,} to {max(int(r["observations"]) for r in subset):,}; {sum(int(r["invalid_observations"]) for r in subset)} invalid GPS rows; {sum(int(r["duplicate_timestamps"]) for r in subset)} duplicate timestamps; {sum(int(r["gaps_over_10s"]) for r in subset)} gaps over 10 seconds; {100*regular/max(steps,1):.2f}% of steps exactly one second.')
    p(f'The full inventory records file sizes, geographic bounds, network counts, weakly connected components, isolated nodes, zero-length arcs and reference-alignment residuals for all 90 trips. Median per-trip reference residual: {statistics.median(float(r["reference_residual_median_m"]) for r in inventory):.2f} m. Networks contain {min(int(r["nodes"]) for r in inventory):,} to {max(int(r["nodes"]) for r in inventory):,} nodes.')
    p('Patterns and limitations','Heading2');p('Nominal one-second sampling is interrupted by gaps, and noisy observations can lie near parallel roads or both directions of the same road. Per-trip networks have different sizes and connectivity. Disconnected components and local arc numbering make direct global arc-ID classification inappropriate. The chosen approach learns a transferable candidate-match function.')
    page();h('2.1 Features and labels')
    table([['Feature family','Features','Reason'],['Spatial fit','Distance in metres; distance rank; projection fraction','Measures proximity to each directed segment'],['Movement alignment','Heading cosine; absolute heading sine; speed; time interval','Distinguishes direction, turns, noise and gaps'],['Road structure','Segment length; source/target node degree','Context at intersections and short road segments'],['Neighbour observations','Distance of previous and next GPS observations to candidate','Local trajectory support; offline bidirectional features']],[100,195,195])
    p('This is an offline reconstruction system: next-observation features require the complete trajectory and do not support strict real-time streaming without modification. KD-tree segment samples are spaced at most 30 m, followed by exact point-to-segment reranking. Twelve directed candidates are retained. Shortlist retrieval is approximate in dense networks; misses are reported rather than removed from scoring.')
    p('The reference sequence is aligned through a dynamic program whose states are positions in the supplied ordered route. Each GPS point incurs squared distance to a route segment; transitions permit remaining at the current position or advancing. A prefix minimum gives O(observations x route length) time. The resulting arc label is used only for training/evaluation. Prediction functions never load the reference file.')
    p('No manual point annotations are supplied. Geometric alignment can be ambiguous around stops, revisits or intersections, and uses information similar to some predictive features. Pointwise performance should therefore be interpreted as agreement with this derived reference. Route arc-set F1 gives a separate, order-insensitive route coverage measure.')
    p('Training design','Heading2');p(f'All 80 training trips participate, with at most {config["maximum_training_observations_per_trip"]} randomly sampled observations per trip: {config["training_observations"]:,} observations and {config["training_observations"]*12:,} candidate rows. This caps computation and gives every trip similar influence; some temporal detail is omitted during fitting. The student test uses every valid observation. Seed {config["seed"]}; three GroupKFold splits preserve complete trips. Scaling is fitted inside each training fold.')
    p('Primary selection uses the mean per-trip observation-level macro F1 on grouped validation folds. Arc IDs are evaluated within each trip; scores are then averaged. Accuracy and log loss pool observations. Candidate misses count as errors; missing true candidates receive epsilon probability for log loss. CV operates on the sampled training observations, so its coverage differs from full student-test evaluation.')
    page();h('3. Five-model comparison')
    table([['Model','CV macro F1','Student accuracy','Student macro F1','Log loss'],*[[r['model'],f"{float(r['cv_macro_f1']):.4f}",f"{float(r['top1_accuracy']):.4f}",f"{float(r['macro_f1']):.4f}",f"{float(r['log_loss']):.4f}"] for r in comparison]],[155,85,85,85,80])
    image('cv_comparison.png');image('student_comparison.png')
    page();h('3.1 Efficiency and probability quality')
    table([['Model','Train + CV (s)','Score all student candidates (s)','Continuity','CV std'],*[[r['model'],f"{float(r['training_and_cv_seconds']):.3f}",f"{float(r['inference_seconds']):.4f}",f"{float(r['route_continuity']):.4f}",f"{float(r['cv_std']):.4f}"] for r in comparison]],[150,85,105,80,70])
    image('logloss_comparison.png')
    p(f'All classifiers share candidate generation, with a student-test candidate miss rate of {100*float(comparison[0]["candidate_miss_rate"]):.4f}%. Inference timing above includes classifier scoring only, excluding file loading, features, shortest-path decoding and API overhead. Route continuity is mean per-trip directed reachability within the decoder search bound.')
    p('Histogram gradient boosting has the strongest baseline grouped CV F1 and student-test probability quality. Random forest is close in F1 but takes longer to fit and score. The observed score difference is small relative to fold variability; no significance claim is made. The selected algorithm is chosen from CV rather than student-test accuracy.')
    explanations={
    'LogisticRegression':('Linear / probabilistic','StandardScaler; C=1; balanced class weights; max_iter=400','Weighted logistic loss with L2 regularisation','A fast, interpretable baseline for spatial and heading evidence. Linear boundaries can miss interactions between road geometry and local motion.'),
    'GaussianNB':('Probabilistic','StandardScaler; var_smoothing=1e-5','Gaussian class-conditional likelihood with naive feature independence','Fast fitting and scoring with few parameters. Distance, rank and neighbour distances are correlated, violating the conditional-independence assumption.'),
    'DecisionTree':('Tree','depth=12; min_samples_leaf=12; balanced weights','Weighted Gini impurity reduction','Captures nonlinear feature thresholds and interactions. One tree is unstable and can overfit local geometric patterns.'),
    'RandomForest':('Bagged ensemble','60 trees; depth=15; min_samples_leaf=5; balanced weights','Average of bootstrapped Gini trees','Averaging improves robustness and captures nonlinear patterns. It costs more CPU and storage and may still overfit frequent trajectory patterns.'),
    'HistGradientBoosting':('Boosting','85 iterations; 23 leaves; learning_rate=0.08; L2=1','Binary log loss with histogram-based gradient boosting','Models nonlinear interactions efficiently and gives useful candidate scores. Hyperparameters and score calibration require care; direction ambiguity remains.')}
    for r in comparison:
        page();name=r['model'];family,parameters,loss,reason=explanations[name];h('3.2 '+name)
        p(f'<b>Family:</b> {family}. <b>Why suitable:</b> {reason}')
        p(f'<b>Initial configuration:</b> {parameters}. <b>Objective:</b> {loss}. The positive label means this candidate matches the aligned reference arc. Probabilities are normalized over the observation candidate set.')
        table([['Measured quantity','Result'],['Training macro F1',f"{float(r['train_macro_f1']):.4f}"],['Grouped CV macro F1 +/- std',f"{float(r['cv_macro_f1']):.4f} +/- {float(r['cv_std']):.4f}"],['Student accuracy',f"{float(r['top1_accuracy']):.4f}"],['Student macro F1',f"{float(r['macro_f1']):.4f}"],['Student macro precision / recall',f"{float(r['precision']):.4f} / {float(r['recall']):.4f}"],['Student log loss',f"{float(r['log_loss']):.4f}"],['Candidate-level macro F1',f"{float(r['candidate_macro_f1']):.4f}"],['Classifier route continuity',f"{float(r['route_continuity']):.4f}"]],[260,230])
        gap=float(r['train_macro_f1'])-float(r['cv_macro_f1'])
        p(f'Training minus grouped-validation macro F1 is {gap:.4f}. '+('This positive gap is evidence of some overfitting, although split sampling and trajectory differences also matter.' if gap>.025 else 'The small gap provides limited evidence of overfitting at this configuration. It does not establish the absence of bias or underfitting.'))
        matrix=details[name]['confusion_matrix_candidate'];table([['Candidate confusion matrix','Predicted other','Predicted selected'],['True other',matrix[0][0],matrix[0][1]],['True aligned match',matrix[1][0],matrix[1][1]]],[220,135,135])
        p('This confusion table is a candidate-level diagnostic after taking one winner per GPS observation. Candidate misses can create observations with no positive candidate label. It is distinct from the per-trip multiclass arc metrics. No artificial epoch-wise loss curves are generated for these estimators.')
    page();h('4. Bayesian hyperparameter optimisation')
    p('The selected histogram gradient boosting model is tuned with a Gaussian-process surrogate (Matern kernel plus noise) and expected improvement. The first three trials initialize randomly; seven subsequent trials maximize expected improvement over 512 randomly proposed candidates. Ten measured evaluations use the same three grouped CV folds; the retained search history includes parameters, fold scores, standard deviations and proposal coordinates.')
    table([['Parameter','Search range'],['learning_rate','10**(-2 + 1.4*u), u in [0,1): approximately 0.01 to 0.2512'],['max_leaf_nodes','int(8 + 40*v), v in [0,1): 8 to 47'],['Objective','Maximize mean grouped-validation observation macro F1']],[160,330])
    p(f'Best parameters: learning_rate={config["bayesian_parameters"].get("learning_rate")}; max_leaf_nodes={config["bayesian_parameters"].get("max_leaf_nodes")}. Search time {config["bayesian_search_seconds"]:.2f} seconds; refit {config["refit_seconds"]:.2f} seconds. Other estimator settings are held fixed.')
    baseline=next(r for r in comparison if r['model']==config['selected_model']);tuned=config['tuned_student_results']
    table([['Metric','Untuned','Tuned'],['CV macro F1',f"{float(baseline['cv_macro_f1']):.4f}",f"{config['bayesian_cv_macro_f1']:.4f}"],['Student accuracy',f"{float(baseline['top1_accuracy']):.4f}",f"{tuned['top1_accuracy']:.4f}"],['Student macro F1',f"{float(baseline['macro_f1']):.4f}",f"{tuned['macro_f1']:.4f}"],['Student log loss',f"{float(baseline['log_loss']):.4f}",f"{tuned['log_loss']:.4f}"],['Student inference (s)',f"{float(baseline['inference_seconds']):.4f}",f"{tuned['inference_seconds']:.4f}"]],[230,130,130])
    p('The tuned model marginally improves CV and student-test F1 and log loss. The CV gain is much smaller than fold variability, so the evidence supports only a small measured improvement. The final refit uses all sampled training trips, not student-test labels.')
    image('bayesian_search.png')
    page();h('5. HMM and Viterbi reconstruction')
    p('Hidden states are directed road-segment candidates at each observation. Emissions are normalized classifier-positive scores, floored before taking logarithms. For transitions, the route distance is the remaining length on the previous arc, the directed shortest path from its end to the current arc start, and the projected distance along the current arc. A forward move on the same arc uses direct projected progress. A 2% backward-fraction tolerance absorbs small GPS noise.')
    p('The unnormalized log transition score is -abs(network distance - observed GPS displacement)/35. Scores are normalized over reachable next candidates for each source state. Viterbi uses delta[t,j] = log emission[t,j] + max_i(delta[t-1,i] + log transition[i,j]); backpointers recover the state sequence. Calculations stay in log space.')
    p('Each shortest-path query is bounded by max(500 m, 4*GPS step+200 m, min(70*dt, 20 km)). This limits CPU use and can classify a long detour as unreachable. Gaps over 30 seconds or steps with no feasible candidate pair reset the decoder. Invalid GPS rows are removed; valid points always get network candidates, including distant ones. The system does not fabricate bridges across disconnected components. Geometry expands actual network connectors between selected arcs.')
    table([['Trip','Raw acc.','HMM acc.','Raw F1','HMM F1','Raw cont.','HMM cont.'],*[[r['trip_id'],f"{float(r['accuracy']):.3f}",f"{float(decoder[i+1]['accuracy']):.3f}",f"{float(r['macro_f1']):.3f}",f"{float(decoder[i+1]['macro_f1']):.3f}",f"{float(r['route_continuity']):.3f}",f"{float(decoder[i+1]['route_continuity']):.3f}"] for i,r in enumerate(decoder) if r['variant']=='classifier']],[70,70,70,70,70,70,70],7)
    raw=[r for r in decoder if r['variant']=='classifier'];hmm=[r for r in decoder if r['variant']=='hmm'];counts={r['trip_id']:int(r['observations']) for r in inventory if r['split']=='student_test'}
    def avg(rs,key):return statistics.mean(float(r[key]) for r in rs)
    def weighted(rs,key):return sum(float(r[key])*counts[r['trip_id']] for r in rs)/sum(counts.values())
    p(f'Mean per-trip macro F1 improves from {avg(raw,"macro_f1"):.4f} to {avg(hmm,"macro_f1"):.4f}; continuity from {avg(raw,"route_continuity"):.4f} to {avg(hmm,"route_continuity"):.4f}. Observation-weighted accuracy changes from {weighted(raw,"accuracy"):.4f} to {weighted(hmm,"accuracy"):.4f}. Mean route arc-set F1 changes from {avg(raw,"route_arc_set_f1"):.4f} to {avg(hmm,"route_arc_set_f1"):.4f}.')
    p('HMM macro F1 improves for these ten trips, while accuracy falls on trips 00000007, 00000065, 00000077 and 00000091, as well as slightly on 00000000. This demonstrates a tradeoff: a more connected and balanced route can disagree with more individual proxy labels. Fixed transition scale and geometric-label ambiguity are plausible causes, but no causal ablation was performed. Decoder constants are documented defaults rather than a separately tuned optimum.')
    page();h('6. Geographical visualisation')
    image('trajectory.png')
    p('The representative trip is student-test trip 00000052. GPS, unlabelled road geometry, supplied reference route and HMM prediction are visualised using the exact data and saved predictions. The reference layer appears only in this development analysis; it is never returned to the evaluator UI. The interactive Folium map includes layer controls, distinct colours, observation tooltips and bounds fitted to the trip.')
    if (OUT/'folium_screenshot.png').exists():image('folium_screenshot.png')
    else:p('Interactive map: reports/development_folium_map.html. A rendered browser screenshot has not yet been included in this report.','SmallNote')
    page();h('7. API, UI and deployment')
    p('FastAPI exposes GET /health, GET /api/trips and POST /api/predict with a validated trip_id. The server loads model artifacts, generates candidates, scores them, decodes the route, then invokes a separate reference scorer. Responses contain GPS observations, predicted geometry, unlabelled roads and aggregate accuracy. Reference labels and reference geometry stay server-side.')
    p('The evaluator interface can run on Laptop 2 and send requests to Laptop 1. It includes an API URL field, trip selector, status/error messages, metrics, layer switches and a geographic plot that requires no external map tiles. The same interface can also be served by the API. Until reserved data is supplied, it explicitly shows Student demo mode and lists only actual student-test trip IDs.')
    evidence_path=ROOT/'results'/'api_validation.json'
    if evidence_path.exists():
        evidence=json.loads(evidence_path.read_text());p(f'Executed API validation: {evidence["successful_trips"]} successful predictions through direct synchronous FastAPI handler calls, with schema and reference-label exclusion checks. Health and trip discovery passed; invalid trip IDs and path traversal requests were rejected. Socket restrictions also block TestClient event-loop startup. This validates handler logic, not HTTP transport, LAN connectivity or a receiving-laptop demonstration.')
    else:p('API source is provided. A complete integration record is still pending.')
    if (OUT/'ui_screenshot.png').exists():image('ui_screenshot.png')
    p('<b>Deployment limitations:</b> the execution sandbox prevents socket access. A live browser-to-API connection and two-laptop request-log screenshot could not be verified here. Start the supplied scripts outside this restricted execution session on the demonstration laptops. Reserved evaluator data is organiser-held and was not supplied; no performance on those trips is claimed. The public GitHub repository has not been published.')
    p('Reproduction','Heading2');p('Install requirements with Python 3.11; run python -m pathsnap.experiment --data PATH --trials 10; run unittest discovery; start uvicorn pathsnap.api:app on Laptop 1; and serve ui/ on Laptop 2. See README.md for exact PowerShell commands and evaluator-root configuration. Keep raw evaluator references only on Laptop 1.')
    page();h('Evidence index and remaining submission work')
    table([['Evidence','File'],['All-trip EDA inventory','results/dataset_inventory.csv'],['Five-model comparison','results/model_comparison.csv'],['Configurations and diagnostics','results/model_details.json'],['Grouped fold assignments','results/cv_splits.json'],['Ten Bayesian trials','results/bayesian_trials.json'],['HMM vs classifier on every student trip','results/decoder_comparison.csv'],['Saved predictions','results/predictions/*.json'],['Final model and settings','artifacts/final_model.joblib; artifacts/config.json'],['Executed environment','results/environment.txt'],['Unit and API checks','tests/; results/api_validation.json'],['Development Folium map','reports/development_folium_map.html'],['Source attribution','vendor/SOURCES.json and accompanying licenses']],[190,300])
    p('Remaining work: add team identity; review the methodology and explain derived labels; publish the prepared source and artifacts to the correct public GitHub repository; run the API and UI across two actual laptops; capture receiving-laptop request logs; and load the ten reserved evaluator trips only during the permitted final evaluation. No external submission or reserved evaluation has been performed.')
    p('The development package is an executed and documented baseline. Its main limitations are proxy point labels, approximate shortlisting, sampled fitting observations, uncalibrated candidate-normalized scores and fixed decoder settings. Further improvement should focus on independently checked alignment, targeted error analysis, probability calibration and decoder ablations using permitted development partitions.')
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#d4e0e8'));canvas.line(45,42,550,42);canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#60788c'));canvas.drawString(45,28,'PathSnap | Executed development experiments | 18 Nov 2026');canvas.drawRightString(550,28,str(doc.page))
    doc=SimpleDocTemplate(str(OUT/'PathSnap_Technical_Report.pdf'),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=45,bottomMargin=55,title='PathSnap Technical Report',author='PathSnap development build')
    doc.build(story,onFirstPage=footer,onLaterPages=footer);print(OUT/'PathSnap_Technical_Report.pdf')


if __name__=='__main__':main()
