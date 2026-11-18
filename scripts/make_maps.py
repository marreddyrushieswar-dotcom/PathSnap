"""Generate development-only Folium maps. Reference layers are never served by API."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vendor'))
import folium
from pathsnap.data import load_trip,load_reference


def main():
    config=json.loads((ROOT/'artifacts'/'config.json').read_text())
    folder=Path(config['data_root'])/'student_test'/'00000052'
    trip=load_trip(folder);_,route,_=load_reference(folder,trip)
    prediction=json.loads((ROOT/'results'/'predictions'/'00000052.json').read_text())
    center=trip.track[:,:2].mean(axis=0)[::-1].tolist()
    m=folium.Map(location=center,zoom_start=14,tiles='OpenStreetMap',control_scale=True)
    roads=folium.FeatureGroup(name='Road network',show=False)
    for arc in trip.arcs:
        folium.PolyLine(trip.nodes[arc,::-1].tolist(),color='#687787',weight=1,opacity=.4).add_to(roads)
    roads.add_to(m)
    gps=folium.FeatureGroup(name='GPS observations')
    for i,point in enumerate(trip.track[:,:2]):
        folium.CircleMarker(point[::-1].tolist(),radius=3,color='#1976d2',fill=True,fill_opacity=.7,tooltip=f'Observation {i}').add_to(gps)
    gps.add_to(m)
    reference=folium.FeatureGroup(name='Reference route (development only)')
    for arc in route:
        folium.PolyLine(trip.nodes[trip.arcs[arc],::-1].tolist(),color='#24a148',weight=6,opacity=.65).add_to(reference)
    reference.add_to(m)
    predicted=folium.FeatureGroup(name='Predicted HMM route')
    for component in prediction['predicted_components']:
        if len(component)>1:folium.PolyLine(component,color='#e87912',weight=3,opacity=.95).add_to(predicted)
    predicted.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds(trip.track[:,:2][:,::-1].tolist())
    legend='<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:12px;border:1px solid #bbb;font:13px Arial"><b>PathSnap | Student trip 00000052</b><br><span style="color:#1976d2">Blue: GPS</span><br><span style="color:#24a148">Green: reference</span><br><span style="color:#e87912">Orange: prediction</span></div>'
    m.get_root().html.add_child(folium.Element(legend))
    output=ROOT/'reports'/'development_folium_map.html';m.save(str(output));print(output)


if __name__=='__main__':main()
