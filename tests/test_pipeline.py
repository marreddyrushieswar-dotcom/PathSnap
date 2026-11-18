import tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from pathsnap.data import load_trip, candidates, load_reference, feature_names
from pathsnap.decoder import decode, route_geometry


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.folder=Path(self.tmp.name)/'toy';self.folder.mkdir()
        np.savetxt(self.folder/'toy.nodes',[[8,50],[8.001,50],[8.002,50],[8.003,50]])
        np.savetxt(self.folder/'toy.arcs',[[0,1],[1,2],[2,3],[1,0],[2,1],[3,2]],fmt='%d')
        np.savetxt(self.folder/'toy.track',[[8.0002,50,1],[8.0008,50,2],[8.0018,50,3],[8.0028,50,4]])
    def tearDown(self): self.tmp.cleanup()
    def test_candidate_generation_needs_no_reference(self):
        trip=load_trip(self.folder); c=candidates(trip,6)
        self.assertEqual(c['ids'].shape,(4,6));self.assertTrue(np.isfinite(c['features']).all())
        self.assertEqual(set(c['ids'][0]),set(range(6)))
    def test_route_rows_are_aligned_to_observations(self):
        np.savetxt(self.folder/'toy.route',[0,1,2],fmt='%d')
        labels,route,residual=load_reference(self.folder,load_trip(self.folder))
        self.assertEqual(labels.tolist(),[0,0,1,2]);self.assertEqual(len(labels),4)
        self.assertLess(residual.max(),1e-4)
    def test_context_features_use_no_reference_and_stop_at_gaps(self):
        trip=load_trip(self.folder)
        trip.track[2:,2]+=100
        first=candidates(trip,6,feature_version=2)
        self.assertEqual(first['features'].shape,(4,6,len(feature_names(2))))
        self.assertTrue(np.isfinite(first['features']).all())
        # Moving the other side of a long gap must not change context before it.
        trip.gps[2:]+=10000
        second=candidates(trip,6,feature_version=2)
        np.testing.assert_allclose(first['features'][:2,:,12:],second['features'][:2,:,12:])
    def test_context_direction_distinguishes_opposite_arcs(self):
        trip=load_trip(self.folder)
        c=candidates(trip,6,feature_version=2)
        heading=feature_names(2).index('heading_cos_3s')
        forward=np.flatnonzero(c['ids'][1]==0)[0]
        reverse=np.flatnonzero(c['ids'][1]==3)[0]
        self.assertGreater(c['features'][1,forward,heading],.99)
        self.assertLess(c['features'][1,reverse,heading],-.99)
    def test_full_inference_runs_without_reference_files(self):
        from sklearn.dummy import DummyClassifier
        from pathsnap.service import predict
        model=DummyClassifier(strategy='prior').fit(np.zeros((2,len(feature_names(2)))),[0,1])
        config={'candidates':6,'feature_version':2,'decoder':{'transition_weight':.15}}
        with patch('pathsnap.service.assets',return_value=(config,model)):
            trip,ids,geometry,resets,arcs=predict(self.folder)
        self.assertEqual(len(ids),len(trip.track))
        self.assertEqual(len(arcs),len(trip.track))
        self.assertTrue(geometry)
    def test_weighted_viterbi_matches_exhaustive_sequence_search(self):
        from itertools import product
        trip=load_trip(self.folder)
        trip.track=trip.track[:3]
        c={'ids':np.array([[0,1],[0,1],[0,1]])}
        p=np.array([[.8,.2],[.35,.65],[.1,.9]])
        distances=np.array([[1.,7.],[5.,1.]])
        beta,weight=15.,.15
        transition=-np.abs(distances-1)/beta
        transition-=np.log(np.exp(transition).sum(axis=1))[:,None]
        def objective(sequence):
            return sum(np.log(p[t,col]) for t,col in enumerate(sequence))+sum(weight*transition[a,b] for a,b in zip(sequence[:-1],sequence[1:]))
        expected=max(product(range(2),repeat=3),key=objective)
        with patch('pathsnap.decoder.transition_distances',return_value=(distances,1.)):
            _,chosen,_=decode(trip,c,p,beta=beta,transition_weight=weight)
        self.assertEqual(chosen.tolist(),list(expected))
    def test_forward_direction_decodes_and_expands_connector(self):
        trip=load_trip(self.folder);c=candidates(trip,6)
        p=np.full((4,6),.001)
        for t,arc in enumerate([0,0,1,2]): p[t,np.flatnonzero(c['ids'][t]==arc)[0]]=1
        p/=p.sum(axis=1,keepdims=True)
        arcs,chosen,resets=decode(trip,c,p)
        self.assertEqual(arcs.tolist(),[0,0,1,2]);self.assertEqual(resets,[0])
        geometry,_=route_geometry(trip,c,chosen,resets)
        self.assertEqual(len(geometry),1);self.assertTrue(any(abs(point[1]-8.001)<1e-7 for point in geometry[0]))
    def test_gap_starts_separate_component(self):
        trip=load_trip(self.folder);trip.track[2:,2]+=100
        c=candidates(trip,6);p=np.ones((4,6))/6
        _,chosen,resets=decode(trip,c,p)
        self.assertIn(2,resets)
        geometry,_=route_geometry(trip,c,chosen,resets)
        self.assertGreaterEqual(len(geometry),2)
    def test_invalid_gps_is_removed_without_renumbering_arcs(self):
        with (self.folder/'toy.track').open('a') as f:f.write('nan 50 5\n')
        trip=load_trip(self.folder)
        self.assertEqual(len(trip.track),4);self.assertEqual(trip.issues['invalid_observations'],1)
        self.assertEqual(trip.original_rows.tolist(),[0,1,2,3])


if __name__=='__main__':unittest.main()
