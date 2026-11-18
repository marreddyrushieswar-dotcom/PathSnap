import tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from pathsnap.data import load_trip, candidates, load_reference
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
