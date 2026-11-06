import unittest
import numpy as np
import pandas as pd
import tempfile
import json
from pathlib import Path
from si_support.analysis import histogram_distribution,change_by_unit,node_participation,partition_similarity,diagnostic_frames
from si_support.core import divide,Context


class ScientificAccountingTests(unittest.TestCase):
    def test_zero_atom_is_not_multiplied_by_bin_count(self):
        z=pd.DataFrame(dict(year=[2020]*2,international=[0]*2,metric=['joint_cross']*2,
                            bin=[0,1],lo=[0,.5],hi=[.5,1],n_positive=[2,3],n_zero=[5,5],n_total=[10,10]))
        cdf,q=histogram_distribution(z)
        np.testing.assert_allclose(cdf.cdf,[.7,1.]);self.assertEqual(q.zero_fraction.iloc[0],.5)
        self.assertEqual(q.median_upper.iloc[0],0)

    def test_histogram_count_mismatch_is_rejected(self):
        z=pd.DataFrame(dict(year=[2020],international=[0],metric=['joint_cross'],bin=[0],lo=[0],hi=[1],n_positive=[2],n_zero=[5],n_total=[10]))
        with self.assertRaises(ValueError):histogram_distribution(z)

    def test_within_composition_entry_exit_close_exactly(self):
        old=pd.DataFrame({'country':[0,1],'field':[0,0],'E_asm':[6.,4.],'B_asm':[3.,1.]})
        new=pd.DataFrame({'country':[0,2],'field':[0,0],'E_asm':[5.,5.],'B_asm':[2.,4.]})
        z=change_by_unit(old,new)
        self.assertAlmostEqual(z.total.sum(),.2)
        self.assertAlmostEqual(z.entry.sum(),.4);self.assertAlmostEqual(z.exit.sum(),-.1)
        self.assertTrue(np.isnan(z.loc[z.country==2,'R0']).all())

    def test_duplicate_units_rejected(self):
        z=pd.DataFrame({'country':[0,0],'field':[0,0],'E_asm':[1,1],'B_asm':[.2,.3]})
        with self.assertRaises(ValueError):change_by_unit(z,z)

    def test_participation_respects_direction_and_removes_diagonal(self):
        W=np.array([[100.,1.,1.],[0.,0.,0.],[0.,0.,0.]])
        po,ro,pi,ri=node_participation(W,np.array([0,0,1]))
        self.assertAlmostEqual(po[0],.5)
        self.assertTrue(np.isnan(pi[0]));self.assertEqual(pi[1],0);self.assertEqual(pi[2],0)
        W[0,0]=0;po2,_,_,_=node_participation(W,np.array([0,0,1]))
        np.testing.assert_allclose(po,po2,equal_nan=True)

    def test_unknown_partition_neighbors_have_explicit_retained_mass(self):
        W=np.array([[0,1,3],[0,0,0],[0,0,0]],float)
        po,ret,_,_=node_participation(W,np.array([0,1,-1]))
        self.assertEqual(po[0],0);self.assertAlmostEqual(ret[0],.25)

    def test_ami_is_label_invariant_and_ignores_unassigned(self):
        ami,vi,n=partition_similarity([0,0,1,1,-1],[9,9,5,5,3])
        self.assertAlmostEqual(ami,1);self.assertAlmostEqual(vi,0);self.assertEqual(n,4)

    def test_undefined_denominator_is_not_zero_score(self):
        z=divide([0,2,0],[0,4,3]);self.assertTrue(np.isnan(z[0]));self.assertEqual(z[1],.5);self.assertEqual(z[2],0)

    def test_optional_tables_are_removed_before_a_new_analysis(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'results/tables').mkdir(parents=True)
            (root/'config.json').write_text(json.dumps({'output_dir':'results'}))
            pd.DataFrame({'country':[0],'country_code':['AA']}).to_csv(root/'results/tables/countries.csv',index=False)
            pd.DataFrame({'field':[0],'field_label':['Field']}).to_csv(root/'results/tables/fields.csv',index=False)
            ctx=Context(root/'config.json');ctx.write('S25_old',pd.DataFrame({'x':[1]}),'old','old')
            ctx.begin_tables();self.assertFalse((ctx.d['tables']/'S25_old.csv').exists())
            self.assertEqual(ctx.inventory,{})

    def test_reusing_tables_rejects_different_threshold(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'results/tables').mkdir(parents=True)
            (root/'config.json').write_text(json.dumps({'output_dir':'results'}))
            pd.DataFrame({'country':[0],'country_code':['AA']}).to_csv(root/'results/tables/countries.csv',index=False)
            pd.DataFrame({'field':[0],'field_label':['Field']}).to_csv(root/'results/tables/fields.csv',index=False)
            ctx=Context(root/'config.json');ctx.begin_tables();ctx.finish()
            other=Context(root/'config.json',min_exposure=100)
            with self.assertRaises(ValueError):other.validate_table_reuse()

    def test_population_diagnostics_batches_preserve_core_and_eligible_denominators(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'results/tables').mkdir(parents=True)
            (root/'config.json').write_text(json.dumps({'output_dir':'results'}))
            pd.DataFrame({'country':[0],'country_code':['AA']}).to_csv(root/'results/tables/countries.csv',index=False)
            pd.DataFrame({'field':[0],'field_label':['Field']}).to_csv(root/'results/tables/fields.csv',index=False)
            ctx=Context(root/'config.json')
            a=pd.DataFrame({'paper_id':np.arange(1,7,dtype=np.uint64)+2**53,'year':[1990,1990,1991,0,1991,1991],
              'eligible':[True,False,True,False,True,False],'known_fields':[1,0,2,0,1,0],
              'known_countries':[1,0,2,0,1,1],'reference_coverage':[.5,0,.9,0,1,0],'joint_cross':[.2,np.nan,.3,np.nan,.4,np.nan]})
            kinds=np.array([0,2,1,2,1,2]);ids=a.paper_id.to_numpy()
            diagnostic_frames(ctx,[a.iloc[:3],a.iloc[3:]],kinds,ids,expected_rows=6)
            labels=ctx.table('S33_full_label_diagnostics');self.assertEqual(labels[labels.metric=='known_fields'].n.sum(),5)
            coverage=ctx.table('S34_full_coverage_diagnostics');self.assertEqual(coverage['count'].sum(),3)
            types=ctx.table('S35_source_type_diagnostics');self.assertAlmostEqual(types.loc[types.kind==1,'mean_q'].iloc[0],.35)
            with self.assertRaises(ValueError):diagnostic_frames(ctx,[a],kinds,ids[::-1],expected_rows=6)


if __name__=='__main__':unittest.main()
