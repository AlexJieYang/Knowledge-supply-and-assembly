from pathlib import Path
import numpy as np
import pandas as pd
import nbformat
from assembly.synthetic import create,configuration
from assembly.runtime import run,dirs


def test_small_pipeline_conservation_missingness_and_sampling(tmp_path):
    create(tmp_path/'input',n=900)
    c=configuration(tmp_path/'input',tmp_path/'result');c['plot_sample_n']=300;c['r_export_fraction']=.5
    for stage in ['metadata','references','metrics','networks','descriptions']:run(c,stage,force=True)
    d=dirs(c);papers=pd.read_parquet(d['data']/'assembly_papers.parquet');star=papers[papers.eligible]
    assert len(papers)==899 and (papers.year==0).sum()==1 and papers.year.max()==2024
    assert star.joint_refs.ge(2).all() and star.known_countries.gt(0).all() and star.known_fields.gt(0).all()
    assert star.completion_lower.le(star.joint_cross+1e-6).all() and star.completion_upper.ge(star.joint_cross-1e-6).all()
    annual=pd.read_csv(d['tables']/'network_annual.csv')
    np.testing.assert_allclose(annual.N.sum(),len(star));np.testing.assert_allclose(annual.H.sum(),star.joint_cross.sum(),rtol=1e-7)
    check=pd.read_csv(d['tables']/'conservation_checks.csv');np.testing.assert_allclose(check.actual,check.expected,atol=1e-5,rtol=1e-7)
    plot=pd.read_parquet(d['data']/'plot_sample.parquet');assert len(plot)==min(300,len(star)) and plot.paper_id.is_unique
    r=pd.read_parquet(d['data']/'r_papers.parquet');assert set(r.paper_id)<=set(star.paper_id) and (r.sample_weight==2).all()
    h=pd.read_csv(d['tables']/'histograms.csv');joint=h[h.metric.eq('joint_cross')]
    counts=joint.groupby(['year','international']).agg(positive=('n_positive','sum'),zero=('n_zero','first'),total=('n_total','first'))
    np.testing.assert_array_equal(counts.positive+counts.zero,counts.total)
    decomposition=pd.read_csv(d['tables']/'change_accounting.csv');np.testing.assert_allclose(decomposition.value.sum(),decomposition.total_difference.iloc[0],atol=1e-10)
    fixed=pd.read_csv(d['tables']/'fixed_support_mean.csv');assert len(fixed)==0 or fixed.year.min()>=c['fixed_support_start']
    assert len(list(d['data'].glob('*.parquet')))==4


def test_notebook_structure_languages_and_clean_outputs():
    root=Path(__file__).resolve().parents[1];files=sorted(root.glob('*.ipynb'));assert len(files)==8
    for p in files:
        nb=nbformat.read(p,as_version=4);nbformat.validate(nb)
        assert all(not cell.get('outputs',[]) for cell in nb.cells if cell.cell_type=='code')
        if p.name.startswith('06_'):assert nb.metadata.kernelspec.language=='R'
        else:
            assert nb.metadata.kernelspec.language=='python'
            for cell in nb.cells:
                if cell.cell_type=='code':compile(cell.source,str(p),'exec')
