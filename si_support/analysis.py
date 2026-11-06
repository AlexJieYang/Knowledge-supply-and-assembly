import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr
from .core import MissingInput,divide,require


def partition_similarity(a,b):
    a=np.asarray(a);b=np.asarray(b);ok=(a>=0)&(b>=0);a=a[ok];b=b[ok];n=len(a)
    if n<2:return np.nan,np.nan,n
    _,a=np.unique(a,return_inverse=True);_,b=np.unique(b,return_inverse=True)
    tab=np.zeros((a.max()+1,b.max()+1),int);np.add.at(tab,(a,b),1)
    aa=tab.sum(1);bb=tab.sum(0);p=tab/n
    with np.errstate(divide='ignore',invalid='ignore'):
        mi=np.sum(np.where(tab>0,p*np.log(np.maximum(tab,1)*n/(aa[:,None]*bb)),0))
    ha=-np.sum(aa/n*np.log(aa/n));hb=-np.sum(bb/n*np.log(bb/n));emi=0.
    for x in aa:
        for y in bb:
            k=np.arange(max(1,x+y-n),min(x,y)+1)
            emi+=np.sum(k/n*np.log(k*n/(x*y))*hypergeom.pmf(k,n,int(x),int(y)))
    den=(ha+hb)/2-emi
    ami=(mi-emi)/den if abs(den)>1e-12 else (1. if abs(ha-hb)<1e-12 else 0.)
    return float(ami),float(max(0.,ha+hb-2*mi)/np.log(n)),n


def histogram_distribution(h):
    """Zero counts are repeated in every bin and must be counted exactly once."""
    keys=['year','international','metric']
    require(h,keys+['bin','lo','hi','n_positive','n_zero','n_total'],'histograms')
    if h.duplicated(keys+['bin']).any():raise ValueError('Duplicate histogram bin rows.')
    h=h.copy();h['period']=h.year.astype(int)//5*5;records=[];quantiles=[]
    for (period,group,metric),z in h.groupby(['period','international','metric']):
        per_year=z.groupby(keys).agg(zero=('n_zero','first'),total=('n_total','first'),
                                    positive=('n_positive','sum'),zero_versions=('n_zero','nunique'),
                                    total_versions=('n_total','nunique'))
        if (per_year.zero_versions!=1).any() or (per_year.total_versions!=1).any():raise ValueError('Histogram metadata varies between bins.')
        if not np.allclose(per_year.zero+per_year.positive,per_year.total):raise ValueError('Histogram totals do not conserve papers.')
        zero=per_year.zero.sum();total=per_year.total.sum()
        bins=z.groupby(['bin','lo','hi'],as_index=False).n_positive.sum().sort_values('bin')
        cdf=(zero+bins.n_positive.cumsum())/total if total>0 else np.full(len(bins),np.nan)
        for row,value in zip(bins.itertuples(),cdf):
            records.append(dict(period=period,international=group,metric=metric,bin=row.bin,
                lo=row.lo,hi=row.hi,n_positive=row.n_positive,n_zero=zero,n_total=total,
                zero_fraction=zero/total if total else np.nan,cdf=value))
        qs={}
        for prob,label in [(0.25,'p25_upper'),(.5,'median_upper'),(.75,'p75_upper')]:
            ix=np.flatnonzero(np.asarray(cdf)>=prob)
            qs[label]=0. if total and zero/total>=prob else (float(bins.hi.iloc[ix[0]]) if len(ix) else np.nan)
        quantiles.append(dict(period=period,international=group,metric=metric,n=total,
                              zero_fraction=zero/total if total else np.nan,**qs))
    return pd.DataFrame(records),pd.DataFrame(quantiles)


def change_by_unit(old,new):
    keys=['country','field'];cols=['E_asm','B_asm']
    require(old,keys+cols,'old roles');require(new,keys+cols,'new roles')
    if old.duplicated(keys).any() or new.duplicated(keys).any():raise ValueError('Duplicate role units in a period.')
    a=old.set_index(keys)[cols];b=new.set_index(keys)[cols];units=a.index.union(b.index)
    a=a.reindex(units,fill_value=0);b=b.reindex(units,fill_value=0)
    if a.E_asm.sum()<=0 or b.E_asm.sum()<=0:raise ValueError('Both periods need positive assembly exposure.')
    p0=a.E_asm.to_numpy()/a.E_asm.sum();p1=b.E_asm.to_numpy()/b.E_asm.sum()
    r0=np.nan_to_num(divide(a.B_asm,a.E_asm));r1=np.nan_to_num(divide(b.B_asm,b.E_asm))
    both=(a.E_asm.to_numpy()>0)&(b.E_asm.to_numpy()>0)
    entry=(a.E_asm.to_numpy()==0)&(b.E_asm.to_numpy()>0);exit=(a.E_asm.to_numpy()>0)&(b.E_asm.to_numpy()==0)
    out=pd.DataFrame(index=units);out['E0']=a.E_asm;out['E1']=b.E_asm;out['weight0']=p0;out['weight1']=p1
    out['R0']=np.where(a.E_asm>0,r0,np.nan);out['R1']=np.where(b.E_asm>0,r1,np.nan)
    out['within']=np.where(both,(p0+p1)/2*(r1-r0),0.)
    out['composition']=np.where(both,(r0+r1)/2*(p1-p0),0.)
    out['entry']=np.where(entry,p1*r1,0.);out['exit']=np.where(exit,-p0*r0,0.)
    out['total']=out[['within','composition','entry','exit']].sum(axis=1)
    delta=(b.B_asm.sum()/b.E_asm.sum())-(a.B_asm.sum()/a.E_asm.sum())
    if not np.isclose(out.total.sum(),delta,atol=1e-10):raise AssertionError('SI change accounting does not conserve the global mean.')
    return out.reset_index()


def node_participation(W,labels):
    W=np.asarray(W,float).copy();np.fill_diagonal(W,0)
    labels=np.asarray(labels);known=labels>=0;out=[]
    for A in (W,W.T):
        p=np.full(len(W),np.nan);ret=divide(A[:,known].sum(1),A.sum(1))
        for i in np.flatnonzero(known&(A[:,known].sum(1)>0)):
            s=np.bincount(labels[known],weights=A[i,known]);p[i]=1-np.sum((s/s.sum())**2)
        out.extend([p,ret])
    return out  # out participation, retained out mass, in participation, retained in mass


def population(ctx):
    z=ctx.raw('population_year',['year','one','country_known','field_known','eligible'])
    z=z[z.year.between(ctx.start,ctx.end)].copy()
    for name in ['country_known','field_known','eligible']:z[name+'_fraction']=divide(z[name],z.one)
    z['eligible_given_country_known']=divide(z.eligible,z.country_known)
    ctx.write('S01_population',z,'Annual population and observed coverage','Core dated population; P* inclusion is conditional on observed metadata and references.')


def sensitivities(ctx):
    annual=ctx.raw('paper_annual',['year','N','joint_cross_mean'])
    definitions=ctx.raw('definition_sensitivity',['year','variant','count','sum','mean'])
    z=definitions.merge(annual[['year','N']],on='year',validate='many_to_one')
    z['retained_fraction']=divide(z['count'],z.N)
    ctx.write('S02_definition_sensitivity',z,'Definition sensitivity and retained paper counts','Each definition has its own explicitly observed support; primary is not proof of multilabel validity.')
    fixed=ctx.raw('fixed_support_mean',['year','mean','n_fixed_cells','retained_baseline_mass'])
    ctx.write('S03_fixed_support',fixed,'Fixed-support country-field analysis','Fixed baseline weights; applies only from the configured support start.')
    periods=ctx.raw('period_endpoint_comparison',['start_year','end_year','N','mean_q'])
    ctx.write('S04_endpoints',periods,'Alternative end-window summaries','Different end-window lengths are descriptive comparisons, not isolated year effects.')


def references(ctx):
    z=ctx.raw('network_annual',['year','N','N_model','H','H_model','H0','H_context'])
    z['q_all']=divide(z.H,z.N);z['q_model']=divide(z.H_model,z.N_model)
    z['q0']=divide(z.H0,z.N_model);z['q_context']=divide(z.H_context,z.N)
    z['gap']=z.q_model-z.q0;z['observed_to_reference']=divide(z.q_model,z.q0)
    z['model_coverage']=divide(z.N_model,z.N)
    ctx.write('S05_reference_comparison',z,'Observed and label-mixing reference contexts','H_model and H0 share the identical subset. H_context uses full-stratum means; q0 follows the configured label-mixing comparison.')


def field_metrics(ctx):
    z=ctx.raw('field_year_means',['year','field','metric','mass','sum','sum_sq','mean'])
    z=z[z.year.between(ctx.start,ctx.end)].copy();z['field_label']=z.field.map(ctx.field)
    ctx.write('S06_field_year',z,'Field-specific annual metrics','Fractional receiving-field weights; all principal metrics use the same joint reference set.')
    z['period']=z.year.astype(int)//5*5
    w=z.groupby(['period','field','metric'],as_index=False)[['mass','sum','sum_sq']].sum()
    w['mean']=divide(w['sum'],w.mass);w['sd_descriptive']=np.sqrt(np.maximum(0,divide(w.sum_sq,w.mass)-w['mean']**2))
    w['field_label']=w.field.map(ctx.field)
    ctx.write('S07_field_period',w,'Field-specific period metrics','Ratio of summed metric totals to summed fractional mass, with descriptive standard deviations.')


def distributions(ctx):
    cdf,quant=histogram_distribution(ctx.raw('histograms'))
    ctx.write('S08_distribution_cdf',cdf,'Full-P* empirical distributions','Zero mass counted once per year and group; CDF combines zero and positive counts.')
    ctx.write('S09_distribution_quantiles',quant,'Histogram-based distribution summaries','Quantiles are upper bin edges; exact zero quantiles are retained. These are not exact raw-data quantiles.')


def roles(ctx):
    w=ctx.raw('role_country_field_period',['period','country','field','E_src','B_src','E_asm','B_asm'])
    if w.duplicated(['period','country','field']).any():raise ValueError('Role table has duplicate units.')
    world=divide(w.groupby('period').B_asm.transform('sum'),w.groupby('period').E_asm.transform('sum'))
    for role in ('src','asm'):
        w['R_'+role]=divide(w['B_'+role],w['E_'+role]);w['LQ_'+role]=divide(w['R_'+role],world)
    w['LQ_gap']=w.LQ_asm-w.LQ_src
    w['both_roles_supported']=(w.E_src>=ctx.minimum)&(w.E_asm>=ctx.minimum)
    w['country_code']=w.country.map(ctx.country);w['field_label']=w.field.map(ctx.field)
    ctx.write('S10_country_field_roles',w,'Country-field role accounts','All cells retained; displayed gaps require the minimum exposure in both roles. Source credits include historical cited stocks.')
    mass=[k for k in w if k.startswith(('E_','B_','B0_'))]
    z=w.groupby(['period','country'],as_index=False)[mass].sum()
    world=divide(z.groupby('period').B_asm.transform('sum'),z.groupby('period').E_asm.transform('sum'))
    for role in ('src','asm'):
        z['R_'+role]=divide(z['B_'+role],z['E_'+role]);z['LQ_'+role]=divide(z['R_'+role],world)
        cols=['E_'+role+'_model','B_'+role+'_model','B0_'+role]
        if all(k in z for k in cols):
            z['q_model_'+role]=divide(z[cols[1]],z[cols[0]])
            z['q0_'+role]=divide(z[cols[2]],z[cols[0]])
            z['model_gap_'+role]=z['q_model_'+role]-z['q0_'+role]
    z['LQ_gap']=z.LQ_asm-z.LQ_src;z['country_code']=z.country.map(ctx.country)
    ctx.write('S11_country_roles',z,'Country role accounts','Ratios of summed credits and exposures; no average of country-field ratios.')
    corr=[]
    for (period,field),t in w[w.both_roles_supported].groupby(['period','field']):
        for x,y in [('LQ_src','LQ_asm'),('RCA','LQ_asm')]:
            if x not in t:continue
            a=t[[x,y]].replace([np.inf,-np.inf],np.nan).dropna()
            rho=spearmanr(a[x],a[y]).statistic if len(a)>2 and a[x].nunique()>1 and a[y].nunique()>1 else np.nan
            corr.append(dict(period=period,field=field,field_label=ctx.field[field],x=x,y=y,n_cells=len(a),spearman_rho=rho))
    ctx.write('S12_role_correlations',pd.DataFrame(corr),'Within-field role correlations','Unweighted country-field rank correlations after applying the same exposure minimum in both roles.')


def concentration(ctx):
    z=ctx.raw('role_concentration',['scope','period','role','quantity','n_units','Gini','top5_share','top10pct_share'])
    ctx.write('S13_concentration',z,'Role and exposure concentration','Uses a fixed union of units, including period-specific zeros; reports top-five-unit and top-five-percent shares separately.')


def communities(ctx):
    runs=ctx.raw('community_runs',['period','layer','gamma','repeat','country','community','quality'])
    stability=ctx.raw('community_stability',['period','layer','gamma','best_repeat','mean_ami_to_best','n_assigned','n_communities'])
    ctx.write('S14_optimisation_stability',stability,'Leiden optimisation stability','Optimisation repeats assess algorithmic reproducibility, not statistical sampling uncertainty.')
    n=len(ctx.countries);best={};best_rows=[]
    for row in stability.itertuples():
        z=runs[(runs.period==row.period)&(runs.layer==row.layer)&np.isclose(runs.gamma,row.gamma)&(runs['repeat']==row.best_repeat)].copy()
        if z.empty or z.country.duplicated().any():raise ValueError('Selected community repeat is missing or has duplicate countries.')
        a=z.set_index('country').community.reindex(np.arange(n),fill_value=-1).to_numpy(int)
        best[(int(row.period),str(row.layer),float(row.gamma))]=a
        best_rows.extend(dict(period=row.period,layer=row.layer,gamma=row.gamma,country=i,community=int(a[i])) for i in range(n))
    ctx.write('S15_selected_communities',pd.DataFrame(best_rows),'Selected communities at every resolution','Best repeats are read from the stored optimisation table.')
    try:net,periods=ctx.networks();lookup={int(p):i for i,p in enumerate(periods)}
    except MissingInput:net=None;lookup={}
    rows=[]
    for (period,gamma),_ in stability.groupby(['period','gamma']):
        for a,b in [('C','F'),('C','H'),('F','H')]:
            if (int(period),a,float(gamma)) not in best or (int(period),b,float(gamma)) not in best:continue
            x=best[(int(period),a,float(gamma))];y=best[(int(period),b,float(gamma))]
            ami,vi,nshared=partition_similarity(x,y);retained=np.nan
            if net is not None and int(period) in lookup:
                F=net[lookup[int(period)],1].copy();np.fill_diagonal(F,0);mass=F.sum(0)+F.sum(1)
                retained=float(divide(mass[(x>=0)&(y>=0)].sum(),mass.sum()))
            rows.append(dict(period=period,gamma=gamma,layers=a+'-'+b,AMI=ami,VI_normalized=vi,n_shared=nshared,international_F_exposure_retained=retained))
    ctx.write('S16_resolution_similarity',pd.DataFrame(rows),'Cross-layer agreement over resolution','AMI equally weights shared assigned countries. Exposure coverage uses non-diagonal F; it does not turn AMI into a weighted statistic.')


def network_diagnostics(ctx):
    net,periods=ctx.networks();labels=ctx.table('S15_selected_communities')
    n=len(ctx.countries);summary=[];nodes=[];edges=[]
    base=labels[(labels.period==ctx.base)&(labels.layer=='F')&np.isclose(labels.gamma,1)]
    if base.empty:raise MissingInput('Baseline F partition at gamma=1')
    fixed=base.set_index('country').community.reindex(np.arange(n),fill_value=-1).to_numpy(int)
    for w,period in enumerate(periods):
        for li,layer in [(1,'F'),(2,'H')]:
            W=net[w,li];A=W.copy();np.fill_diagonal(A,0)
            summary.append(dict(period=int(period),layer=layer,total_mass=W.sum(),international_mass=A.sum(),domestic_mass=np.trace(W),domestic_fraction=float(divide(np.trace(W),W.sum()))))
            po,ro,pi,ri=node_participation(W,fixed)
            for i in range(n):nodes.append(dict(period=int(period),layer=layer,country=i,country_code=ctx.country[i],total_strength_in=W[:,i].sum(),total_strength_out=W[i].sum(),international_strength_in=A[:,i].sum(),international_strength_out=A[i].sum(),participation_out=po[i],participation_in=pi[i],retained_out_mass=ro[i],retained_in_mass=ri[i],baseline_F_community=int(fixed[i])))
        F=net[w,1].copy();H=net[w,2].copy();np.fill_diagonal(F,0);np.fill_diagonal(H,0)
        if np.any(H>F+1e-5):raise ValueError('H exceeds F in cached networks.')
        if int(period)==ctx.current:
            ii,jj=np.nonzero(F>0)
            edges=pd.DataFrame({'source':ii,'target':jj,'F':F[ii,jj],'H':H[ii,jj],'H_over_F':divide(H[ii,jj],F[ii,jj])})
    ctx.write('S17_network_mass',pd.DataFrame(summary),'Domestic and international F/H mass','Domestic fraction is computed from diagonal mass; C symmetric adjacency is not mixed into these fractions.')
    ctx.write('S18_network_nodes',pd.DataFrame(nodes),'Directed network node diagnostics','Source-to-receiver orientation; out describes source spread, in describes received-source diversity. Participation uses a frozen baseline F partition and non-diagonal weights.')
    if isinstance(edges,pd.DataFrame):ctx.write('S19_network_edges_current',edges,'Recent international edge weights','Off-diagonal edges with F>0; H/F records exposure-weighted context intensity.')


def composition(ctx):
    w=ctx.table('S10_country_field_roles')
    z=change_by_unit(w[w.period==ctx.base],w[w.period==ctx.current])
    z['country_code']=z.country.map(ctx.country);z['field_label']=z.field.map(ctx.field)
    ctx.write('S20_change_by_unit',z,'Exact change accounting by country-field unit','Symmetric within/composition plus entry and exit; within a broad unit still contains finer compositional changes.')
    for key in ['country','field']:
        a=z.groupby(key,as_index=False)[['within','composition','entry','exit','total']].sum()
        a[key+'_label']=a[key].map(ctx.country if key=='country' else ctx.field)
        ctx.write('S21_change_by_'+key,a,'Change accounting by '+key,'All contributions share the same global denominators and sum to the overall change.')


def cohorts(ctx):
    z=ctx.raw('reference_cohorts',['citing_year','cited_year','E','B'])
    z['age']=z.citing_year-z.cited_year
    if (z.age<0).any() or (z.B>z.E+1e-5).any():raise ValueError('Invalid reference cohort chronology or credit.')
    z['period']=z.citing_year.astype(int)//5*5
    z['cohort']=np.where(z.cited_year<1950,1945,z.cited_year.astype(int)//5*5)
    a=z.groupby(['period','cohort'],as_index=False)[['E','B']].sum()
    a['intensity']=divide(a.B,a.E);a['exposure_share']=divide(a.E,a.groupby('period').E.transform('sum'))
    ctx.write('S22_source_cohorts',a,'Source cohorts and observed credit','All source ages are retained, with pre-1950 cohorts pooled for display. B/E records contextual credit per unit exposure.')
    labels=['0','1-2','3-5','6-10','11-20','21-40','41+']
    z['age_band']=pd.cut(z.age,[-.5,.5,2.5,5.5,10.5,20.5,40.5,np.inf],labels=labels)
    a=z.groupby(['period','age_band'],observed=True,as_index=False)[['E','B']].sum()
    a['intensity']=divide(a.B,a.E);a['exposure_share']=divide(a.E,a.groupby('period').E.transform('sum'))
    ctx.write('S23_reference_age',a,'Reference-age structure','Age is citing year minus cited year for each paper-year pair.')


def surfaces(ctx):
    z=ctx.raw('response_surface',['team_bin','reference_bin','team_label','reference_label','n','mean'])
    z['displayed']=z.n>=int(ctx.c.get('min_surface_n',100))
    ctx.write('S24_response_surface',z,'Observed bin support and means','Full recent P* counts and observed means in ordinal bins with unequal widths.')


def regression(ctx):
    z=ctx.raw('coefficients',['model','term','estimate','std_error','conf_low','conf_high','n'],folder='r')
    scope=ctx.raw('model_scope',folder='r')
    z['CI_crosses_zero']=(z.conf_low<=0)&(z.conf_high>=0)
    ctx.write('S25_regression_coefficients',z,'Reported conditional associations','Imported model estimates and covariance-specific confidence intervals.')
    ctx.write('S26_regression_scope',scope,'Regression populations, weights and covariance','Model units and fixed effects differ; AI-classified subsets and aggregate cells are separate estimands.')


def sample_diagnostics(ctx):
    s=ctx.sample();rows=[]
    for metric in ['known_fields','known_countries','joint_refs','core_refs','team_size']:
        if metric not in s:continue
        counts=s[metric].dropna().value_counts().sort_index()
        rows.extend(dict(metric=metric,value=float(v),n=int(n),fraction=n/len(s),n_sample=len(s)) for v,n in counts.items())
    ctx.write('S27_sample_distributions',pd.DataFrame(rows),'Observed sample metadata distributions',ctx.sample_scope+' All fractions use the entire selected sample as their denominator.')
    rows=[]
    for year,t in s.groupby('year'):
        row={'year':year,'n_sample':len(t)}
        for metric in ['reference_coverage','joint_refs','joint_cross']:
            v=t[metric].dropna();row[metric+'_n']=len(v);row[metric+'_mean']=v.mean()
            for prob,label in [(.1,'p10'),(.5,'median'),(.9,'p90')]:row[metric+'_'+label]=v.quantile(prob)
        rows.append(row)
    annual=pd.DataFrame(rows)
    try:
        pop=ctx.table('S01_population',['year','eligible'])
        annual=annual.merge(pop[['year','eligible']],on='year',how='left',validate='one_to_one')
        annual['sample_fraction']=divide(annual.n_sample,annual.eligible)
    except MissingInput:pass
    ctx.write('S28_sample_year',annual,'Plot-sample support by year',ctx.sample_scope+' Annual counts describe the selected plotting sample.')
    valid=s.reference_coverage.between(0,1)&s.joint_cross.notna()
    z=s.loc[valid].copy();z['coverage_bin']=pd.cut(z.reference_coverage,np.linspace(0,1,11),include_lowest=True,labels=False)
    out=z.groupby('coverage_bin',as_index=False).agg(n=('joint_cross','size'),mean_q=('joint_cross','mean'),mean_coverage=('reference_coverage','mean'),mean_joint_refs=('joint_refs','mean'))
    out['lower']=out.coverage_bin/10;out['upper']=(out.coverage_bin+1)/10
    ctx.write('S29_sample_coverage',out,'Sample co-use and observed reference coverage',ctx.sample_scope+' Descriptive summaries use observed reference-coverage bins.')


def semantic(ctx):
    rows=[]
    try:joint=ctx.raw('joint_semantic',['q_same_refs','cosine_pair_distance'])
    except MissingInput:joint=pd.DataFrame()
    if len(joint):
        z=joint[np.isfinite(joint.q_same_refs)&np.isfinite(joint.cosine_pair_distance)].copy()
        z['scope']='same_reference_set'
        ctx.write('S30_joint_semantic',z,'Exact same-reference semantic check','Uses computed country, Fields and valid-embedding information on the same references.')
        rows.append(dict(scope='same_reference_set',n=len(z),rho=spearmanr(z.q_same_refs,z.cosine_pair_distance).statistic if len(z)>2 and z.q_same_refs.nunique()>1 and z.cosine_pair_distance.nunique()>1 else np.nan))
    try:s=ctx.sample()
    except MissingInput:s=pd.DataFrame()
    if {'semantic_existing','semantic_coverage'}<=set(s):
        z=s[s.semantic_coverage.ge(.5)&np.isfinite(s.semantic_existing)&np.isfinite(s.joint_cross)].copy()
        if len(z):
            ctx.write('S31_broader_semantic',z,'Broader-reference semantic association',ctx.sample_scope+' Existing semantic distance uses its documented broader reference set.')
            rows.append(dict(scope='different_broader_reference_set',n=len(z),rho=spearmanr(z.joint_cross,z.semantic_existing).statistic if len(z)>2 and z.joint_cross.nunique()>1 and z.semantic_existing.nunique()>1 else np.nan))
    if not rows:raise MissingInput('No valid same-reference or broader-reference semantic measurements.')
    ctx.write('S32_semantic_summary',pd.DataFrame(rows),'Semantic scope and rank associations','Unweighted Spearman rank correlations are reported separately for each reference scope.')


def population_diagnostics(ctx,batch_rows=250000):
    """Optional streaming population diagnostics of existing paper outputs."""
    try:import pyarrow.parquet as pq
    except ImportError as exc:raise MissingInput('pyarrow is required for streaming population diagnostics.') from exc
    path=ctx.results/'data/assembly_papers.parquet'
    if not path.is_file():raise MissingInput(str(path))
    ctx.track(path);pf=pq.ParquetFile(path)
    cols=['paper_id','year','eligible','known_fields','known_countries','reference_coverage','joint_cross']
    if not set(cols)<=set(pf.schema_arrow.names):raise ValueError('Population diagnostics require the expected assembly-paper schema.')
    kinds=ids=None;kind_path=ctx.results/'cache/kind.npy';id_path=ctx.results/'cache/paper_id.npy'
    if kind_path.is_file() and id_path.is_file():
        kinds=np.load(kind_path,mmap_mode='r',allow_pickle=False);ids=np.load(id_path,mmap_mode='r',allow_pickle=False)
        if len(kinds)!=pf.metadata.num_rows or len(ids)!=len(kinds):raise ValueError('Kind/ID cache length does not match assembly rows.')
        ctx.track(kind_path);ctx.track(id_path)
    frames=(b.to_pandas() for b in pf.iter_batches(columns=cols,batch_size=int(batch_rows),use_threads=False))
    return diagnostic_frames(ctx,frames,kinds,ids,expected_rows=pf.metadata.num_rows)


def diagnostic_frames(ctx,frames,kinds=None,ids=None,expected_rows=None):
    """Summarize population membership and coverage in bounded batches."""
    from collections import Counter
    if kinds is not None and (ids is None or len(kinds)!=len(ids)):
        raise ValueError('Source kinds require an equally long exact-ID index.')
    counts=Counter();coverage=[];types=[];offset=0
    for number,frame in enumerate(frames):
        a=frame.copy();stop=offset+len(a)
        if kinds is not None:
            if not np.array_equal(a.paper_id.to_numpy(),ids[offset:stop]):raise ValueError('Kind cache row alignment failed the paper-ID check.')
            a['kind']=np.asarray(kinds[offset:stop])
        offset=stop;a=a[a.year.between(ctx.start,ctx.end)].copy()
        for metric in ['known_fields','known_countries']:
            for (year,value),n in a.groupby(['year',metric],observed=True).size().items():counts[(int(year),metric,int(value))]+=int(n)
        z=a[a.eligible].copy()
        if ((~z.reference_coverage.between(0,1)).any() or (~z.joint_cross.between(0,1)).any()
            or (z.known_fields<1).any() or (z.known_countries<1).any()):
            raise ValueError('Eligible paper must have valid coverage and co-use values.')
        z['coverage_bin']=pd.cut(z.reference_coverage,np.linspace(0,1,11),include_lowest=True,labels=False)
        x=z.dropna(subset=['coverage_bin','joint_cross']).groupby(['year','coverage_bin']).joint_cross.agg(['count','sum']).reset_index();coverage.append(x)
        if kinds is not None:types.append(z.groupby(['year','kind']).joint_cross.agg(['count','sum']).reset_index())
        if number%20==0:print(f'[population diagnostics] scanned {offset:,} existing paper rows',flush=True)
    if expected_rows is not None and offset!=expected_rows:raise ValueError('Population row count changed during the scan.')
    labels=pd.DataFrame([dict(year=y,metric=m,value=v,n=n) for (y,m,v),n in counts.items()])
    ctx.write('S33_full_label_diagnostics',labels,'Full dated-core membership diagnostics','All dated core papers in the configured analysis interval, including non-P* papers.')
    cov=pd.concat(coverage,ignore_index=True).groupby(['year','coverage_bin'],as_index=False)[['count','sum']].sum()
    cov['mean_q']=divide(cov['sum'],cov['count'])
    cov['fraction_within_year']=divide(cov['count'],cov.groupby('year')['count'].transform('sum'))
    ctx.write('S34_full_coverage_diagnostics',cov,'Full-P* reference-coverage diagnostics','Classification coverage of observed core references.')
    if types:
        z=pd.concat(types,ignore_index=True).groupby(['year','kind'],as_index=False)[['count','sum']].sum()
        z['mean_q']=divide(z['sum'],z['count']);z['kind_label']=z.kind.map({0:'Journal',1:'Conference',2:'Unknown'})
        if z.kind_label.isna().any():raise ValueError('Unrecognised cached source type.')
        z['share']=divide(z['count'],z.groupby('year')['count'].transform('sum'))
        ctx.write('S35_source_type_diagnostics',z,'Full-P* source-type diagnostics','Cached source kind is row-matched using exact paper IDs and the pipeline source-type flag.')
    else:ctx.event('tables','source_type_diagnostics','skipped','kind.npy and paper_id.npy are both required for aligned source-type diagnostics.')


ANALYSES=[population,sensitivities,references,field_metrics,distributions,roles,concentration,
          communities,network_diagnostics,composition,cohorts,surfaces,sample_diagnostics,semantic,regression]

def build_tables(ctx):
    ctx.begin_tables()
    for fn in ANALYSES:
        try:fn(ctx)
        except MissingInput as exc:ctx.event('tables',fn.__name__,'skipped',exc)
        else:ctx.event('tables',fn.__name__,'complete')
    ctx.finish()
    return ctx
