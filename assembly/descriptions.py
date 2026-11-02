import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.integrate import trapezoid
from .runtime import (dirs,arr,batches,csv,ratio,hash64,Writer,check_memory,json_write,
                      ids,Lookup)
from .kernels import field_summaries,core_output

EDGES=np.linspace(0,1,101)
TEAM_BINS=np.array([0,1,2,4,8,16,32,64,np.inf])
REF_BINS=np.array([1,2,4,9,19,39,79,159,np.inf])
TEAM_LABELS=['1','2','3-4','5-8','9-16','17-32','33-64','65+']
REF_LABELS=['2','3-4','5-9','10-19','20-39','40-79','80-159','160+']

def rca_frame(z,mass,group='year'):
    z=z.copy();total=z.groupby(group)[mass].transform('sum')
    country=z.groupby([group,'country'])[mass].transform('sum');field=z.groupby([group,'field'])[mass].transform('sum')
    z['RCA']=ratio(z[mass]*total,country*field)
    z['RSCA']=ratio(z.RCA-1,z.RCA+1)
    return z

def roles(c):
    d=dirs(c);z=pd.read_csv(d['tables']/'role_masses.csv')
    z=rca_frame(z,'E_asm')
    g=z.groupby('year');world=ratio(g.B_asm.transform('sum'),g.E_asm.transform('sum'))
    for role in ('src','asm'):
        z['R_'+role]=ratio(z['B_'+role],z['E_'+role]);z['LQ_'+role]=ratio(z['R_'+role],world)
        z['R_model_'+role]=ratio(z['B_'+role+'_model'],z['E_'+role+'_model'])
        z['R0_'+role]=ratio(z['B0_'+role],z['E_'+role+'_model'])
    csv(z,d['tables']/'role_country_field_year.csv')
    p=z.copy();p['period']=(p.year//5*5).astype(int)
    mass=[x for x in z if x.startswith(('E_','B_','B0_','B_context'))]
    w=p.groupby(['period','country','field'],as_index=False)[mass].sum()
    w=rca_frame(w,'E_asm','period');wg=w.groupby('period');world=ratio(wg.B_asm.transform('sum'),wg.E_asm.transform('sum'))
    for role in ('src','asm'):
        w['R_'+role]=ratio(w['B_'+role],w['E_'+role]);w['LQ_'+role]=ratio(w['R_'+role],world)
        w['R_model_'+role]=ratio(w['B_'+role+'_model'],w['E_'+role+'_model'])
        w['R0_'+role]=ratio(w['B0_'+role],w['E_'+role+'_model'])
    csv(w,d['tables']/'role_country_field_period.csv')
    country=w.groupby(['period','country'],as_index=False)[mass].sum()
    wg=country.groupby('period');world=ratio(wg.B_asm.transform('sum'),wg.E_asm.transform('sum'))
    for role in ('src','asm'):
        country['R_'+role]=ratio(country['B_'+role],country['E_'+role]);country['LQ_'+role]=ratio(country['R_'+role],world)
    csv(country,d['tables']/'role_country_period.csv')
    lor=[];conc=[]
    universe=(w.groupby(['country','field'])[['E_src','E_asm']].sum().sum(axis=1)>0)
    units=universe[universe].index
    countries=country.groupby('country')[['E_src','E_asm']].sum().sum(axis=1)
    cunit=countries[countries>0].index
    for scope,table,index in [('country_field',w,units),('country',country,cunit)]:
        keys=['country','field'] if scope=='country_field' else ['country']
        for period,t in table.groupby('period'):
            t=t.set_index(keys).reindex(index).fillna(0)
            for role in ('src','asm'):
                for quantity in ('E','B'):
                    v=np.sort(t[quantity+'_'+role].to_numpy(float));n=len(v)
                    shares=np.r_[0,np.cumsum(v)/v.sum()] if v.sum()>0 else np.full(n+1,np.nan)
                    gini=1-2*trapezoid(shares,dx=1/n) if n and v.sum()>0 else np.nan
                    lor.extend(dict(scope=scope,period=period,role=role,quantity=quantity,pop_frac=k/n,cum_share=shares[k]) for k in range(n+1))
                    conc.append(dict(scope=scope,period=period,role=role,quantity=quantity,n_units=n,Gini=gini,
                        top5_share=float(v[-5:].sum()/v.sum()) if v.sum() else np.nan,
                        top10pct_share=float(v[-max(1,int(np.ceil(n*.1))):].sum()/v.sum()) if v.sum() else np.nan))
    csv(pd.DataFrame(lor),d['tables']/'role_lorenz.csv');csv(pd.DataFrame(conc),d['tables']/'role_concentration.csv')
    correlations=[]
    for period,t in w.groupby('period'):
        for a,b in [('RCA','LQ_asm'),('LQ_src','LQ_asm'),('E_src','R_src')]:
            ok=t[[a,b]].notna().all(axis=1)&(t.E_asm>=c['min_cell_mass'])&(t.E_src>=c['min_cell_mass'])
            v=t.loc[ok]
            rho=spearmanr(v[a],v[b]).statistic if len(v)>2 and v[a].nunique()>1 and v[b].nunique()>1 else np.nan
            correlations.append(dict(period=period,x=a,y=b,n_cells=len(v),spearman_rho=rho,min_exposure_each_role=c['min_cell_mass']))
    csv(pd.DataFrame(correlations),d['tables']/'role_correlations.csv')
    # Exact within/composition/entry/exit accounting of changes in the global mean.
    b=c['baseline_years'][0];e=c['comparison_years'][0]
    old=w.loc[w.period.eq(b)].set_index(['country','field']);new=w.loc[w.period.eq(e)].set_index(['country','field'])
    common=old.index.union(new.index);a=old.reindex(common).fillna(0);v=new.reindex(common).fillna(0)
    p0=ratio(a.E_asm,a.E_asm.sum());p1=ratio(v.E_asm,v.E_asm.sum());r0=np.nan_to_num(ratio(a.B_asm,a.E_asm));r1=np.nan_to_num(ratio(v.B_asm,v.E_asm))
    both=(a.E_asm>0)&(v.E_asm>0);entry=(a.E_asm==0)&(v.E_asm>0);exit=(a.E_asm>0)&(v.E_asm==0)
    components={'within':float(np.sum((p0[both]+p1[both])/2*(r1[both]-r0[both]))),
        'composition':float(np.sum((r0[both]+r1[both])/2*(p1[both]-p0[both]))),
        'entry':float(np.sum(p1[entry]*r1[entry])),'exit':float(-np.sum(p0[exit]*r0[exit]))}
    difference=float(np.sum(p1*r1)-np.sum(p0*r0))
    if not np.isclose(sum(components.values()),difference,atol=1e-10):raise AssertionError('Change accounting failed.')
    csv(pd.DataFrame([dict(component=k,value=v,baseline=b,comparison=e,total_difference=difference) for k,v in components.items()]),d['tables']/'change_accounting.csv')
    # A fixed set is defined only in its explicitly eligible 2000--2024 interval.
    eligible=z[z.year.between(c['fixed_support_start'],c['end_year'])]
    ny=c['end_year']-c['fixed_support_start']+1
    cells=eligible.pivot(index=['country','field'],columns='year',values='E_asm').fillna(0)
    fixed=cells.index[(cells>=c['min_cell_mass']).sum(axis=1)==ny]
    base=z[z.year.between(*c['baseline_years'])].groupby(['country','field']).E_asm.sum()
    ww=base.reindex(fixed).fillna(0);retained=ww.sum()/base.sum() if base.sum() else np.nan
    lines=[]
    if ww.sum()>0:
        for year,t in eligible.groupby('year'):
            a=t.set_index(['country','field']).reindex(fixed)
            value=float(np.sum(ww.to_numpy()/ww.sum()*a.R_asm.to_numpy()))
            lines.append(dict(year=year,mean=value,n_fixed_cells=len(fixed),retained_baseline_mass=retained,
                support_start=c['fixed_support_start'],weighting='fixed baseline country x Fields output'))
    csv(pd.DataFrame(lines,columns=['year','mean','n_fixed_cells','retained_baseline_mass','support_start','weighting']),d['tables']/'fixed_support_mean.csv')
    return z,w

def population_rca(c):
    d=dirs(c);year=arr(c,'year');cp=arr(c,'country_ptr');fp=arr(c,'field_ptr');cc=arr(c,'country_code');cw=arr(c,'country_weight');fc=arr(c,'field_code');fw=arr(c,'field_weight')
    nc=len(pd.read_csv(d['tables']/'countries.csv'));nf=len(pd.read_csv(d['tables']/'fields.csv'))
    mask=(year>=c['trend_start'])&(year<=c['end_year'])&(np.diff(cp)>0)&(np.diff(fp)>0)
    rows=np.flatnonzero(mask).astype(np.uint32);rows=rows[np.argsort(year[rows],kind='stable')]
    counts=np.bincount(year[rows].astype(int)-c['trend_start'],minlength=c['end_year']-c['trend_start']+1);ptr=np.r_[0,np.cumsum(counts)]
    out=core_output(ptr,rows,nc,nf,cp,cc,cw,fp,fc,fw);records=[]
    for y in range(len(counts)):
        t=pd.DataFrame({'year':y+c['trend_start'],'country':np.repeat(np.arange(nc),nf),
            'field':np.tile(np.arange(nf),nc),'output_mass':out[y].ravel()});records.append(t)
    z=rca_frame(pd.concat(records,ignore_index=True),'output_mass')
    csv(z,d['tables']/'full_core_rca.csv')
    return int(counts.sum())

def field_tables(c):
    d=dirs(c);gp=arr(c,'group_ptr');gr=arr(c,'group_rows');fp=arr(c,'field_ptr');fc=arr(c,'field_code');fw=arr(c,'field_weight');m=arr(c,'paper_metrics')
    out=field_summaries(gp,gr,len(pd.read_csv(d['tables']/'fields.csv')),fp,fc,fw,m)
    rows=[];metrics=['geo_cross','field_cross','joint_cross','primary_joint_cross','mean_ref_age_joint','coupling']
    for y in range(out.shape[0]):
        for f in range(out.shape[1]):
            for k,metric in enumerate(metrics):
                n,s,ss=out[y,f,k];rows.append(dict(year=y+c['trend_start'],field=f,metric=metric,mass=n,sum=s,sum_sq=ss,mean=s/n if n else np.nan))
    csv(pd.DataFrame(rows),d['tables']/'field_year_means.csv')

def semantic_joint(c,sample):
    """Optional exact embedding check on the SAME newly measurable reference set."""
    d=dirs(c);cfg=c['joint_semantic'];columns=['paper_id','year','n_joint_embedding_refs','joint_ref_coverage','q_same_refs','cosine_pair_distance']
    if not cfg['enabled']:
        csv(pd.DataFrame(columns=columns),d['tables']/'joint_semantic.csv')
        json_write({'enabled':False,'reason':'No embedding matrix/index configured. Existing semantic metrics have a broader reference scope and are labelled separately.'},d['status']/'semantic.json');return
    from pathlib import Path
    for key in ('index_parquet','matrix_npy'):
        if not Path(cfg[key]).is_file():raise FileNotFoundError(f'joint_semantic.{key}: {cfg[key]}')
    sample=sample.nsmallest(min(cfg['max_papers'],len(sample)),'_hash')
    pids=arr(c,'paper_id');lookup=Lookup(pids,c['lookup']);selected=lookup.get(sample.paper_id.to_numpy());del lookup
    rp=arr(c,'ref_ptr');rn=arr(c,'ref_count');ri=arr(c,'ref_index');cp=arr(c,'country_ptr');fp=arr(c,'field_ptr')
    cc=arr(c,'country_code');cw=arr(c,'country_weight');fc=arr(c,'field_code');fw=arr(c,'field_weight')
    refs=[]
    for i in selected:
        j=np.asarray(ri[rp[i]:rp[i]+rn[i]]);j=j[(cp[j+1]>cp[j])&(fp[j+1]>fp[j])];refs.append(j)
    needed=np.unique(np.concatenate(refs)) if refs else np.array([],dtype=np.uint32)
    if len(needed)==0:raise ValueError('No references for the configured semantic check.')
    get=Lookup(pids[needed],c['lookup']);embedding_rows=np.full(len(needed),-1,dtype=np.int64)
    seen=np.zeros(len(needed),bool)
    for b in batches(cfg['index_parquet'],[cfg['id_column'],cfg['row_column']],c['batch_rows']):
        ix=get.get(ids(b[cfg['id_column']]));ok=ix>=0;ii=ix[ok]
        if pd.Index(ii).has_duplicates or seen[ii].any():raise ValueError('Duplicate IDs in embedding row index.')
        seen[ii]=True;r=pd.to_numeric(b.loc[ok,cfg['row_column']],errors='raise').to_numpy()
        if np.any(~np.isfinite(r)) or np.any(r<0) or np.any(r%1):raise ValueError('Embedding rows must be zero-based nonnegative integers.')
        embedding_rows[ii]=r.astype(np.int64)
    matrix=np.load(cfg['matrix_npy'],mmap_mode='r',allow_pickle=False)
    if matrix.ndim!=2:raise ValueError('Embedding matrix must be a 2-D numeric .npy array.')
    ok=embedding_rows>=0
    if np.any(embedding_rows[ok]>=len(matrix)):raise ValueError('Embedding row index exceeds matrix length.')
    estimate=len(needed)*matrix.shape[1]*4/2**30
    if estimate>cfg['max_work_gib']:raise MemoryError('Reduce joint_semantic.max_papers; selected embedding working set exceeds max_work_gib.')
    embedding=np.full((len(needed),matrix.shape[1]),np.nan,np.float32)
    valid=np.zeros(len(needed),dtype=bool)
    found=np.flatnonzero(ok)
    for start in range(0,len(found),10000):
        ii=found[start:start+10000]
        x=np.asarray(matrix[embedding_rows[ii]],dtype=np.float32)
        norm=np.linalg.norm(x,axis=1)
        good=np.isfinite(x).all(axis=1)&(norm>0)
        x[good]/=norm[good,None]
        embedding[ii]=x;valid[ii]=good
        check_memory(c,'joint semantic embeddings')
    ok=valid
    nc=len(pd.read_csv(d['tables']/'countries.csv'));nf=len(pd.read_csv(d['tables']/'fields.csv'));results=[]
    paper_year=arr(c,'year');paper_metrics=arr(c,'paper_metrics')
    for paper,ref,record in zip(selected,refs,sample.itertuples(index=False)):
        ix=np.searchsorted(needed,ref);valid=ok[ix];ref=ref[valid];ix=ix[valid];m=len(ref)
        if m<2:continue
        A=np.zeros(nc);T=np.zeros(nf);V=np.zeros((nc,nf));a2=t2=v2=0.
        for j in ref:
            cs=cc[cp[j]:cp[j+1]];ws=cw[cp[j]:cp[j+1]];fs=fc[fp[j]:fp[j+1]];ts=fw[fp[j]:fp[j+1]]
            A[cs]+=ws;T[fs]+=ts;V[np.ix_(cs,fs)]+=np.outer(ws,ts)
            na=np.dot(ws,ws);nt=np.dot(ts,ts);a2+=na;t2+=nt;v2+=na*nt
        q=1-(np.dot(A,A)-a2+np.dot(T,T)-t2-(np.sum(V*V)-v2))/(m*(m-1))
        x=embedding[ix].astype(np.float64);s=x.sum(0)
        distance=1-(np.dot(s,s)-np.sum(x*x))/(m*(m-1))
        results.append(dict(paper_id=int(pids[paper]),year=int(paper_year[paper]),n_joint_embedding_refs=m,
            joint_ref_coverage=m/max(1,int(paper_metrics[paper,0])),
            q_same_refs=q,cosine_pair_distance=distance))
    csv(pd.DataFrame(results,columns=columns),d['tables']/'joint_semantic.csv')
    json_write({'enabled':True,'papers':len(results),'reference_scope':'Same country/Fields/valid-embedding references; no pair enumeration; exact mean cosine pair distance.'},d['status']/'semantic.json')

def build(c):
    d=dirs(c);yearrows=[];hist=[];sensitivity=[];surface=np.zeros((8,8,2));sample=None;samplehash=None
    model=arr(c,'model_q');offset=0
    rcols=['paper_id','year','primary_field','source_code','team_size','known_countries','known_fields',
        'joint_refs','core_refs','reference_coverage','joint_cross','geo_cross','field_cross','mean_ref_age_joint','AIProb']
    with Writer(d['data']/'r_papers.parquet') as rw:
        for b in batches(d['data']/'assembly_papers.parquet',None,c['batch_rows']):
            lo=offset;offset+=len(b);b['model_q']=model[lo:offset]
            z=b.loc[b.eligible].copy()
            if len(z)==0:continue
            z['coupling']=z.joint_cross-z.geo_cross*z.field_cross
            z['international']=(z.known_countries>=2).astype(np.int8)
            z['positive_q']=(z.joint_cross>1e-10).astype(np.int8)
            metrics=['geo_cross','field_cross','joint_cross','coupling','reference_coverage','completion_lower','completion_upper','model_q']
            a=z.groupby('year')[metrics].agg(['count','sum']);a.columns=['_'.join(x) for x in a.columns]
            a['N']=z.groupby('year').size();a['q_event_n']=(z.joint_cross*z.joint_refs).groupby(z.year).sum();a['event_d']=z.joint_refs.groupby(z.year).sum();yearrows.append(a)
            for (year,international),g in z.groupby(['year','international']):
                for metric in ['joint_cross','geo_cross','field_cross']:
                    v=g[metric].to_numpy();positive=v[v>1e-10];hh,_=np.histogram(positive,bins=EDGES)
                    hist.append(pd.DataFrame({'year':year,'international':international,'metric':metric,
                        'bin':np.arange(100),'lo':EDGES[:-1],'hi':EDGES[1:],'n_positive':hh,'n_zero':int((v<=1e-10).sum()) if len(v) else 0,'n_total':len(v)}))
            for variant,mask,metric in [('main',np.ones(len(z),bool),'joint_cross'),('at_least_5',z.joint_refs.ge(5),'joint_cross'),
                ('coverage_ge_08',z.reference_coverage.ge(.8),'joint_cross'),('primary_same_refs',z.primary_joint_cross.notna(),'primary_joint_cross')]:
                a=z.loc[mask].groupby('year')[metric].agg(['count','sum']).reset_index();a['variant']=variant;sensitivity.append(a)
            recent=z[z.year.between(*c['comparison_years'])&z.team_size.gt(0)]
            ti=np.searchsorted(TEAM_BINS,recent.team_size.to_numpy(),side='left')-1
            rr=np.searchsorted(REF_BINS,recent.joint_refs.to_numpy(),side='left')-1
            np.add.at(surface[:,:,0],(ti,rr),1);np.add.at(surface[:,:,1],(ti,rr),recent.joint_cross.to_numpy())
            hh=hash64(z.paper_id.to_numpy(),c['seed']);limit=np.uint64(2**64-1) if samplehash is None or len(samplehash)<c['plot_sample_n'] else samplehash.max()
            select=hh<=limit;candidate=z.loc[select].copy();candidate['_hash']=hh[select]
            sample=candidate if sample is None else pd.concat([sample,candidate],ignore_index=True)
            if len(sample)>c['plot_sample_n']:
                take=np.argpartition(sample._hash.to_numpy(),c['plot_sample_n']-1)[:c['plot_sample_n']];sample=sample.iloc[take].copy()
            samplehash=sample._hash.to_numpy()
            rexport=z[rcols].copy()
            if c['r_export_fraction']<1:
                u=hash64(rexport.paper_id.to_numpy(),c['seed']+17).astype(np.float64)/2**64
                rexport=rexport.loc[u<c['r_export_fraction']].copy()
            rexport['sample_weight']=1/c['r_export_fraction'];rw.write(rexport)
            check_memory(c,'descriptive export')
            if offset%(25*c['batch_rows'])<c['batch_rows']:print(f'Description pass: {offset:,} core records',flush=True)
    if sample is None or sample.empty:raise ValueError('No P* papers reached description stage.')
    sample.sort_values('_hash',inplace=True,ignore_index=True)
    with Writer(d['data']/'plot_sample.parquet') as w:w.write(sample)
    a=pd.concat(yearrows).groupby(level=0).sum()
    for col in a.columns.tolist():
        if col.endswith('_sum'):
            stem=col[:-4];a[stem+'_mean']=ratio(a[col],a[stem+'_count'])
    a['q_event_mean']=ratio(a.q_event_n,a.event_d);a=a.reset_index();csv(a,d['tables']/'paper_annual.csv')
    endpoints=[]
    for start,end in [tuple(c['baseline_years']),(c['comparison_years'][0],2023),tuple(c['comparison_years'])]:
        t=a[a.year.between(start,end)]
        endpoints.append(dict(start_year=start,end_year=end,N=int(t.N.sum()),
            mean_q=float(t.joint_cross_sum.sum()/t.joint_cross_count.sum()) if t.joint_cross_count.sum() else np.nan,
            label='complete end window' if end==2024 else 'endpoint comparison'))
    csv(pd.DataFrame(endpoints),d['tables']/'period_endpoint_comparison.csv')
    h=pd.concat(hist,ignore_index=True)
    # n_zero/n_total are repeated metadata within each histogram, not additive across bins.
    h=h.groupby(['year','international','metric','bin','lo','hi'],as_index=False)[['n_positive','n_zero','n_total']].sum()
    csv(h,d['tables']/'histograms.csv')
    s=pd.concat(sensitivity,ignore_index=True).groupby(['year','variant'],as_index=False)[['count','sum']].sum();s['mean']=ratio(s['sum'],s['count']);csv(s,d['tables']/'definition_sensitivity.csv')
    rows=[]
    for i in range(8):
        for j in range(8):
            n,total=surface[i,j];rows.append(dict(team_bin=i,reference_bin=j,team_label=TEAM_LABELS[i],reference_label=REF_LABELS[j],
                n=int(n),mean=total/n if n else np.nan))
    csv(pd.DataFrame(rows),d['tables']/'response_surface.csv')
    z,w=roles(c);field_tables(c);full_n=population_rca(c);semantic_joint(c,sample)
    # Cohort intensity is bridge credit / paper-normalised exposure, not citation counts.
    cohort=pd.read_csv(d['tables']/'reference_cohorts.csv');cohort['citing_period']=cohort.citing_year//5*5
    cohort['source_cohort']=np.where(cohort.cited_year<1950,1945,cohort.cited_year//5*5)
    cohort=cohort.groupby(['citing_period','source_cohort'],as_index=False)[['E','B']].sum();cohort['intensity']=ratio(cohort.B,cohort.E)
    csv(cohort,d['tables']/'cohort_intensity.csv')
    csv(pd.DataFrame([{'table':'plot_sample','n':len(sample),'scope':'Bottom-k uniform deterministic hash sample from all P* papers; no stratification or oversampling.'},
        {'table':'r_papers','n':rw.n,'scope':f'All P* or uniform deterministic hash fraction {c["r_export_fraction"]}; complete cases selected per model in R.'},
        {'table':'full_core_rca','n':full_n,'scope':'Known country and Fields, 1990-2024; does not require >=2 usable references.'}]),d['tables']/'sample_scope.csv')
    json_write({'research_type':'Descriptive and conditional association; no causal effects or experiments.',
        'main_population':'P*: dated 1990-2024 core papers, focal country/Fields known, at least two jointly known core references.',
        'unknown_date':'Retained in metadata and assembly tables but not dated comparisons.',
        'core_completion_bounds':'Conditional on P*, bounded only over the observed core induced reference list; not all references in the published article.',
        'existing_semantic':'Broader original reference set; never labelled same-reference semantic validation.',
        'role_universe':'Country x Fields units positive in either role in at least one observed period; held fixed for concentration comparisons.'},d['status']/'interpretation.json')
    return dict(main_papers=int(a.N.sum()),plot_sample=len(sample),r_export=rw.n,full_core_rca_papers=full_n,
                parquet_files=4,advanced_main_figures=5)
