import gc
import numpy as np
import pandas as pd
from .runtime import arr,mapped_array,dirs,csv,save_array,Writer,check_memory
from .kernels import metric_batch

METRICS=['joint_refs','geo_cross','field_cross','joint_cross','primary_joint_cross',
         'mean_ref_age_joint','completion_lower','completion_upper']

def shared_arrays(c):
    return [arr(c,n) for n in ['year','primary','country_ptr','country_code','country_weight',
                               'field_ptr','field_code','field_weight','ref_ptr','ref_count','ref_index']]

def build(c):
    d=dirs(c);year,primary,cp,cc,cw,fp,fc,fw,rp,rn,ri=shared_arrays(c)
    n=len(year);nc=len(pd.read_csv(d['tables']/'countries.csv'));nf=len(pd.read_csv(d['tables']/'fields.csv'))
    maximum=int(rn.max());sigma=mapped_array(c,'sigma',np.float32,(len(ri),));sigma[:]=np.nan
    measure=mapped_array(c,'paper_metrics',np.float64,(n,len(METRICS)))
    eligible=mapped_array(c,'eligible',np.bool_,(n,));pids=arr(c,'paper_id')
    team=arr(c,'team');sources=arr(c,'source_code');ai=arr(c,'ai');sem=arr(c,'semantic_existing');semcov=arr(c,'semantic_coverage')
    annual=[];ps=0
    with Writer(d['data']/'assembly_papers.parquet') as writer:
        for lo in range(0,n,c['metric_batch_papers']):
            hi=min(n,lo+c['metric_batch_papers'])
            z=metric_batch(lo,hi,c['threads']*2,nc,nf,maximum,year,primary,cp,cc,cw,fp,fc,fw,rp,rn,ri,sigma)
            measure[lo:hi]=z
            ok=(year[lo:hi]>=c['trend_start'])&(year[lo:hi]<=c['end_year'])&(np.diff(cp[lo:hi+1])>0)&(np.diff(fp[lo:hi+1])>0)&(z[:,0]>=2)
            eligible[lo:hi]=ok;ps+=int(ok.sum())
            for k in [1,2,3,4,6,7]:
                v=z[:,k];good=np.isfinite(v)
                if np.any((v[good]<-1e-6)|(v[good]>1+1e-6)):raise AssertionError(f'{METRICS[k]} outside [0,1].')
            finite=np.isfinite(z[:,3]);g=z[finite,1];f=z[finite,2];q=z[finite,3]
            if np.any(q>g+1e-6) or np.any(q>f+1e-6) or np.any(1-g-f+q < -1e-6):
                raise AssertionError('Reference-pair context masses are inconsistent.')
            df=pd.DataFrame(z,columns=METRICS)
            df['paper_id']=pids[lo:hi];df['year']=year[lo:hi];df['eligible']=ok
            df['known_countries']=np.diff(cp[lo:hi+1]).astype(np.int16)
            df['known_fields']=np.diff(fp[lo:hi+1]).astype(np.int16)
            df['reference_coverage']=np.divide(z[:,0],rn[lo:hi],out=np.full(hi-lo,np.nan),where=rn[lo:hi]>0)
            df['core_refs']=rn[lo:hi];df['team_size']=team[lo:hi];df['source_code']=sources[lo:hi]
            df['primary_field']=primary[lo:hi];df['AIProb']=ai[lo:hi]
            df['semantic_existing']=sem[lo:hi];df['semantic_coverage']=semcov[lo:hi]
            writer.write(df)
            yy=df.assign(country_known=df.known_countries.gt(0),field_known=df.known_fields.gt(0),one=1)
            annual.append(yy.groupby('year',sort=False)[['one','country_known','field_known','eligible']].sum())
            if lo%(10*c['metric_batch_papers'])==0:print(f'Assembly metrics: {hi:,}/{n:,}; P*: {ps:,}',flush=True)
            check_memory(c,'assembly metrics');del z,df,yy;gc.collect()
    if ps==0:raise ValueError('P* is empty. Check membership links, edge direction and trend_start.')
    sigma.flush();measure.flush();eligible.flush()
    a=pd.concat(annual).groupby(level=0).sum().reset_index();csv(a,d['tables']/'population_year.csv')
    selected=np.flatnonzero(eligible).astype(np.uint32);order=np.argsort(year[selected],kind='stable');selected=selected[order]
    counts=np.bincount(year[selected].astype(int)-c['trend_start'],minlength=c['end_year']-c['trend_start']+1)
    group_ptr=np.r_[0,np.cumsum(counts,dtype=np.int64)]
    save_array(c,'group_ptr',group_ptr);save_array(c,'group_rows',selected)
    return dict(core_rows=n,main_papers=ps,main_start=c['trend_start'],end_year=c['end_year'],
                reference_scope='Same jointly country/Fields-observed core references for all main measures.',
                sigma_dtype='float32; q is its paper mean; other accumulators float64',parquet_files=2)
