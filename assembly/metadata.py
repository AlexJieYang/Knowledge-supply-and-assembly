import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .runtime import (FIELDS,source,dirs,batches,ids,numeric,Lookup,save_array,arr,
                      csv,json_write,Writer,check_memory)

BASE = ['paper_id','year','source_id','primary_field_id','is_conference','team_size']

def preflight(c):
    d=dirs(c);rows=[]
    for k in c['paths']:
        p=source(c,k)
        if p.is_file():
            pf=pq.ParquetFile(p)
            rows.append(dict(input=k,path=str(p),rows=pf.metadata.num_rows,
                             columns=';'.join(pf.schema_arrow.names),gib=p.stat().st_size/2**30))
        else:rows.append(dict(input=k,path=str(p),rows=0,columns='MISSING',gib=0))
    t=pd.DataFrame(rows);csv(t,d['tables']/'input_catalog.csv')
    needed=['papers','countries','edges']+(['fields'] if c['field_mode']=='multilabel' else [])
    missing=[str(source(c,k)) for k in needed if not source(c,k).is_file()]
    print(t.to_string(index=False));print('RSS / cap GiB:',check_memory(c))
    if missing: raise FileNotFoundError('Required inputs missing:\n'+'\n'.join(missing)+'\nInspect results/tables/input_catalog.csv and edit config.paths. No primary-field fallback is applied.')
    schemas={k:pq.ParquetFile(source(c,k)).schema_arrow.names for k in needed}
    for key,columns in [('papers',['paper_id','year','primary_field_id']),
                        ('countries',list(c['country_columns'].values())),
                        ('edges',list(c['edge_columns'].values()))]:
        absent=set(columns)-set(schemas[key])
        if absent:raise KeyError(f'{key}: missing {sorted(absent)}; actual={schemas[key]}. Configure column names before starting 01.')
    if c['field_mode']=='multilabel':
        names=schemas['fields'];cols=c['field_columns']
        resolve(names,cols['paper'],['paper_id','work_id','PaperID'])
        resolve(names,cols['field'],['field_id','category_id','concept_id','field','Fields'])
        resolve(names,cols['level'],['level','field_level','concept_level'],False)
    return t

def resolve(names,requested,candidates,required=True):
    if requested!='auto':
        if requested in names:return requested
        if required:raise KeyError(f'Column {requested} missing; actual columns={names}')
        return None
    found=[n for n in candidates if n in names]
    if len(found)==1:return found[0]
    if len(found)>1:raise ValueError(f'Ambiguous columns {found}; select one in config.json.')
    if required:raise KeyError(f'Expected one of {candidates}; actual columns={names}. Edit config.json explicitly.')
    return None

def field_ids(s):
    if pd.api.types.is_numeric_dtype(s):return pd.to_numeric(s,errors='coerce')
    return pd.to_numeric(s.astype('string').str.replace(r'^https?://openalex.org/C','',regex=True).str.replace(r'^C','',regex=True),errors='coerce')

def attach_many(c,lookup,key,columns,outputs):
    n=len(lookup.arr);arrays={col:np.full(n,np.nan,dtype=np.float32) for col in columns}
    p=source(c,key)
    if p.is_file():
        names=pq.ParquetFile(p).schema_arrow.names;wanted=[col for col in columns if col in names]
        if 'paper_id' in names and wanted:
            seen=np.zeros(n,dtype=bool)
            for b in batches(p,['paper_id']+wanted,c['batch_rows']):
                ix=lookup.get(ids(b.paper_id));ok=ix>=0;ix=ix[ok]
                if pd.Index(ix).has_duplicates or seen[ix].any():raise ValueError(f'Duplicate core paper ID in {p}')
                seen[ix]=True
                for col in wanted:arrays[col][ix]=numeric(b.loc[ok,col])
                check_memory(c,'attach '+key)
        else:print(f'Optional {key} not attached: no compatible columns.',flush=True)
    else:print(f'Optional {key} absent; primary population retained.',flush=True)
    counts={}
    for col,out in zip(columns,outputs):
        a=arrays[col]
        if col=='AIProb' and np.any(np.isfinite(a)&((a<0)|(a>1))):raise ValueError('AIProb outside [0,1].')
        save_array(c,out,a);counts[col]=int(np.isfinite(a).sum())
    return counts

def membership(c,lookup,kind,field_codes=None):
    n=len(lookup.arr);frames=[];d=dirs(c)
    if kind=='country':
        p=source(c,'countries');cols=c['country_columns']
        pn,cn,wn=cols['paper'],cols['country'],cols['weight'];ln=None
    else:
        p=source(c,'fields');names=pq.ParquetFile(p).schema_arrow.names;cols=c['field_columns']
        pn=resolve(names,cols['paper'],['paper_id','work_id','PaperID'])
        cn=resolve(names,cols['field'],['field_id','category_id','concept_id','field','Fields'])
        wn=resolve(names,cols['weight'],['weight','field_weight','credit'],False)
        ln=resolve(names,cols['level'],['level','field_level','concept_level'],False)
    use=list(dict.fromkeys([pn,cn]+([wn] if wn else [])+([ln] if ln else [])))
    raw=0;level_dropped=0
    for b in batches(p,use,c['batch_rows']):
        raw+=len(b)
        if ln is not None:
            keep=pd.to_numeric(b[ln],errors='coerce').eq(c['field_level'])
            level_dropped+=int((~keep).sum());b=b.loc[keep]
        if len(b)==0:continue
        if kind=='field':
            first=next((x for x in b[cn] if isinstance(x,(list,np.ndarray,tuple)) or pd.notna(x)),None)
            if isinstance(first,(list,np.ndarray,tuple)):
                if wn:raise ValueError('List-valued Fields with a separate weight column requires a long-form table. Set config.paths.fields to that table.')
                b=b.explode(cn,ignore_index=True)
            b=b.assign(_code=field_ids(b[cn]))
            b=b.loc[b._code.isin(field_codes)].copy()
            b['_code']=b._code.map(field_codes).astype(np.int16)
        else:
            b=b.loc[b[cn].notna()].copy()
            b['_code']=b[cn].astype('string').str.strip().str.upper()
            if not b._code.str.fullmatch(r'[A-Z]{2,3}').all():raise ValueError('Country entities must be ISO-style country codes; configure country_columns.country.')
        if len(b)==0:continue
        ix=lookup.get(ids(b[pn]));ok=ix>=0;b=b.loc[ok].copy();ix=ix[ok]
        if len(ix)==0:continue
        w=numeric(b[wn],'float64') if wn else np.ones(len(b),dtype=float)
        if kind=='country' or c['field_weighting']=='input':
            if not np.isfinite(w).all() or (w<=0).any():raise ValueError(f'{kind}: invalid credit weights.')
        else:w=np.ones(len(b),dtype=float)
        frames.append(pd.DataFrame({'row':ix.astype(np.uint32),'code':b._code.to_numpy(),'weight':w}))
        if raw%(25*c['batch_rows'])<c['batch_rows']:print(f'{kind}: {raw:,} rows',flush=True)
        check_memory(c,'read '+kind)
    if not frames:raise ValueError(f'No {kind} assignments matched. Inspect table schema, paper IDs and the configured top-level Fields IDs.')
    t=pd.concat(frames,ignore_index=True);del frames;gc.collect()
    if kind=='country':
        labels=sorted(t.code.unique().tolist());mapping={x:i for i,x in enumerate(labels)}
        t['code']=t.code.map(mapping).astype(np.int16)
        csv(pd.DataFrame({'country':np.arange(len(labels)),'country_code':labels}),d['tables']/'countries.csv')
    else: labels=list(field_codes)
    t.sort_values(['row','code'],kind='stable',inplace=True,ignore_index=True)
    duplicate=t.duplicated(['row','code'])
    nd=int(duplicate.sum())
    if nd and kind=='country':raise ValueError(f'{nd:,} duplicate paper-country assignments. Country weights must already represent distinct countries.')
    if nd:t=t.loc[~duplicate].copy()
    ix=t.row.to_numpy();codes=t.code.to_numpy(np.int16);w=t.weight.to_numpy(float)
    start=np.r_[0,np.flatnonzero(ix[1:]!=ix[:-1])+1]
    sums=np.add.reduceat(w,start)
    if kind=='country' and not np.allclose(sums,1,atol=1e-5,rtol=0):
        raise ValueError('country_members credits do not sum to 1. Do not silently change their scientific meaning.')
    lengths=np.diff(np.r_[start,len(t)])
    # Normalise rounding noise, or equal distinct-label field credits.
    w=w/np.repeat(sums,lengths)
    counts=np.zeros(n,dtype=np.uint32);counts[ix[start]]=lengths
    ptr=np.empty(n+1,dtype=np.int64);ptr[0]=0;np.cumsum(counts,out=ptr[1:])
    save_array(c,kind+'_ptr',ptr);save_array(c,kind+'_code',codes);save_array(c,kind+'_weight',w)
    return dict(input_rows=raw,matched_assignments=len(t),distinct_papers=len(start),
                duplicates_removed=nd,level_rows_excluded=level_dropped,categories=len(labels))

def build(c):
    preflight(c);d=dirs(c);parts=[];future=unknown=raw=0
    names=pq.ParquetFile(source(c,'papers')).schema_arrow.names
    required=['paper_id','year','primary_field_id']
    if not set(required)<=set(names):raise KeyError('Required core columns: '+str(required))
    for b in batches(source(c,'papers'),[x for x in BASE if x in names],c['batch_rows']):
        raw+=len(b);y=numeric(b.year,'float64');dated=np.isfinite(y)&(y>=1)&(y%1==0)&(y<=32767)
        keep=~dated|(y<=c['end_year']);future+=int((~keep).sum());b=b.loc[keep].copy();y=y[keep];dated=dated[keep]
        unknown+=int((~dated).sum());b['paper_id']=ids(b.paper_id)
        b['year']=np.where(dated,y,0).astype(np.int16)
        for col in ['team_size','is_conference']:
            b[col]=numeric(b[col]) if col in b else np.full(len(b),np.nan,dtype=np.float32)
        b['primary_field_id']=field_ids(b.primary_field_id).fillna(-1).astype(np.int64)
        if 'source_id' in b:
            s=b.source_id
            if pd.api.types.is_numeric_dtype(s):known=s.notna()&s.gt(0)
            else:known=s.notna()&~s.astype('string').isin(['','0','-1'])
            vv=np.zeros(len(b),dtype=np.uint64)
            if known.any():vv[known]=ids(s.loc[known])
            b['source_id']=vv
        else:b['source_id']=np.zeros(len(b),dtype=np.uint64)
        parts.append(b)
        check_memory(c,'core columns')
    if not parts:raise ValueError('Empty core population.')
    m=pd.concat(parts,ignore_index=True);del parts;gc.collect()
    if len(m)>=2**32:raise ValueError('This compact row index supports fewer than 2^32 core papers.')
    lookup=Lookup(m.paper_id.to_numpy(),c['lookup']);print(f'Core through 2024 plus unknown dates: {len(m):,}',flush=True)
    fids=[int(f) for f in c['field_ids']];fm={f:i for i,f in enumerate(fids)}
    csv(pd.DataFrame({'field':np.arange(len(fids)),'field_id':fids,
         'field_label':[c['field_labels'].get(str(f),FIELDS.get(f,str(f))) for f in fids]}),d['tables']/'fields.csv')
    m['primary_field']=m.primary_field_id.map(fm).fillna(-1).astype(np.int16)
    m['source_code'],source_ids=pd.factorize(m.source_id,sort=True)
    m.loc[m.source_id.eq(0),'source_code']=-1;m['source_code']=m.source_code.astype(np.int32)
    csv(pd.DataFrame({'source_code':np.arange(len(source_ids)),'source_id':source_ids}),d['tables']/'sources.csv')
    for name,col in [('paper_id','paper_id'),('year','year'),('primary','primary_field'),('team','team_size'),('source_code','source_code')]:
        save_array(c,name,m[col].to_numpy())
    save_array(c,'kind',np.where(m.is_conference.eq(1),1,np.where(m.is_conference.eq(0),0,2)).astype(np.int8))
    country=membership(c,lookup,'country')
    if c['field_mode']=='multilabel':fields=membership(c,lookup,'field',fm)
    else:
        good=m.primary_field.to_numpy()>=0;count=good.astype(np.int64)
        ptr=np.r_[0,np.cumsum(count)];save_array(c,'field_ptr',ptr)
        save_array(c,'field_code',m.loc[good,'primary_field'].to_numpy(np.int16));save_array(c,'field_weight',np.ones(int(good.sum())))
        fields=dict(categories=len(fids),distinct_papers=int(good.sum()),mode='explicit primary-field sensitivity')
    ai_n=attach_many(c,lookup,'ai',['AIProb'],['ai'])['AIProb']
    semantic_n=attach_many(c,lookup,'semantic',['emb_ref_pair_mean_distance','emb_ref_coverage'],
        ['semantic_existing','semantic_coverage'])['emb_ref_pair_mean_distance']
    cp=arr(c,'country_ptr');fp=arr(c,'field_ptr')
    m['known_countries']=np.diff(cp).astype(np.int16);m['known_fields']=np.diff(fp).astype(np.int16)
    m.drop(columns=['source_id','primary_field_id'],inplace=True)
    with Writer(d['data']/'metadata.parquet') as w:
        for start in range(0,len(m),c['batch_rows']):w.write(m.iloc[start:start+c['batch_rows']])
    summary=dict(input_core=raw,retained_core=len(m),excluded_future=future,unknown_date=unknown,
        country=country,fields=fields,ai_known=ai_n,existing_semantic_known=semantic_n,
        country_credit='Distinct observed institutional countries; input weights, rounding renormalised.',
        field_credit=c['field_weighting'],field_mode=c['field_mode'],
        population='Existing core journal/conference corpus, not all global publications.')
    json_write(summary,d['status']/'population.json');return summary
