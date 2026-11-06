from pathlib import Path
import gc
import ast
import hashlib
import json
import os
import time
import uuid
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from threadpoolctl import threadpool_limits
from .paths import project_path, input_path

ROOT = Path(__file__).resolve().parents[1]
_LIMIT = threadpool_limits(limits=1)
pa.set_cpu_count(2)
pa.set_io_thread_count(2)

FIELDS = {
 121332964:'Physics',127313418:'Geology',127413603:'Engineering',138885662:'Philosophy',
 142362112:'Art',144024400:'Sociology',144133560:'Business',15744967:'Psychology',
 162324750:'Economics',17744445:'Political science',185592680:'Chemistry',
 192562407:'Materials science',205649164:'Geography',33923547:'Mathematics',
 39432304:'Environmental science',41008148:'Computer science',71924100:'Medicine',
 86803240:'Biology',95457728:'History'}
STAGES = ['metadata','references','metrics','networks','descriptions','figures']
DEPENDENCIES = {'metadata':[], 'references':['metadata'], 'metrics':['references'],
                'networks':['metrics'], 'descriptions':['networks'], 'figures':['descriptions']}

def config(path=None):
    p=Path(path or os.environ.get('GKA_CONFIG', ROOT/'config.json')).expanduser().resolve()
    c=json.loads(p.read_text(encoding='utf-8-sig'))
    c['_project_dir']=str(p.parent)
    # Optional embedding inputs follow the same configuration-relative convention.
    for key in ('index_parquet','matrix_npy'):
        value=c.get('joint_semantic',{}).get(key)
        if value:c['joint_semantic'][key]=str(project_path(c,value).resolve())
    if c['end_year']!=2024: raise ValueError('This design uses an end date of 2024.')
    if not 0 < c['r_export_fraction'] <= 1: raise ValueError('r_export_fraction must be in (0,1].')
    if c['field_mode'] not in ('multilabel','primary'): raise ValueError('field_mode must be multilabel or primary.')
    if c['lookup'] not in ('pandas','sorted'): raise ValueError('lookup must be pandas or sorted.')
    if not 1<=c['threads']<=24: raise ValueError('Choose 1..24 Numba threads, independently of BLAS=1.')
    if c['trend_start']%5 or c['end_year']%5!=4: raise ValueError('Use complete natural five-year windows.')
    if c['baseline_years'] != [2010,2014] or c['comparison_years'] != [2020,2024]:
        raise ValueError('This study fixes baseline=2010-2014 and comparison=2020-2024; update the study/figure definitions before changing these windows.')
    for key in ('baseline_years','comparison_years'):
        lo,hi=c[key]
        if lo%5 or hi!=lo+4 or lo<c['trend_start'] or hi>c['end_year']:
            raise ValueError(key+' must be one complete observed natural five-year window.')
    if not c['trend_start']<=c['fixed_support_start']<=c['end_year']:raise ValueError('Invalid fixed_support_start.')
    if not any(abs(float(g)-1)<1e-12 for g in c['community_resolutions']):raise ValueError('Include resolution=1.0.')
    if c['community_repeats']<2:raise ValueError('Use at least two optimisation repeats.')
    if c['field_weighting'] not in ('equal_distinct','input'):raise ValueError('field_weighting must be equal_distinct or input.')
    if not c['field_ids'] or len(set(c['field_ids']))!=len(c['field_ids']):raise ValueError('Field IDs must be a nonempty unique list.')
    import numba
    if c['threads']>numba.config.NUMBA_NUM_THREADS:
        raise RuntimeError('Restart the kernel with ASSEMBLY_THREADS at least config.threads, before importing Numba.')
    numba.set_num_threads(c['threads'])
    out=dirs(c)['root'].resolve(); inp=project_path(c,c['research_dir']).resolve()
    if out==inp or inp in out.parents:
        raise ValueError('output_dir must be outside the upstream research directory.')
    return c

def dirs(c):
    root=project_path(c,c['output_dir'])
    d={'root':root}
    for k in ('cache','data','tables','figures','r','status'):
        d[k]=root/k;d[k].mkdir(parents=True,exist_ok=True)
    return d

def source(c,key):
    return input_path(c,key)

def json_write(obj,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.building')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    os.replace(tmp,p)

def csv(df,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(p,index=False,encoding='utf-8-sig',float_format='%.12g')

def batches(path,columns=None,size=1000000):
    pf=pq.ParquetFile(path)
    absent=set(columns or [])-set(pf.schema_arrow.names)
    if absent: raise KeyError(f'{path}: missing {sorted(absent)}; actual schema: {pf.schema_arrow.names}')
    for b in pf.iter_batches(batch_size=size,columns=columns,use_threads=False):
        yield b.to_pandas()

def ids(s):
    s=pd.Series(s,copy=False)
    if s.isna().any(): raise ValueError('Null paper ID.')
    if pd.api.types.is_float_dtype(s):
        if (s.abs()>=2**53).any() or (s%1!=0).any(): raise ValueError('Unsafe float paper ID.')
    elif not pd.api.types.is_numeric_dtype(s):
        s=s.astype('string').str.replace(r'^https?://openalex.org/W','',regex=True).str.replace(r'^W','',regex=True)
        if not s.str.fullmatch(r'\d+').all(): raise ValueError('Unrecognised paper IDs; configure the original exact integer/text column.')
    if (pd.to_numeric(s,errors='raise')<=0).any(): raise ValueError('IDs must be positive.')
    return s.to_numpy(dtype=np.uint64)

def numeric(s,dtype='float32'):
    return pd.to_numeric(s,errors='coerce').to_numpy(dtype=dtype,na_value=np.nan)

class Lookup:
    def __init__(self,arr,mode='pandas'):
        self.arr=np.asarray(arr)
        self.mode=mode
        if mode=='pandas':
            self.index=pd.Index(self.arr)
            if not self.index.is_unique: raise ValueError('Non-unique paper IDs.')
        else:
            self.order=np.argsort(self.arr)
            self.sorted=self.arr[self.order]
            if np.any(self.sorted[1:]==self.sorted[:-1]): raise ValueError('Non-unique paper IDs.')
    def get(self,values):
        v=np.asarray(values,dtype=np.uint64)
        if self.mode=='pandas': return self.index.get_indexer(v)
        p=np.searchsorted(self.sorted,v)
        ok=p<len(self.sorted);safe=np.minimum(p,len(self.sorted)-1)
        ok &= self.sorted[safe]==v
        ans=np.full(len(v),-1,dtype=np.int64);ans[ok]=self.order[p[ok]]
        return ans

def save_array(c,name,a):
    p=dirs(c)['cache']/(name+'.npy');tmp=p.with_name(p.name+'.building')
    with tmp.open('wb') as f: np.save(f,a,allow_pickle=False)
    os.replace(tmp,p)

def arr(c,name,mode='r'):
    return np.load(dirs(c)['cache']/(name+'.npy'),mmap_mode=mode,allow_pickle=False)

def mapped_array(c,name,dtype,shape):
    p=dirs(c)['cache']/(name+'.npy')
    return np.lib.format.open_memmap(p,mode='w+',dtype=dtype,shape=shape)

def check_memory(c,label=''):
    try:
        physical=psutil.virtual_memory().total/2**30
    except (OSError,RuntimeError):
        physical=os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')/2**30
    try:
        rss=psutil.Process().memory_info().rss/2**30
    except (OSError,psutil.Error):
        import resource
        import sys
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss=peak/(2**30 if sys.platform=='darwin' else 2**20)
    cap=min(c['max_rss_gib'],physical*c['physical_memory_fraction'])
    if rss>cap: raise MemoryError(f'{label}: RSS={rss:.1f} GiB, limit={cap:.1f} GiB. Close other kernels; reduce batch_rows/metric_batch_papers or use lookup=sorted.')
    return round(rss,2),round(cap,2)

class Writer:
    def __init__(self,path):
        self.path=Path(path);self.tmp=self.path.with_name(self.path.name+'.building')
        self.w=None;self.schema=None;self.n=0
    def __enter__(self): return self
    def write(self,df):
        t=pa.Table.from_pandas(df,preserve_index=False)
        if self.w is None:
            self.schema=t.schema
            self.w=pq.ParquetWriter(self.tmp,self.schema,compression='zstd',compression_level=3)
        elif t.schema!=self.schema: t=t.cast(self.schema)
        if len(df): self.w.write_table(t,row_group_size=len(df))
        self.n+=len(df)
    def __exit__(self,typ,value,tb):
        if self.w is not None: self.w.close()
        if typ is None and self.w is not None:
            if pq.ParquetFile(self.tmp).metadata.num_rows!=self.n: raise IOError('Parquet row conservation failed.')
            os.replace(self.tmp,self.path)
        elif self.tmp.exists(): self.tmp.unlink()

def ratio(a,b):
    a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
    return np.divide(a,b,out=np.full(a.shape,np.nan),where=b>0)

def hash64(v,seed):
    x=np.asarray(v,np.uint64)^np.uint64(seed)
    with np.errstate(over='ignore'):
        x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
    return x^(x>>np.uint64(31))

def marker(c,stage): return dirs(c)['status']/(stage+'.json')

CONFIG_KEYS = {
 'metadata':['research_dir','end_year','field_mode','field_weighting','field_ids','field_columns','field_level','country_columns'],
 'references':['research_dir','edge_columns','end_year'],
 'metrics':['trend_start','end_year'],
 'networks':['trend_start','end_year','baseline_years','comparison_years','min_null_pool','community_resolutions','community_repeats','seed'],
 'descriptions':['trend_start','end_year','baseline_years','comparison_years','fixed_support_start','plot_sample_n','r_export_fraction','min_cell_mass','seed','joint_semantic'],
 'figures':['comparison_years','baseline_years','min_cell_mass','min_surface_n','field_labels','synthetic']}
INPUT_KEYS = {'metadata':['papers','countries','fields','ai','semantic'],'references':['edges']}
KERNEL_KEYS = {'references':['increment','scatter','sort_unique_rows'],
 'metrics':['membership_sum','metric_batch'],
 'networks':['age_stratum','build_pools','aggregate_networks'],
 'descriptions':['field_summaries','core_output']}

def signature(c,stage):
    science={k:c.get(k) for k in CONFIG_KEYS[stage]};sources={}
    for key in INPUT_KEYS.get(stage,[]):
        p=source(c,key)
        sources[key]=[str(p),p.stat().st_size,p.stat().st_mtime_ns] if p.is_file() else [str(p),None,None]
    if stage=='descriptions' and c['joint_semantic']['enabled']:
        for key in ('index_parquet','matrix_npy'):
            p=Path(c['joint_semantic'][key])
            sources[key]=[str(p),p.stat().st_size,p.stat().st_mtime_ns] if p.is_file() else [str(p),None,None]
    deps={s:json.loads(marker(c,s).read_text(encoding='utf-8'))['token'] if marker(c,s).exists() else None for s in DEPENDENCIES[stage]}
    h=hashlib.sha256(json.dumps([science,sources,deps],sort_keys=True).encode())
    for f in [stage+'.py','runtime.py']:
        p=Path(__file__).parent/f
        if p.exists():h.update(p.read_bytes())
    if stage in KERNEL_KEYS:
        p=Path(__file__).with_name('kernels.py');text=p.read_text(encoding='utf-8');lines=text.splitlines()
        for node in ast.parse(text).body:
            if isinstance(node,ast.FunctionDef) and node.name in KERNEL_KEYS[stage]:
                start=min([node.lineno]+[d.lineno for d in node.decorator_list])
                h.update('\n'.join(lines[start-1:node.end_lineno]).encode())
    return h.hexdigest()

def run(c,stage,force=False):
    import importlib
    def check_upstream(target):
        for dep in DEPENDENCIES[target]:
            check_upstream(dep)
            p=marker(c,dep)
            if not p.exists(): raise RuntimeError(f'Run {dep} first.')
            old=json.loads(p.read_text(encoding='utf-8'))
            if old['signature']!=signature(c,dep): raise RuntimeError(f'{dep} inputs/configuration changed. Rerun that stage first.')
    check_upstream(stage)
    sig=signature(c,stage);p=marker(c,stage)
    if not force and p.exists() and json.loads(p.read_text(encoding='utf-8')).get('signature')==sig:
        print(stage+': complete; reusing the current results. force=True recomputes this stage.');return
    p.unlink(missing_ok=True)
    t=time.monotonic();print(stage+': start',flush=True)
    result=importlib.import_module('.'+stage,package='assembly').build(c)
    json_write({'stage':stage,'token':uuid.uuid4().hex,'signature':sig,
                'minutes':round((time.monotonic()-t)/60,3),'result':result},p)
    gc.collect();print(f'{stage}: finished in {(time.monotonic()-t)/60:.1f} min',flush=True)
    return result
