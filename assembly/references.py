import gc
import os
import numpy as np
from .runtime import arr,Lookup,source,batches,ids,dirs,save_array,check_memory
from .kernels import increment,scatter,sort_unique_rows

def edge_batches(c,lookup,year,statistics=False):
    ec=c['edge_columns'];stats=dict(input_edges=0,unmatched=0,unknown_year=0,self_edges=0,future_references=0,accepted_before_dedup=0)
    for b in batches(source(c,'edges'),[ec['citing'],ec['cited']],c['batch_rows']):
        i=lookup.get(ids(b[ec['citing']]));j=lookup.get(ids(b[ec['cited']]))
        stats['input_edges']+=len(i);matched=(i>=0)&(j>=0);stats['unmatched']+=int((~matched).sum())
        i=i[matched];j=j[matched];yi=year[i];yj=year[j]
        dated=(yi>0)&(yj>0);selfedge=i==j;future=(yj>yi)&dated
        stats['unknown_year']+=int((~dated).sum());stats['self_edges']+=int((dated&selfedge).sum());stats['future_references']+=int(future.sum())
        keep=dated&~selfedge&~future;ii=i[keep].astype(np.uint32);jj=j[keep].astype(np.uint32)
        stats['accepted_before_dedup']+=len(ii)
        yield ii,jj,stats

def build(c):
    d=dirs(c);paper=arr(c,'paper_id');year=arr(c,'year');n=len(paper)
    lookup=Lookup(paper,c['lookup']);counts=np.zeros(n,dtype=np.int64);stats={}
    for k,(i,j,stats) in enumerate(edge_batches(c,lookup,year,True)):
        increment(counts,i)
        if k%25==0:print(f'Count pass: {stats["input_edges"]:,} input edges',flush=True)
        check_memory(c,'reference count')
    ptr=np.empty(n+1,dtype=np.int64);ptr[0]=0;np.cumsum(counts,out=ptr[1:]);total=int(ptr[-1])
    if total==0:raise ValueError('No dated valid core-to-core references. Check edge direction/IDs.')
    path=d['cache']/'ref_index.npy';tmp=path.with_name(path.name+'.building')
    target=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint32,shape=(total,))
    cursor=ptr[:-1].copy()
    for k,(i,j,s) in enumerate(edge_batches(c,lookup,year)):
        scatter(cursor,target,i,j)
        if k%25==0:print(f'Fill pass: {s["input_edges"]:,} input edges',flush=True)
        check_memory(c,'reference fill')
    if not np.array_equal(cursor,ptr[1:]):raise AssertionError('CSR edge counts changed between scans.')
    del cursor,lookup;gc.collect()
    unique=np.empty(n,dtype=np.uint32)
    print('Sorting and removing repeated references within each citing paper.',flush=True)
    sort_unique_rows(ptr,target,unique)
    target.flush();del target;os.replace(tmp,path)
    save_array(c,'ref_ptr',ptr);save_array(c,'ref_count',unique)
    stats['unique_valid_core_edges']=int(unique.sum(dtype=np.int64))
    stats['duplicates_removed']=total-stats['unique_valid_core_edges']
    stats['max_unique_reference_count']=int(unique.max())
    stats['scope']='Dated, non-self, temporally consistent core induced graph; same-year references allowed.'
    return stats
