"""Labelled synthetic inputs for functional tests, never empirical findings."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def create(root,n=6000,seed=426):
    root=Path(root);rng=np.random.default_rng(seed)
    for folder in ('core','extended','stage/fields','stage/edges','metrics'):(root/folder).mkdir(parents=True,exist_ok=True)
    pids=np.arange(2**53+1000,2**53+1000+n,dtype=np.uint64)
    years=rng.integers(1985,2025,n,dtype=np.int16);years[:8]=[0,2025,2024,1990,2000,2010,2014,2023]
    fids=np.array([41008148,71924100,121332964,185592680,86803240,33923547],dtype=np.int64)
    primary=rng.choice(fids,n);primary[2]=-1
    countries=np.array(['US','CN','GB','DE','JP','IN','BR','ZA']);home=rng.integers(0,len(countries),n)
    paper=pd.DataFrame({'paper_id':pids,'year':years,'source_id':rng.integers(1,35,n,dtype=np.uint64),
        'primary_field_id':primary,'is_conference':rng.random(n)<.15,'team_size':rng.integers(1,25,n)})
    paper.to_parquet(root/'core/papers.parquet',index=False,row_group_size=317)
    cs=[];fs=[]
    for i in range(n):
        if rng.random()>.07:
            members=[home[i]]
            if rng.random()<.20:
                other=int(rng.integers(0,len(countries)));members=sorted(set([home[i],other]))
            for j in members:cs.append((pids[i],int(years[i]),countries[j],1/len(members)))
        if rng.random()>.06 and primary[i]>0:
            fields=[primary[i]]
            if rng.random()<.30:fields=sorted(set([primary[i],int(rng.choice(fids))]))
            for f in fields:fs.append((pids[i],f,0))
    pd.DataFrame(cs,columns=['paper_id','year','entity_id','weight']).sample(frac=1,random_state=3).to_parquet(root/'extended/country_members.parquet',index=False,row_group_size=411)
    pd.DataFrame(fs,columns=['paper_id','field_id','level']).sample(frac=1,random_state=4).to_parquet(root/'stage/fields/data.parquet',index=False,row_group_size=509)
    ei=[];ej=[]
    for i in range(n):
        if years[i]<=0:continue
        candidate=np.flatnonzero((years>0)&(years<=years[i])&(np.arange(n)!=i))
        if not len(candidate):continue
        weight=np.where((home[candidate]%2)==(home[i]%2),5.,1.)
        weight*=np.where(primary[candidate]==primary[i],1.8,1.)
        selected=rng.choice(candidate,size=min(int(rng.integers(1,19)),len(candidate)),replace=False,p=weight/weight.sum())
        ei.extend(pids[np.repeat(i,len(selected))]);ej.extend(pids[selected])
    edges=pd.DataFrame({'citing_id':np.asarray(ei,dtype=np.uint64),'cited_id':np.asarray(ej,dtype=np.uint64)})
    edges=pd.concat([edges,edges.iloc[:17],pd.DataFrame({'citing_id':pids[2:4],'cited_id':pids[2:4]})],ignore_index=True)
    edges.sample(frac=1,random_state=55).to_parquet(root/'stage/edges/data.parquet',index=False,row_group_size=433)
    pd.DataFrame({'paper_id':pids,'AIProb':rng.beta(.5,3,n)}).to_parquet(root/'extended/AI_papers.parquet',index=False,row_group_size=331)
    pd.DataFrame({'paper_id':pids,'emb_ref_pair_mean_distance':rng.uniform(.1,.6,n),
        'emb_ref_coverage':rng.uniform(.3,1,n)}).to_parquet(root/'metrics/embedding_paper.parquet',index=False,row_group_size=307)
    x=rng.normal(size=(n,8)).astype(np.float32);x[5]=0
    np.save(root/'embeddings.npy',x)
    pd.DataFrame({'paper_id':pids,'row':np.arange(n)}).to_parquet(root/'embedding_index.parquet',index=False)
    return root


def configuration(root,output,project=None):
    p=Path(project or Path(__file__).resolve().parents[1]);c=json.loads((p/'config.json').read_text(encoding='utf-8-sig'))
    c.update(research_dir=str(Path(root).resolve()),output_dir=str(Path(output).resolve()),_project_dir=str(p),
        batch_rows=701,metric_batch_papers=607,threads=2,max_rss_gib=12,plot_sample_n=2500,
        community_repeats=2,min_cell_mass=.05,min_surface_n=3,min_null_pool=1)
    return c
