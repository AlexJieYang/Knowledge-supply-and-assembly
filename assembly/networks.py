import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from .runtime import arr,dirs,csv,ratio,mapped_array,check_memory,json_write
from .kernels import build_pools,aggregate_networks
from .metrics import shared_arrays

LAYERS=['C','F','H','F_model','H_model','H0','H_context']
ROLE=['E_src','B_src','E_asm','B_asm','E_src_model','B_src_model','B0_src',
      'E_asm_model','B_asm_model','B0_asm','B_context_src','B_context_asm']

def partition_similarity(a,b):
    a=np.asarray(a);b=np.asarray(b);good=(a>=0)&(b>=0);a=a[good];b=b[good];n=len(a)
    if n<2:return np.nan,np.nan,n
    _,a=np.unique(a,return_inverse=True);_,b=np.unique(b,return_inverse=True)
    tab=np.zeros((a.max()+1,b.max()+1),dtype=int);np.add.at(tab,(a,b),1)
    aa=tab.sum(1);bb=tab.sum(0);p=tab/n
    with np.errstate(divide='ignore',invalid='ignore'):
        mi=float(np.sum(np.where(tab>0,p*np.log(np.maximum(tab,1)*n/(aa[:,None]*bb[None,:])),0)))
    ha=-np.sum(aa/n*np.log(aa/n));hb=-np.sum(bb/n*np.log(bb/n));emi=0.
    for x in aa:
        for y in bb:
            k=np.arange(max(1,x+y-n),min(x,y)+1)
            emi+=np.sum(k/n*np.log(k*n/(x*y))*hypergeom.pmf(k,n,int(x),int(y)))
    den=(ha+hb)/2-emi
    ami=(mi-emi)/den if abs(den)>1e-12 else (1. if abs(ha-hb)<1e-12 else 0.)
    return float(ami),float(max(0.,ha+hb-2*mi)/np.log(n)),n

def leiden(W,directed,gamma,seed):
    import igraph as ig
    import leidenalg as la
    W=np.asarray(W,float).copy();np.fill_diagonal(W,0)
    active=np.flatnonzero(W.sum(0)+W.sum(1)>0);membership=np.full(len(W),-1,dtype=int)
    if len(active)<2:return membership,np.nan
    a=W[np.ix_(active,active)];ii,jj=np.nonzero(a>0)
    if not directed:
        keep=ii<jj;ii=ii[keep];jj=jj[keep]
    g=ig.Graph(n=len(active),edges=list(zip(ii.tolist(),jj.tolist())),directed=directed)
    g.es['weight']=a[ii,jj].tolist()
    part=la.find_partition(g,la.RBConfigurationVertexPartition,weights='weight',
                          resolution_parameter=float(gamma),n_iterations=-1,seed=int(seed))
    membership[active]=part.membership
    return membership,float(part.quality())

def positions(W):
    W=(W+W.T)/2;np.fill_diagonal(W,0)
    deg=W.sum(1);inv=np.divide(1,np.sqrt(deg),out=np.zeros_like(deg),where=deg>0)
    lap=np.diag((deg>0).astype(float))-W*inv[:,None]*inv[None,:]
    vals,vec=np.linalg.eigh(lap)
    positive=np.flatnonzero(vals>1e-9)
    use=positive[:2]
    if len(use)<2:
        a=np.linspace(0,2*np.pi,len(W),endpoint=False);return np.c_[np.cos(a),np.sin(a)]
    xy=vec[:,use]
    for k in range(2):
        j=np.argmax(np.abs(xy[:,k]));xy[:,k]*=1 if xy[j,k]>=0 else -1
    scale=np.max(np.abs(xy),axis=0);return xy/np.where(scale>0,scale,1)

def node_roles(W,labels):
    W=np.asarray(W,float).copy();np.fill_diagonal(W,0)
    good=labels>=0;labels=np.asarray(labels);n=len(labels)
    p_out=np.full(n,np.nan);p_in=np.full(n,np.nan);z_out=np.full(n,np.nan);z_in=np.full(n,np.nan)
    retained=np.full(n,np.nan)
    for directed,p,z in [(W,p_out,z_out),(W.T,p_in,z_in)]:
        x=directed[:,good];total=x.sum(1);within=np.full(n,np.nan)
        for i in np.flatnonzero(good&(total>0)):
            s=np.bincount(labels[good],weights=x[i],minlength=labels[good].max()+1)
            p[i]=1-np.sum((s/total[i])**2);within[i]=s[labels[i]]
        for lab in np.unique(labels[good]):
            ix=(labels==lab)&np.isfinite(within);v=within[ix]
            if len(v):z[ix]=(v-v.mean())/v.std() if v.std()>0 else 0.
    total=(W+W.T).sum(1);keep=(W[:,good].sum(1)+W[good,:].sum(0))
    retained=ratio(keep,total)
    return p_out,p_in,z_out,z_in,retained

def analyse_graphs(c,network,countries):
    d=dirs(c);ny=network.shape[0];periods=np.arange(c['trend_start'],c['end_year']+1,5)
    window=np.stack([network[y:y+5].sum(0) for y in range(0,ny,5)])
    np.savez_compressed(d['cache']/'window_networks.npz',network=window,periods=periods)
    records=[];stability=[];best={};similarity=[];edges=[]
    for w,p in enumerate(periods):
        for li,layer in enumerate(LAYERS):
            a=window[w,li];ii,jj=np.nonzero(a>0)
            if layer=='C':mask=ii<jj;ii=ii[mask];jj=jj[mask]
            edges.append(pd.DataFrame({'period':int(p),'layer':layer,'source':ii,'target':jj,'weight':a[ii,jj]}))
        for li,layer in enumerate(LAYERS[:3]):
            for gamma in c['community_resolutions']:
                candidates=[]
                for repeat in range(c['community_repeats']):
                    labs,quality=leiden(window[w,li],layer!='C',gamma,c['seed']+w*1000+li*100+repeat)
                    candidates.append((labs,quality))
                    records.extend([dict(period=int(p),layer=layer,gamma=gamma,repeat=repeat,country=k,
                        community=int(labs[k]),quality=quality) for k in range(len(countries))])
                kbest=max(range(len(candidates)),key=lambda k:np.nan_to_num(candidates[k][1],nan=-np.inf))
                labs=candidates[kbest][0]
                if np.isclose(gamma,1):best[(int(p),layer)]=labs
                scores=[partition_similarity(labs,x[0])[0] for x in candidates]
                stability.append(dict(period=int(p),layer=layer,gamma=gamma,best_repeat=kbest,
                    mean_ami_to_best=float(np.nanmean(scores)) if np.isfinite(scores).any() else np.nan,
                    n_assigned=int((labs>=0).sum()),n_communities=len(np.unique(labs[labs>=0]))))
        if not all((int(p),layer) in best for layer in LAYERS[:3]):raise ValueError('community_resolutions must include 1.0.')
        for a,b in [('C','F'),('C','H'),('F','H')]:
            ami,vi,n=partition_similarity(best[(int(p),a)],best[(int(p),b)])
            common=(best[(int(p),a)]>=0)&(best[(int(p),b)]>=0)
            strength=window[w,1].sum(0)+window[w,1].sum(1)
            similarity.append(dict(period=int(p),layers=a+'-'+b,AMI=ami,VI_normalized=vi,n_shared=n,
                F_exposure_retained=float(strength[common].sum()/strength.sum()) if strength.sum() else np.nan))
        print(f'Network partitions: {p}-{p+4}',flush=True)
    csv(pd.concat(edges,ignore_index=True),d['tables']/'network_edges.csv')
    csv(pd.DataFrame(records),d['tables']/'community_runs.csv');csv(pd.DataFrame(stability),d['tables']/'community_stability.csv')
    csv(pd.DataFrame(similarity),d['tables']/'community_similarity.csv')
    base=c['baseline_years'][0]
    if base not in periods:raise ValueError('baseline_years must select a complete observed five-year window.')
    baseidx=int(np.flatnonzero(periods==base)[0]);fixed=best[(base,'F')];nr=[]
    for w,p in enumerate(periods):
        for mode,labels in [('current_F',best[(int(p),'F')]),('fixed_F',fixed)]:
            for li,layer in [(1,'F'),(2,'H')]:
                po,pi,zo,zi,ret=node_roles(window[w,li],labels)
                for k in range(len(countries)):
                    nr.append(dict(period=int(p),partition=mode,layer=layer,country=k,
                        participation_out=po[k],participation_in=pi[k],within_z_out=zo[k],within_z_in=zi[k],
                        retained_neighbor_mass=ret[k],strength_in=window[w,li,:,k].sum(),
                        strength_out=window[w,li,k,:].sum(),community=int(labels[k])))
    csv(pd.DataFrame(nr),d['tables']/'network_node_roles.csv')
    layout=positions(window[-1,1].copy());csv(pd.DataFrame({'country':np.arange(len(countries)),'x':layout[:,0],'y':layout[:,1]}),d['tables']/'network_layout.csv')
    # Fixed exposure-defined country sets; report H alongside its observed F exposure.
    a=window[baseidx,1].copy();np.fill_diagonal(a,0);s=a.sum(0)+a.sum(1);active=np.flatnonzero(s>0);order=active[np.argsort(-s[active],kind='stable')]
    clubs=[]
    for fraction in [.05,.10,.20]:
        chosen=order[:max(1,int(np.ceil(len(order)*fraction)))];mask=np.zeros(len(s),bool);mask[chosen]=True
        for w,p in enumerate(periods):
            for li,layer in [(1,'F'),(2,'H')]:
                a=window[w,li].copy();np.fill_diagonal(a,0)
                clubs.append(dict(period=int(p),fraction=fraction,layer=layer,n_fixed=len(chosen),
                    share_inside=float(a[np.ix_(mask,mask)].sum()/a.sum()) if a.sum() else np.nan,
                    countries=';'.join(countries.iloc[chosen].country_code.astype(str))))
    csv(pd.DataFrame(clubs),d['tables']/'fixed_exposure_clubs.csv')
    # Community overlap and alluvial use the same common countries and F exposure weights.
    w=len(periods)-1;p=int(periods[-1]);labs={l:best[(p,l)] for l in LAYERS[:3]}
    common=np.logical_and.reduce([labs[l]>=0 for l in labs]);mass=window[w,1].sum(0)+window[w,1].sum(1)
    aligned=pd.DataFrame({'country':np.flatnonzero(common),'weight':mass[common]})
    for l in labs:aligned[l]=labs[l][common]
    csv(aligned,d['tables']/'alluvial_countries.csv')
    return window

def balanced_groups(gp,gr,reference_counts,threads):
    """Balance static Numba scheduling by reference work; preserve each year pool."""
    ny=len(gp)-1;nworkers=min(threads,ny)
    capacity=np.full(nworkers,ny//nworkers);capacity[:ny%nworkers]+=1
    load=np.zeros(nworkers);buckets=[[] for _ in range(nworkers)]
    work=np.array([reference_counts[gr[gp[y]:gp[y+1]]].sum(dtype=np.int64) for y in range(ny)])
    for y in np.argsort(-work,kind='stable'):
        available=[w for w in range(nworkers) if len(buckets[w])<capacity[w]]
        w=min(available,key=lambda w:load[w]);buckets[w].append(int(y));load[w]+=work[y]
    order=np.array([y for bucket in buckets for y in bucket],int)
    rows=np.concatenate([gr[gp[y]:gp[y+1]] for y in order])
    ptr=np.r_[0,np.cumsum(np.diff(gp)[order],dtype=np.int64)]
    return ptr,rows,order

def build(c):
    d=dirs(c);year,primary,cp,cc,cw,fp,fc,fw,rp,rn,ri=shared_arrays(c)
    gp=arr(c,'group_ptr');gr=arr(c,'group_rows');sigma=arr(c,'sigma');metrics=arr(c,'paper_metrics');kind=arr(c,'kind')
    countries=pd.read_csv(d['tables']/'countries.csv');fields=pd.read_csv(d['tables']/'fields.csv');nc=len(countries);nf=len(fields)
    print('Building reference-slot composition pools.',flush=True)
    scheduled_gp,scheduled_gr,order=balanced_groups(gp,gr,rn,c['threads'])
    inverse=np.argsort(order)
    pools=build_pools(scheduled_gp,scheduled_gr,nc,nf,year,kind,cp,cc,cw,fp,fc,fw,rp,rn,ri,sigma)
    np.savez_compressed(d['cache']/'reference_pools.npz',pools=pools[inverse],years=np.arange(c['trend_start'],c['end_year']+1))
    check_memory(c,'reference pools');model_q=mapped_array(c,'model_q',np.float32,(len(year),));model_q[:]=np.nan
    print('Aggregating C/F/H and leave-paper-out label-mixing references.',flush=True)
    net,role,cohort,nmodel=aggregate_networks(scheduled_gp,scheduled_gr,nc,nf,int(rn.max()),c['end_year'],c['min_null_pool'],pools,
        year,kind,cp,cc,cw,fp,fc,fw,rp,rn,ri,sigma,metrics,model_q)
    net=net[inverse];role=role[inverse];cohort=cohort[inverse];nmodel=nmodel[inverse]
    model_q.flush();del pools,scheduled_gr,scheduled_gp
    checks=[]
    for y in range(len(gp)-1):
        rows=gr[gp[y]:gp[y+1]];n=len(rows);q=float(np.sum(metrics[rows,3]));ni=int(np.sum(cp[rows+1]-cp[rows]>=2))
        expected={'F':n,'H':q,'C_upper':ni,'E_src':n,'E_asm':n,'B_src':q,'B_asm':q,'cohort_E':n,'cohort_B':q}
        actual={'F':net[y,1].sum(),'H':net[y,2].sum(),'C_upper':net[y,0].sum()/2,
            'E_src':role[y,0].sum(),'E_asm':role[y,2].sum(),'B_src':role[y,1].sum(),'B_asm':role[y,3].sum(),
            'cohort_E':cohort[y,:,0].sum(),'cohort_B':cohort[y,:,1].sum()}
        for k in expected:
            if not np.isclose(actual[k],expected[k],atol=1e-5,rtol=2e-6):raise AssertionError(f'Conservation: year={y}, {k}: {actual[k]} != {expected[k]}')
            checks.append(dict(year=y+c['trend_start'],quantity=k,expected=expected[k],actual=actual[k]))
        if np.any(net[y,2]>net[y,1]+1e-5):raise AssertionError('H exceeds F.')
        if not np.isclose(net[y,3].sum(),nmodel[y],atol=1e-5,rtol=2e-6):raise AssertionError('Model-subset F does not conserve papers.')
        rows0=rows[np.isfinite(model_q[rows])]
        if not np.isclose(net[y,4].sum(),np.sum(metrics[rows0,3]),atol=1e-5,rtol=2e-6):raise AssertionError('H_model and observed model papers disagree.')
        if not np.isclose(net[y,5].sum(),np.sum(model_q[rows0],dtype=np.float64),atol=1e-5,rtol=2e-6):raise AssertionError('H0 and reference model papers disagree.')
        if np.any(net[y,5]>net[y,3]+1e-5):raise AssertionError('H0 exceeds its same-paper exposure F_model.')
    csv(pd.DataFrame(checks),d['tables']/'conservation_checks.csv')
    np.savez_compressed(d['cache']/'annual_networks.npz',network=net,roles=role,cohorts=cohort)
    rows=[]
    for y in range(role.shape[0]):
        a=role[y].transpose(1,2,0).reshape(nc*nf,len(ROLE))
        z=pd.DataFrame(a,columns=ROLE);z['year']=y+c['trend_start'];z['country']=np.repeat(np.arange(nc),nf);z['field']=np.tile(np.arange(nf),nc)
        rows.append(z)
    roles=pd.concat(rows,ignore_index=True);csv(roles,d['tables']/'role_masses.csv')
    coh=[]
    for y in range(cohort.shape[0]):
        jj=np.flatnonzero(cohort[y,:,0]>0)
        coh.append(pd.DataFrame({'citing_year':y+c['trend_start'],'cited_year':jj,'E':cohort[y,jj,0],'B':cohort[y,jj,1]}))
    csv(pd.concat(coh,ignore_index=True),d['tables']/'reference_cohorts.csv')
    totals=pd.DataFrame({'year':np.arange(c['trend_start'],c['end_year']+1),'N':np.diff(gp),
        'N_model':nmodel,'F':net[:,1].sum((1,2)),'H':net[:,2].sum((1,2)),
        'H_model':net[:,4].sum((1,2)),'H0':net[:,5].sum((1,2)),'H_context':net[:,6].sum((1,2))})
    totals['model_coverage']=ratio(totals.N_model,totals.N)
    csv(totals,d['tables']/'network_annual.csv')
    analyse_graphs(c,net,countries)
    json_write({'label_reference':'Reference-slot country x Fields composition conditional on citing year, receiving Fields, source age band and source type; the entire focal paper is excluded.',
        'min_remaining_fractional_slots':c['min_null_pool'],'individual_source_degrees_preserved':False,
        'comparison':'H_model vs H0 on identical eligible papers; full H is separately reported.',
        'context_reference':'Slot-weighted empirical mean sigma within the same context strata; different reference object from H0.',
        'community_reference':'Directed strength configuration objective; not the same as either reference-composition model.'},d['status']/'reference_definitions.json')
    return dict(main_papers=int(np.diff(gp).sum()),model_papers=int(nmodel.sum()),countries=nc,fields=nf,conservation='passed')
