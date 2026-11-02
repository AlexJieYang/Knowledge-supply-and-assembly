"""Deterministic non-empirical fixtures for SI numerical and layout checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid


def create(root,ncountries=26,nfields=19):
    root=Path(root).resolve();results=root/'results'
    for k in ['tables','cache','data','r']:(results/k).mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(1179);years=np.arange(1990,2025);periods=np.arange(1990,2025,5)
    countries=['CN','US','IN','GB','DE','FR','JP','BR','CA','AU','KR','ES','IT','ID','MX','ZA','NG','EG','SE','NO','FI','AR','CL','TR','PL','NZ'][:ncountries]
    names=['Physics','Geology','Engineering','Philosophy','Art','Sociology','Business','Psychology','Economics','Political science','Chemistry','Materials science','Geography','Mathematics','Environmental science','Computer science','Medicine','Biology','History'][:nfields]
    nc=len(countries);nf=len(names)
    def write(name,z,folder='tables'):z.to_csv(results/folder/(name+'.csv'),index=False)
    write('countries',pd.DataFrame({'country':range(nc),'country_code':countries}))
    write('fields',pd.DataFrame({'field':range(nf),'field_id':np.arange(nf)+100,'field_label':names}))
    cells=[]
    for yi,year in enumerate(years):
        N=50000+yi*5000;weights=rng.lognormal(1-np.repeat(np.arange(nc),nf)*.07,.75);ea=N*weights/weights.sum()
        ra=np.clip(.21+.0033*yi+.025*rng.normal(size=nc*nf),.06,.7)
        es=ea*rng.lognormal(0,.35,len(ea));es*=N/es.sum();rs=np.clip(ra+.025*rng.normal(size=len(ea)),.06,.7);rs*=np.dot(ea,ra)/np.dot(es,rs)
        for j in range(nc*nf):
            c=j//nf;f=j%nf;r={'year':year,'country':c,'field':f,'E_src':es[j],'B_src':es[j]*rs[j],'E_asm':ea[j],'B_asm':ea[j]*ra[j]}
            for role,E,R in [('src',es[j],rs[j]),('asm',ea[j],ra[j])]:
                r['E_'+role+'_model']=E*.999;r['B_'+role+'_model']=E*.999*R;r['B0_'+role]=E*.999*min(.95,R+.14)
            cells.append(r)
    yearly=pd.DataFrame(cells);yearly['period']=yearly.year//5*5
    masses=[x for x in yearly if x.startswith(('E_','B_','B0_'))]
    w=yearly.groupby(['period','country','field'],as_index=False)[masses].sum()
    for z,group in [(w,'period'),(yearly,'year')]:
        total=z.groupby(group).E_asm.transform('sum');cc=z.groupby([group,'country']).E_asm.transform('sum');ff=z.groupby([group,'field']).E_asm.transform('sum')
        z['RCA']=z.E_asm*total/(cc*ff);world=z.groupby(group).B_asm.transform('sum')/total
        for role in ['src','asm']:z['R_'+role]=z['B_'+role]/z['E_'+role];z['LQ_'+role]=z['R_'+role]/world
    write('role_country_field_period',w);write('role_country_field_year',yearly.drop(columns='period'))
    country=w.groupby(['period','country'],as_index=False)[masses].sum()
    world=country.groupby('period').B_asm.transform('sum')/country.groupby('period').E_asm.transform('sum')
    for role in ['src','asm']:country['R_'+role]=country['B_'+role]/country['E_'+role];country['LQ_'+role]=country['R_'+role]/world
    write('role_country_period',country)
    ann=yearly.groupby('year',as_index=False)[['E_asm','B_asm']].sum();ann['N']=np.rint(ann.E_asm).astype(int);ann['joint_cross_mean']=ann.B_asm/ann.E_asm
    ann['geo_cross_mean']=np.linspace(.58,.77,len(ann));ann['field_cross_mean']=np.linspace(.35,.44,len(ann))
    ann['completion_lower_mean']=ann.joint_cross_mean*.7;ann['completion_upper_mean']=ann.joint_cross_mean*.7+.3
    ann['reference_coverage_mean']=np.linspace(.7,.9,len(ann));write('paper_annual',ann)
    pop=pd.DataFrame({'year':years,'one':ann.N*2,'country_known':ann.N*1.4,'field_known':ann.N*1.99,'eligible':ann.N});write('population_year',pop)
    defs=[]
    for variant,factor,delta in [('main',1,0),('at_least_5',.9,.003),('coverage_ge_08',.6,-.002),('primary_same_refs',.999,.001)]:
        z=pd.DataFrame({'year':years,'variant':variant,'count':ann.N*factor,'mean':ann.joint_cross_mean+delta});z['sum']=z['count']*z['mean'];defs.append(z)
    write('definition_sensitivity',pd.concat(defs,ignore_index=True))
    write('fixed_support_mean',pd.DataFrame({'year':years[10:],'mean':ann.joint_cross_mean.iloc[10:].to_numpy()-.004,'n_fixed_cells':nc*nf,'retained_baseline_mass':.93}))
    ends=[]
    for a,b in [(2010,2014),(2020,2023),(2020,2024)]:
        z=ann[ann.year.between(a,b)];ends.append(dict(start_year=a,end_year=b,N=int(z.N.sum()),mean_q=z.B_asm.sum()/z.E_asm.sum()))
    write('period_endpoint_comparison',pd.DataFrame(ends))
    write('network_annual',pd.DataFrame({'year':years,'N':ann.N,'N_model':ann.N*.999,'H':ann.B_asm,'H_model':ann.B_asm*.999,'H0':ann.N*.999*(ann.joint_cross_mean+.14),'H_context':ann.N*(ann.joint_cross_mean+.015)}))
    frows=[]
    for (year,f),z in yearly.groupby(['year','field']):
        mass=z.E_asm.sum();q=z.B_asm.sum()/mass;g=float(ann.loc[ann.year==year,'geo_cross_mean'].iloc[0]);d=float(ann.loc[ann.year==year,'field_cross_mean'].iloc[0])
        for metric,value in [('geo_cross',g),('field_cross',d),('joint_cross',q),('primary_joint_cross',q+.001),('coupling',q-g*d),('mean_ref_age_joint',12.)]:frows.append(dict(year=year,field=f,metric=metric,mass=mass,sum=mass*value,sum_sq=mass*(value*value+.01),mean=value))
    write('field_year_means',pd.DataFrame(frows))
    hist=[]
    for i,year in enumerate(years):
        for group in [0,1]:
            total=int(ann.N.iloc[i]*(.25 if group else .75));zero=int(total*.08)
            for metric,mu in [('geo_cross',.7),('field_cross',.36),('joint_cross',.27)]:
                x=(np.arange(100)+.5)/100;weights=np.exp(-((x-mu-.03*group)/.19)**2);counts=rng.multinomial(total-zero,weights/weights.sum())
                hist.extend(dict(year=year,international=group,metric=metric,bin=k,lo=k/100,hi=(k+1)/100,n_positive=int(counts[k]),n_zero=zero,n_total=total) for k in range(100))
    write('histograms',pd.DataFrame(hist))
    conc=[]
    for scope,table,keys in [('country_field',w,['country','field']),('country',country,['country'])]:
        for period,z in table.groupby('period'):
            for role in ['src','asm']:
                for quantity in ['E','B']:
                    v=np.sort(z[quantity+'_'+role].to_numpy());shares=np.r_[0,np.cumsum(v)/v.sum()];gini=1-2*trapezoid(shares,dx=1/len(v))
                    conc.append(dict(scope=scope,period=period,role=role,quantity=quantity,n_units=len(v),Gini=gini,top5_share=v[-5:].sum()/v.sum(),top10pct_share=v[-max(1,int(np.ceil(.1*len(v)))):].sum()/v.sum()))
    write('role_concentration',pd.DataFrame(conc))
    runs=[];stability=[];net=np.zeros((len(periods),7,nc,nc))
    for wi,period in enumerate(periods):
        t=country[country.period==period].sort_values('country');N=t.E_asm.sum();q=t.B_asm.sum()/N
        F=np.outer(t.E_src,t.E_asm)/N;H=F*q
        C=(F+F.T)/2;np.fill_diagonal(C,0);C*=N*.5/C.sum()
        net[wi]=[C,F,H,F*.999,H*.999,F*.999*(q+.14),F*(q+.015)]
        for layer in ['C','F','H']:
            for gamma in [.5,1.,1.5,2.]:
                k=max(2,int(round((4 if layer=='C' else 3)*gamma)));labs=np.arange(nc)%k
                if layer=='H':labs[-2]=0
                for repeat in [0,1,2]:
                    runs.extend(dict(period=period,layer=layer,gamma=gamma,repeat=repeat,country=c,community=int(labs[c]),quality=1-repeat*.01) for c in range(nc))
                stability.append(dict(period=period,layer=layer,gamma=gamma,best_repeat=0,mean_ami_to_best=1.,n_assigned=nc,n_communities=len(np.unique(labs))))
    np.savez_compressed(results/'cache/window_networks.npz',network=net,periods=periods)
    write('community_runs',pd.DataFrame(runs));write('community_stability',pd.DataFrame(stability))
    coh=[]
    for year in years:
        for cited in np.arange(1930,year+1,5):
            E=float(rng.uniform(100,900));coh.append(dict(citing_year=year,cited_year=cited,E=E,B=E*(.24+.0006*(year-cited))))
    write('reference_cohorts',pd.DataFrame(coh))
    teams=['1','2','3-4','5-8','9-16','17-32','33-64','65+'];refs=['2','3-4','5-9','10-19','20-39','40-79','80-159','160+']
    write('response_surface',pd.DataFrame([dict(team_bin=i,reference_bin=j,team_label=teams[i],reference_label=refs[j],n=50 if (i,j)==(7,0) else 200+i*30+j*80,mean=.22+.1*np.exp(-((j-4)/2)**2)-.006*i) for i in range(8) for j in range(8)]))
    n=5000;sample=pd.DataFrame({'paper_id':np.arange(1,n+1,dtype=np.uint64),'year':rng.choice(years,n),'primary_field':rng.integers(0,nf,n),'known_fields':np.ones(n,int),'known_countries':1+rng.binomial(1,.25,n),'joint_refs':2+rng.poisson(25,n),'team_size':1+rng.poisson(5,n)})
    sample['core_refs']=sample.joint_refs+rng.poisson(5,n);sample['reference_coverage']=sample.joint_refs/sample.core_refs
    sample['geo_cross']=rng.beta(6,2,n);sample['field_cross']=rng.beta(2,4,n);sample['joint_cross']=sample.geo_cross*sample.field_cross
    sample['semantic_coverage']=rng.uniform(.6,1,n);sample['semantic_existing']=np.clip(.09+.08*sample.joint_cross+rng.normal(0,.016,n),.01,.4)
    sample.to_csv(results/'data/plot_sample.csv',index=False)
    joint=sample.iloc[:600]
    write('joint_semantic',pd.DataFrame({'paper_id':joint.paper_id,'year':joint.year,'n_joint_embedding_refs':joint.joint_refs,'joint_ref_coverage':.9,'q_same_refs':joint.joint_cross,'cosine_pair_distance':joint.semantic_existing}))
    models=['paper_joint_cross','paper_geo_cross','paper_field_cross','paper_AI_profile','country_field_assembly'];coef=[];scope=[]
    for model in models:
        terms=['log1p(RCA)','log1p(E_asm)'] if model=='country_field_assembly' else ['international','log_team','log_refs','mean_ref_age_joint']
        if model=='paper_AI_profile':terms=['ai_high']+terms
        for term in terms:
            estimate=float(rng.normal(.01,.008));se=.001
            coef.append(dict(model=model,term=term,estimate=estimate,std_error=se,statistic=estimate/se,p_value=.01,conf_low=estimate-1.96*se,conf_high=estimate+1.96*se,n=50000))
        scope.append(dict(model=model,unit='country x Fields x year cell' if model=='country_field_assembly' else 'one citing paper',n_export=50100,n_complete=50050,n_model=50000,weights='synthetic example',fixed_effects='synthetic example',covariance='synthetic example',interpretation='SYNTHETIC ONLY'))
    write('coefficients',pd.DataFrame(coef),'r');write('model_scope',pd.DataFrame(scope),'r')
    config={'output_dir':'results','trend_start':1990,'end_year':2024,'baseline_years':[2010,2014],'comparison_years':[2020,2024],'min_cell_mass':25,'min_surface_n':100,'synthetic':True}
    root.mkdir(exist_ok=True);(root/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    return root/'config.json',results/'data/plot_sample.csv'
