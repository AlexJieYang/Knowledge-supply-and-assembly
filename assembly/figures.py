"""Five data-driven publication layouts. All panels expose companion source CSVs.
Synthetic outputs are watermarked and are never packaged as empirical results.
"""
import json
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Polygon,PathPatch,FancyArrowPatch
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D
from matplotlib.text import Text
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from .runtime import dirs,csv,json_write,ROOT

TEAL='#008B83';ORANGE='#D18436';NAVY='#254B69';RED='#AD5362';GRAY='#73808A'
COLORS=[TEAL,ORANGE,NAVY,RED,'#7D71A7','#6C985E','#A49370','#67A6BF','#B8A7C5','#888888']
plt.rcParams.update({'font.family':'sans-serif',
    'font.sans-serif':['Arial','Helvetica','Liberation Sans','DejaVu Sans'],
    'font.size':7,'axes.titlesize':7,
    'axes.labelsize':6.8,'xtick.labelsize':5.8,'ytick.labelsize':5.8,'legend.fontsize':5.8,
    'axes.linewidth':.5,'lines.linewidth':1.1,'pdf.fonttype':42,'ps.fonttype':42,
    'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white',
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.labelpad':2,'xtick.major.pad':2,'ytick.major.pad':2,
    'figure.constrained_layout.h_pad':.025,
    'figure.constrained_layout.w_pad':.025,'figure.constrained_layout.hspace':.035,
    'figure.constrained_layout.wspace':.045})

class Data:
    def __init__(self,c):
        self.c=c;self.d=dirs(c);self.fields=self.read('fields');self.countries=self.read('countries')
        self.country=dict(zip(self.countries.country,self.countries.country_code))
        self.fields['field_label']=[c['field_labels'].get(str(int(row.field_id)),row.field_label) for row in self.fields.itertuples()]
        self.field=dict(zip(self.fields.field,self.fields.field_label))
        csv(self.fields,self.d['tables']/'display_fields.csv')
        self.source={}
    def read(self,name):return pd.read_csv(self.d['tables']/(name+'.csv'))
    def use(self,figure,panel,table,scope=''):
        self.source.setdefault(figure,[]).append(dict(panel=panel,table=table,scope=scope))
    def new(self,title,height=165):
        fig=plt.figure(figsize=(180/25.4,height/25.4),layout='constrained')
        # Titles and explanatory text belong to the separate manuscript legend.
        fig._assembly_title=title
        fig._assembly_panel_titles={}
        fig._assembly_layout_followers=[]
        fig._assembly_country_labels=[]
        return fig
    def save(self,fig,name,caption):
        if self.c.get('synthetic',False):
            fig.text(.5,.5,'SYNTHETIC / FUNCTIONAL TEST',ha='center',va='center',fontsize=23,
                     rotation=30,color='#B04C4C',alpha=.10,zorder=1000)
        fig.canvas.draw()
        # 3-D projected labels can collapse constrained-layout's size solver.
        # Reserve their cell with a normal axes, then place the 3-D view in it.
        for ax,slot in fig._assembly_layout_followers:
            ax.set_position(slot.get_position())
        # Freeze one resolved layout for identical PDF/SVG/PNG panel positions.
        fig.set_layout_engine('none')
        place_country_labels(fig)
        fig.canvas.draw()
        for ext in ('pdf','svg','png'):
            fig.savefig(self.d['figures']/(name+'.'+ext),dpi=450)
        panels=' '.join(f'{letter}, {text}.' for letter,text in
                        sorted(fig._assembly_panel_titles.items()))
        legend_text=f'{fig._assembly_title}.\n\n{panels}\n\n{caption}\n'
        (self.d['figures']/(name+'_caption.txt')).write_text(legend_text,encoding='utf-8')
        csv(pd.DataFrame(self.source.get(name,[])),self.d['figures']/(name+'_sources.csv'))
        # Individual panels remain editable vector objects in PDF/SVG.
        plt.close(fig)

def title(ax,letter,text):
    ax.figure._assembly_panel_titles[letter]=text
    ax.set_title(letter,loc='left',pad=2,fontsize=7,fontweight='bold')

def row_axes(fig,spec,ncols,width_ratios=None):
    grid=spec.subgridspec(1,ncols,width_ratios=width_ratios)
    return [fig.add_subplot(grid[0,k]) for k in range(ncols)]

def country_label(ax,x,y,label,fontsize=5.8):
    ann=ax.annotate(label,(x,y),xytext=(3,3),textcoords='offset points',
                    fontsize=fontsize,arrowprops=dict(arrowstyle='-',lw=.35,color=GRAY))
    ax.figure._assembly_country_labels.append(ann)

def place_country_labels(fig):
    """Place a few country names without moving any data points or axis limits."""
    renderer=fig.canvas.get_renderer();occupied={}
    candidates=[(3,3),(3,-10),(-3,3),(-3,-10),(3,12),(3,-18),(-3,12),(-3,-18),(12,3),(-12,3)]
    for ann in fig._assembly_country_labels:
        ax=ann.axes;others=occupied.setdefault(ax,[]);best=None
        for dx,dy in candidates:
            ann.set_position((dx,dy));ann.set_ha('left' if dx>0 else 'right')
            ann.update_positions(renderer);box=Text.get_window_extent(ann,renderer).padded(1)
            overlap=sum(max(0,min(box.x1,b.x1)-max(box.x0,b.x0))*
                        max(0,min(box.y1,b.y1)-max(box.y0,b.y0)) for b in others)
            overflow=(max(0,ax.bbox.x0-box.x0)+max(0,box.x1-ax.bbox.x1)+
                      max(0,ax.bbox.y0-box.y0)+max(0,box.y1-ax.bbox.y1))
            cost=overlap+100*overflow
            if best is None or cost<best[0]:best=(cost,dx,dy,box)
            if cost==0:break
        _,dx,dy,box=best;ann.set_position((dx,dy));ann.set_ha('left' if dx>0 else 'right')
        ann.update_positions(renderer);others.append(box)

def empty(ax,message):
    ax.text(.5,.5,'\n'.join(textwrap.wrap(message,30)),ha='center',va='center',transform=ax.transAxes,color=GRAY,fontsize=6.8)
    ax.set_xticks([]);ax.set_yticks([])

def legend(ax,**kwargs):
    ax.legend(frameon=False,handlelength=1.5,**kwargs)

def role_maps(ax,s,db,norm):
    geo=json.loads((ROOT/'assets/world.geojson').read_text(encoding='utf-8'))
    cmap=plt.get_cmap('BrBG')
    for feature in geo['features']:
        props=feature['properties'];code=props.get('iso_a2',props.get('ISO_A2',''))
        value=s.get(code,np.nan);face=cmap(norm(value)) if np.isfinite(value) else '#E4E7E9'
        geom=feature['geometry']
        if geom is None:continue
        poly=geom['coordinates']
        if geom['type']=='Polygon':poly=[poly]
        if geom['type'] not in ('Polygon','MultiPolygon'):continue
        for item in poly:
            if not item:continue
            xy=np.asarray(item[0]);ax.add_patch(Polygon(xy,closed=True,facecolor=face,edgecolor='white',linewidth=.16))
    ax.set_xlim(-180,180);ax.set_ylim(-60,85);ax.set_aspect('equal');ax.set_axis_off()
    return plt.cm.ScalarMappable(norm=norm,cmap=cmap)


def figure1(db):
    name='Fig1_Knowledge_Assembly';fig=db.new('The geography of knowledge assembly',154)
    grid=fig.add_gridspec(3,1,height_ratios=[1,.63,1.02])
    axes=dict(zip('ABCDEF',[ax for r in range(3) for ax in row_axes(fig,grid[r],2)]))
    ax=axes['A'];title(ax,'a','From reference context to credit')
    for y,text,sig in [(.82,'A / Field 1','1/2'),(.50,'A / Field 1','1/2'),(.18,'B / Field 2','1')]:
        ax.scatter([.16],[y],s=110,color=TEAL if text.startswith('A') else ORANGE,zorder=3)
        ax.text(.16,y+.09,text,ha='center',fontsize=6)
        ax.add_patch(FancyArrowPatch((.22,y),(.78,.5),arrowstyle='-|>',mutation_scale=8,linewidth=.8,color=GRAY))
        ax.text(.50,(y+.5)/2+.04,r'$\sigma='+sig+'$',ha='center',fontsize=6.5)
    ax.scatter([.84],[.5],s=175,color=NAVY,zorder=4);ax.text(.84,.68,'Focal paper\nB / Field 2',ha='center',fontsize=6)
    ax.text(.48,.01,r'$q=2/3;\quad \sum_j b_{ij}=q_i$',ha='center',fontsize=7);ax.set_xlim(0,1);ax.set_ylim(-.06,1);ax.axis('off')
    db.use(name,'a','Definition only','Illustrative three-reference example; not an estimate')
    p=db.read('population_year');p=p[p.year.between(db.c['trend_start'],2024)]
    ax=axes['B'];title(ax,'b','Observed population and joint support')
    for col,label,color in [('one','All core',NAVY),('country_known','Country known',GRAY),('eligible','P*',TEAL)]:
        ax.plot(p.year,p[col]/1e6,label=label,color=color)
    ax.set(xlabel='Publication year',ylabel='Papers (million)');legend(ax,loc='upper left');db.use(name,'b','population_year','Core population; P* requires >=2 jointly known references')
    z=db.read('role_country_period');z=z[z.period.eq(db.c['comparison_years'][0])]
    maxv=max(1.5,float(np.nanquantile(np.r_[z.LQ_src,z.LQ_asm],.95)))
    norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=maxv)
    for panel,role,label in [('C','src','Source role'),('D','asm','Assembler role')]:
        ax=axes[panel];title(ax,panel.lower(),label+' | '+str(db.c['comparison_years'][0])+'-2024')
        s=dict(zip(z.country.map(db.country),z['LQ_'+role]));sm=role_maps(ax,s,db,norm)
        db.use(name,panel.lower(),'role_country_period','Same P*, same color scale; institutional country credits')
    cb=fig.colorbar(sm,ax=[axes['C'],axes['D']],orientation='horizontal',
                    fraction=.12,pad=.015,shrink=.65,aspect=45,extend='max')
    cb.set_label('Bridge credit / exposure, relative to world',fontsize=5.8)
    ax=axes['E'];title(ax,'e','Source and assembler roles')
    size=12+65*np.sqrt(z.E_asm/max(1,z.E_asm.max()))
    ax.scatter(z.LQ_src,z.LQ_asm,s=size,color=TEAL,alpha=.7,edgecolors='white',linewidths=.3)
    for row in z.nlargest(7,'E_asm').itertuples():country_label(ax,row.LQ_src,row.LQ_asm,db.country[row.country])
    ax.axhline(1,color=GRAY,linewidth=.5,linestyle=':');ax.axvline(1,color=GRAY,linewidth=.5,linestyle=':')
    ax.set(xlabel='Source location quotient',ylabel='Assembler location quotient');db.use(name,'e','role_country_period','Point size: assembler exposure')
    a=db.read('paper_annual');ax=axes['F'];title(ax,'f','Annual boundary structure')
    for col,label,color in [('geo_cross_mean','Geography',NAVY),('field_cross_mean','Fields',ORANGE),('joint_cross_mean','Joint',TEAL)]:
        ax.plot(a.year,a[col],label=label,color=color)
    ax.set(xlabel='Citing year',ylabel='Mean cross-boundary co-use',ylim=(0,1));legend(ax,loc='upper left');db.use(name,'f','paper_annual','Paper-weighted P* means')
    db.save(fig,name,'P*: 1990-2024 core papers with focal country/Fields and >=2 jointly known core references. Country is institutional location. F is paper-normalised citation exposure; H credits the joint geographic/Fields reference context. Zero or missing exposure is not a role score. Maps share one scale; upper colors may be clipped for display.')


def align_labels(reference,other,weights=None):
    ref=np.asarray(reference);other=np.asarray(other);good=(ref>=0)&(other>=0)
    rs=np.unique(ref[good]);os=np.unique(other[good]);tab=np.zeros((len(rs),len(os)))
    weights=np.ones(len(ref)) if weights is None else np.asarray(weights)
    for i,r in enumerate(rs):
        for j,o in enumerate(os):tab[i,j]=weights[good&(ref==r)&(other==o)].sum()
    mapping={}
    if tab.size:
        ii,jj=linear_sum_assignment(-tab)
        mapping.update({int(os[j]):int(rs[i]) for i,j in zip(ii,jj)})
    nxt=int(rs.max()+1) if len(rs) else 0
    for o in os:
        if int(o) not in mapping:mapping[int(o)]=nxt;nxt+=1
    return np.array([mapping.get(int(x),-1) for x in other])


def network_view(ax,W,labels,xy,country,layer):
    W=W.copy();np.fill_diagonal(W,0);ii,jj=np.nonzero(W>0)
    if layer=='C':keep=ii<jj;ii=ii[keep];jj=jj[keep]
    weight=W[ii,jj];chosen=np.argsort(-weight)[:60];shown=weight[chosen].sum()/weight.sum() if weight.sum() else 0
    maxw=weight.max() if len(weight) else 1
    for k in chosen:
        i,j=ii[k],jj[k];lw=.2+1.1*np.sqrt(weight[k]/maxw)
        if layer=='C':ax.plot(xy[[i,j],0],xy[[i,j],1],color='#B8C4CD',linewidth=lw,alpha=.6,zorder=1)
        else:ax.add_patch(FancyArrowPatch(xy[i],xy[j],arrowstyle='-|>',mutation_scale=3.2,linewidth=lw,color='#A7B6C3',alpha=.55,zorder=1))
    strength=W.sum(0)+W.sum(1);size=3+38*np.sqrt(strength/max(1,strength.max()))
    colors=[COLORS[int(x)%len(COLORS)] if x>=0 else '#D8DEE2' for x in labels]
    ax.scatter(xy[:,0],xy[:,1],s=size,c=colors,linewidths=.25,edgecolors='white',zorder=2)
    for i in np.argsort(-strength)[:4]:ax.text(xy[i,0]+.025,xy[i,1]+.025,country[i],fontsize=5.5)
    ax.set_axis_off()
    ax.text(.5,1.01,layer,transform=ax.transAxes,ha='center',va='bottom',fontsize=7)
    ax.margins(.055)
    return shown


def alluvial(ax,t):
    layers=['C','F','H'];data=t.copy();data['C']=align_labels(data.F,data.C,data.weight);data['H']=align_labels(data.F,data.H,data.weight)
    total=data.weight.sum();levels={};nxt={}
    if total<=0:empty(ax,'No common assigned countries');return
    data['weight']/=total
    for x,layer in enumerate(layers):
        masses=data.groupby(layer).weight.sum().sort_index();offset=0;levels[layer]={}
        gap=.018
        for lab,mass in masses.items():
            levels[layer][lab]=offset;ax.bar(x,mass,width=.045,bottom=offset,color=COLORS[int(lab)%len(COLORS)],linewidth=0)
            if mass>.04:ax.text(x,offset+mass/2,str(int(lab)+1),ha='center',va='center',color='white',fontsize=5.5)
            offset+=mass+gap
        nxt[layer]=offset
        ax.text(x,-.075,layer,ha='center',fontsize=7,fontweight='bold')
    for x,(a,b) in enumerate(zip(layers[:-1],layers[1:])):
        aa=levels[a].copy();bb=levels[b].copy()
        for (left,right),mass in data.groupby([a,b]).weight.sum().items():
            y0=aa[left];y1=bb[right];xx0=x+.025;xx1=x+.975
            verts=[(xx0,y0),(xx0+.38,y0),(xx1-.38,y1),(xx1,y1),(xx1,y1+mass),
                (xx1-.38,y1+mass),(xx0+.38,y0+mass),(xx0,y0+mass),(xx0,y0)]
            codes=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MPath(verts,codes),facecolor=COLORS[int(left)%len(COLORS)],edgecolor='none',alpha=.35))
            aa[left]+=mass;bb[right]+=mass
    ax.set_xlim(-.18,2.18);ax.set_ylim(-.1,max(nxt.values())+.03);ax.axis('off')


def figure2(db):
    name='Fig2_Multilayer_Boundaries';fig=db.new('Collaboration, citation exposure and contextual assembly',168)
    grid=fig.add_gridspec(4,1,height_ratios=[1,1.02,.65,1.08])
    p=db.c['comparison_years'][0];data=np.load(db.d['cache']/'window_networks.npz');wi=int(np.flatnonzero(data['periods']==p)[0]);net=data['network'][wi]
    runs=db.read('community_runs');stab=db.read('community_stability');labs={}
    for layer in ['C','F','H']:
        r=int(stab[(stab.period==p)&(stab.layer==layer)&np.isclose(stab.gamma,1)].best_repeat.iloc[0])
        labs[layer]=runs[(runs.period==p)&(runs.layer==layer)&np.isclose(runs.gamma,1)&(runs.repeat==r)].sort_values('country').community.to_numpy(int)
    labs['C']=align_labels(labs['F'],labs['C']);labs['H']=align_labels(labs['F'],labs['H'])
    xy=db.read('network_layout').sort_values('country')[['x','y']].to_numpy();country=db.countries.sort_values('country').country_code.to_numpy()
    retained=[]
    for k,(ax,layer) in enumerate(zip(row_axes(fig,grid[0],3),['C','F','H'])):
        retained.append(f'{layer}: {network_view(ax,net[k],labs[layer],xy,country,layer):.0%}')
        if k==0:title(ax,'a','Aligned international C/F/H views')
    shown='; '.join(retained)
    db.use(name,'a','network_edges; network_layout; community_runs','One common F-based spectral layout; top 60 display edges retain '+shown+'; all positive edges analysed')
    common=np.logical_and.reduce([labs[x]>=0 for x in labs]);weight=net[1].sum(0)+net[1].sum(1)
    for k,(ax,(a,b)) in enumerate(zip(row_axes(fig,grid[1],3),[('C','F'),('C','H'),('F','H')])):
        table=pd.crosstab(labs[a][common],labs[b][common],values=weight[common],aggfunc='sum').fillna(0)
        if table.size==0:
            empty(ax,'No shared positive-strength assignments');continue
        norm=table.to_numpy()/table.to_numpy().sum() if table.to_numpy().sum() else table.to_numpy()
        image=ax.imshow(norm,cmap='Blues',aspect='auto',vmin=0)
        ax.set(xlabel=b+' community',ylabel=a+' community');ax.set_xticks(np.arange(len(table.columns)),[str(i+1) for i in table.columns]);ax.set_yticks(np.arange(len(table.index)),[str(i+1) for i in table.index])
        if k==0:title(ax,'b','Exposure-weighted community correspondence: C/F, C/H and F/H')
        cb=fig.colorbar(image,ax=ax,fraction=.04,pad=.03);cb.ax.tick_params(labelsize=5.5)
    db.use(name,'b','alluvial_countries','Same common countries and F-exposure weights in each cross-layer comparison')
    ax=fig.add_subplot(grid[2]);title(ax,'c','Cross-layer affiliations of the same countries');alluvial(ax,db.read('alluvial_countries'))
    db.use(name,'c','alluvial_countries','Membership correspondence across layers, not migration or temporal flows')
    bottom=row_axes(fig,grid[3],3)
    ax=bottom[0];title(ax,'d','Cross-layer agreement');sim=db.read('community_similarity')
    for k,(group,t) in enumerate(sim.groupby('layers')):ax.plot(t.period+2,t.AMI,'o-',markersize=2.5,color=COLORS[k],label=group)
    ax.set(xlabel='Window midpoint',ylabel='Adjusted mutual information');legend(ax,loc='best');db.use(name,'d','community_similarity','Unweighted AMI of common assigned countries; retained exposure in source table')
    ax=bottom[1];title(ax,'e','Boundary-spanning positions');roles=db.read('network_node_roles');r=roles[(roles.period==p)&(roles.partition=='current_F')]
    for k,layer in enumerate(['F','H']):
        v=np.sort(r.loc[r.layer==layer,'participation_out'].dropna());ax.plot(v,np.arange(1,len(v)+1)/max(1,len(v)),label=layer,color=[NAVY,TEAL][k])
    ax.set(xlabel='Outgoing participation',ylabel='Country empirical CDF');legend(ax);db.use(name,'e','network_node_roles','F partition held common for F/H roles; zero-strength nodes undefined')
    ax=bottom[2];title(ax,'f','Optimisation stability')
    for k,layer in enumerate(['C','F','H']):
        r=stab[(stab.period==p)&(stab.layer==layer)];ax.plot(r.gamma,r.mean_ami_to_best,'o-',markersize=2.5,label=layer,color=COLORS[k])
    ax.set(xlabel='Resolution parameter',ylabel='Mean AMI to best run',ylim=(-.02,1.03));legend(ax);db.use(name,'f','community_stability','Repeated optimisation stability, not statistical confidence')
    db.save(fig,name,'International non-diagonal C/F/H on identical P*. C is undirected; F/H run directed weighted Leiden with the configuration objective. Community colors align to F for display. Each network displays its top 60 edges; retained edge mass is '+shown+'. Alluvial widths use one F-exposure weight per common country across all layers; they are affiliations, not actual movement. Domestic mass is retained in total-credit analyses.')


def raincloud(ax,s,metric):
    rng=np.random.default_rng(24)
    for group,y,color,label in [(0,0,NAVY,'One known country'),(1,1,TEAL,'2+ known countries')]:
        v=s.loc[s.international.eq(group),metric].dropna().to_numpy()
        if len(v)==0:continue
        bins=np.linspace(0,1,80);hist,_=np.histogram(v,bins=bins,density=True);centers=(bins[:-1]+bins[1:])/2
        density=gaussian_filter1d(hist,1.2);height=.31*density/max(density.max(),1e-12)
        ax.fill_between(centers,y,y+height,color=color,alpha=.35,linewidth=.4)
        scatter=v[rng.choice(len(v),min(450,len(v)),replace=False)]
        ax.scatter(scatter,y-rng.uniform(.05,.23,len(scatter)),s=1.1,color=color,alpha=.18,rasterized=True)
        lo,med,hi=np.quantile(v,[.25,.5,.75]);ax.plot([lo,hi],[y-.035,y-.035],color=color,linewidth=3)
        ax.scatter([med],[y-.035],s=9,facecolors='white',edgecolors=color,zorder=5,linewidths=.6)
    ax.set_yticks([0,1],['One known country','2+ known countries']);ax.set(xlim=(0,1),ylim=(-.3,1.42),xlabel='Co-use intensity')


def figure3(db):
    name='Fig3_Boundary_Distributions';fig=db.new('How reference contexts cross geographic and disciplinary boundaries',170)
    grid=fig.add_gridspec(3,1,height_ratios=[.67,1.15,1.02])
    s=pd.read_parquet(db.d['data']/'plot_sample.parquet')
    for k,(ax,(metric,heading)) in enumerate(zip(row_axes(fig,grid[0],2),[('geo_cross','Geographic boundary'),('joint_cross','Joint geographic / Fields boundary')])):
        title(ax,'ab'[k],heading);raincloud(ax,s,metric)
        ax.set_xlabel('Geographic co-use' if k==0 else 'Joint geographic / Fields co-use')
        if k==1:ax.tick_params(axis='y',labelleft=False)
        db.use(name,'ab'[k],'plot_sample.parquet','Uniform paper sample: density, scatter, median and IQR; no oversampling')
    middle=row_axes(fig,grid[1],3,width_ratios=[1,1,1.12])
    ax=middle[0];title(ax,'c','Changing distribution of joint co-use')
    h=db.read('histograms');h=h[h.metric.eq('joint_cross')];h['period']=h.year//5*5
    yy=h.groupby(['period','bin','lo','hi'],as_index=False)[['n_positive','n_zero','n_total']].sum()
    for offset,(period,t) in enumerate(yy.groupby('period')):
        t=t.sort_values('bin');x=(t.lo+t.hi)/2;v=t.n_positive.to_numpy(float);v=gaussian_filter1d(v,1)
        height=.8*v/max(1,v.max());ax.fill_between(x,offset,offset+height,color=plt.get_cmap('viridis')(.1+.8*offset/max(1,yy.period.nunique()-1)),alpha=.75)
        zero=t.n_zero.iloc[0]/max(1,t.n_total.iloc[0]);ax.scatter([-.065],[offset+.15],s=4+32*zero,color=GRAY)
    periods=sorted(yy.period.unique());ax.set_yticks(np.arange(len(periods)),[f'{p}-{str(p+4)[-2:]}' for p in periods]);ax.set(xlim=(-.11,1.02),xlabel='Joint co-use');db.use(name,'c','histograms','All P*: positive distributions scaled to ridge height; left dots encode zero mass separately')
    ax=middle[1];title(ax,'d','Four reference-pair contexts');a=db.read('paper_annual')
    joint=a.joint_cross_mean.to_numpy();geo=a.geo_cross_mean.to_numpy();field=a.field_cross_mean.to_numpy()
    values=np.maximum(np.vstack([1-geo-field+joint,geo-joint,field-joint,joint]),0)
    ax.stackplot(a.year,values,colors=['#DDE3E7',NAVY,ORANGE,TEAL],labels=['Same / same','Geography only','Fields only','Both'],linewidth=0)
    ax.set(xlabel='Citing year',ylabel='Fractional pair mass',ylim=(0,1));legend(ax,loc='upper left',fontsize=5.5);db.use(name,'d','paper_annual','Average within-paper pair-type fractions; distinct reference papers only')
    ax=middle[2];title(ax,'e','Boundary coupling by Fields')
    f=db.read('field_year_means');p=f.pivot(index=['field','year'],columns='metric',values='mean').reset_index()
    table=p.pivot(index='field',columns='year',values='coupling');v=table.to_numpy();limit=max(.02,float(np.nanquantile(np.abs(v),.95)))
    im=ax.imshow(v,aspect='auto',cmap='PuOr_r',vmin=-limit,vmax=limit)
    ax.set_yticks(np.arange(len(table)),[short(db.field[i]) for i in table.index],fontsize=5.5)
    cols=list(table.columns);ticks=np.linspace(0,len(cols)-1,4,dtype=int);ax.set_xticks(ticks,[cols[i] for i in ticks],rotation=45);ax.set_xlabel('Citing year')
    cb=fig.colorbar(im,ax=ax,fraction=.04,pad=.02,extend='both');cb.set_label('q - geography x Fields',fontsize=5.5);db.use(name,'e','field_year_means','Weighted mean of within-paper q - geographic co-use x Fields co-use; descriptive coupling')
    bottom=grid[2].subgridspec(1,2,width_ratios=[1,1.05])
    surf=db.read('response_surface');means=surf.pivot(index='team_bin',columns='reference_bin',values='mean').to_numpy();n=surf.pivot(index='team_bin',columns='reference_bin',values='n').to_numpy()
    means=np.where(n>=db.c['min_surface_n'],means,np.nan);ax=fig.add_subplot(bottom[0]);title(ax,'f','Observed response surface')
    im=ax.imshow(means,origin='lower',aspect='auto',cmap='viridis',vmin=0,vmax=1)
    for i in range(8):
        for j in range(8):
            if np.isfinite(means[i,j]):ax.text(j,i,f'{means[i,j]:.2f}',ha='center',va='center',fontsize=5.5,color='white' if means[i,j]<.5 else '#17323E')
    team=surf.drop_duplicates('team_bin').sort_values('team_bin').team_label;ref=surf.drop_duplicates('reference_bin').sort_values('reference_bin').reference_label
    ax.set_xticks(np.arange(8),ref,rotation=50,ha='right');ax.set_yticks(np.arange(8),team);ax.set(xlabel='Joint core references (bins)',ylabel='Authors (bins)');fig.colorbar(im,ax=ax,fraction=.035,pad=.025,label='Mean q')
    db.use(name,'f','response_surface',f'2020-2024 full P* bin means; n<{db.c["min_surface_n"]} masked; bin counts in source')
    slot=fig.add_subplot(bottom[1]);slot.set_axis_off();title(slot,'g','Same binned data in three dimensions')
    ax=fig.add_axes(slot.get_position(),projection='3d');ax.set_in_layout(False)
    fig._assembly_layout_followers.append((ax,slot))
    xx,yy=np.meshgrid(np.arange(8),np.arange(8))
    if np.isfinite(means).any():ax.plot_surface(xx,yy,np.ma.masked_invalid(means),cmap='viridis',vmin=0,vmax=1,linewidth=.25,edgecolor='white',antialiased=True)
    ax.view_init(elev=28,azim=-128);ax.set_xticks([0,2,4,6],[ref.iloc[x] for x in [0,2,4,6]]);ax.set_yticks([0,2,4,6],[team.iloc[x] for x in [0,2,4,6]])
    ax.set_box_aspect((1.2,1,.8),zoom=1.04)
    ax.set(xlabel='Reference bin',ylabel='Author bin',zlabel='Mean q',zlim=(0,1));ax.tick_params(pad=0,labelsize=5.5)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.label.set_size(5.8)
    ax.grid(False)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.pane.fill=False
    db.use(name,'g','response_surface','Ordinal bin axes; adjacent observed means are connected for display; not a fitted or causal response')
    db.save(fig,name,'All measures use identical jointly country/Fields-known references. Group labels count observed institutional countries. Raincloud marks come from a uniform deterministic sample; ridges, context fractions, field means and surfaces use full data. Ridge height is normalised within period; zero mass is separate. Panels f/g show observed bin means on ordinal axes; g connects adjacent means for display. No response model is fitted; sparse bins are masked, with counts in source data.')


def short(label):
    return {'Computer science':'Computer sci.','Environmental science':'Environmental sci.','Materials science':'Materials sci.','Political science':'Political sci.'}.get(label,label)


def figure4(db):
    name='Fig4_Role_Division';fig=db.new('Source and assembler roles across countries and Fields',168)
    grid=fig.add_gridspec(3,1,height_ratios=[1.15,1,.88])
    top_axes=row_axes(fig,grid[0],2,width_ratios=[2.4,1])
    w=db.read('role_country_field_period');p=db.c['comparison_years'][0];latest=w[w.period.eq(p)].copy();country=db.read('role_country_period');recent=country[country.period.eq(p)]
    ax=top_axes[0];title(ax,'a','Country x Fields role atlas')
    top=recent.nlargest(13,'E_asm').country;latest['role_gap']=latest.LQ_asm-latest.LQ_src
    t=latest.pivot(index='country',columns='field',values='role_gap').reindex(top)
    support=latest.pivot(index='country',columns='field',values='E_asm').reindex(top)
    t=t.where(support>=db.c['min_cell_mass']);limit=max(.2,float(np.nanquantile(np.abs(t.to_numpy()),.95))) if np.isfinite(t.to_numpy()).any() else 1
    im=ax.imshow(t,aspect='auto',cmap='BrBG',vmin=-limit,vmax=limit)
    ax.set_yticks(np.arange(len(t)),[db.country[i] for i in t.index]);ax.set_xticks(np.arange(len(t.columns)),[short(db.field[i]) for i in t.columns],rotation=65,ha='right',fontsize=5.5)
    cb=fig.colorbar(im,ax=ax,fraction=.03,pad=.02,extend='both');cb.set_label('LQ assembler - LQ source',fontsize=5.8)
    db.use(name,'a','role_country_field_period','Top 13 current assembler-exposure countries; cells below min exposure masked; clipping only visual')
    ax=top_axes[1];title(ax,'b','Credit and exposure concentration');t=db.read('role_lorenz');t=t[(t.period==p)&(t.scope=='country_field')]
    for k,role in enumerate(['src','asm']):
        for quantity,ls in [('E','--'),('B','-')]:
            a=t[(t.role==role)&(t.quantity==quantity)];ax.plot(a.pop_frac,a.cum_share,color=[NAVY,TEAL][k],ls=ls,label=role+' '+quantity)
    ax.plot([0,1],[0,1],color=GRAY,lw=.5,ls=':');ax.set(xlabel='Cumulative unit share',ylabel='Cumulative E or B share',xlim=(0,1),ylim=(0,1));legend(ax,loc='upper left');db.use(name,'b','role_lorenz','Same fixed union of active country x Fields units for all four distributions')
    middle=row_axes(fig,grid[1],3)
    ax=middle[0];title(ax,'c','Specialisation and assembly role')
    t=latest[(latest.E_asm>=db.c['min_cell_mass'])&latest.RCA.notna()&latest.LQ_asm.notna()]
    if len(t):
        hb=ax.hexbin(np.log2(1+t.RCA),np.log2(1+t.LQ_asm),gridsize=24,mincnt=1,cmap='YlGnBu',linewidths=0)
        cb=fig.colorbar(hb,ax=ax,fraction=.04,pad=.03);cb.set_label('Cells',fontsize=5.5)
    else:empty(ax,'No defined role quotients at the selected exposure threshold')
    ax.set(xlabel='log2(1 + RCA)',ylabel='log2(1 + assembler LQ)');db.use(name,'c','role_country_field_period','RCA and LQ share P*; log1p transformation retains zero credit')
    ax=middle[1];title(ax,'d','Exposure and source role')
    t=latest[latest.E_src>=db.c['min_cell_mass']];ax.scatter(t.E_src,t.R_src,s=3,c=TEAL,alpha=.35,rasterized=True)
    ax.set(xscale='log',xlabel='Source exposure E',ylabel='Source bridge intensity',ylim=(0,1))
    if len(t) and t.E_src.min()>0:
        ticks=np.geomspace(t.E_src.min(),t.E_src.max(),3);ax.set_xticks(ticks,[f'{v:.0f}' if v<1000 else f'{v/1000:.1f}k' for v in ticks])
        from matplotlib.ticker import NullFormatter
        ax.xaxis.set_minor_formatter(NullFormatter())
    db.use(name,'d','role_country_field_period','Country x Fields units; denominator and minimum exposure shown')
    ax=middle[2];title(ax,'e','Network position and assembly role')
    nr=db.read('network_node_roles');nr=nr[(nr.period==p)&(nr.partition=='fixed_F')&(nr.layer=='H')]
    t=recent.merge(nr[['country','participation_out']],on='country');ax.scatter(t.participation_out,t.LQ_asm,s=10,color=NAVY,alpha=.65)
    for r in t.nlargest(4,'E_asm').itertuples():
        if np.isfinite(r.participation_out):country_label(ax,r.participation_out,r.LQ_asm,db.country[r.country],fontsize=5.5)
    ax.set(xlabel='H outgoing participation',ylabel='Country assembler LQ');db.use(name,'e','network_node_roles; role_country_period','Country means; baseline F partition fixed across windows; undefined nodes excluded')
    bottom=row_axes(fig,grid[2],2)
    ax=bottom[0];title(ax,'f','Continuous role trajectories')
    chosen=recent.nlargest(4,'E_asm').country
    for k,country_id in enumerate(chosen):
        t=country[country.country.eq(country_id)].sort_values('period')
        ax.plot(t.period+2,t.LQ_asm,color=COLORS[k],label=db.country[country_id]);ax.plot(t.period+2,t.LQ_src,color=COLORS[k],ls='--',lw=.9)
    ax.axhline(1,color=GRAY,lw=.5,ls=':');ax.set(xlabel='Window midpoint',ylabel='Role location quotient')
    country_key=ax.legend(frameon=False,ncol=2,loc='upper right',handlelength=1.5)
    ax.add_artist(country_key)
    ax.legend(handles=[Line2D([],[],color='black',label='Assembler'),
                       Line2D([],[],color='black',ls='--',label='Source')],
              loc='upper left',frameon=False,handlelength=1.5,fontsize=5.5)
    db.use(name,'f','role_country_period','Four countries selected by recent E_asm; solid assembler and dashed source; continuous ratios of summed credits')
    ax=bottom[1];title(ax,'g','Within fixed high-exposure countries')
    club=db.read('fixed_exposure_clubs')
    for layer,color,ls in [('F',NAVY,'--'),('H',TEAL,'-')]:
        for frac in [.05,.1,.2]:
            t=club[(club.layer==layer)&np.isclose(club.fraction,frac)]
            ax.plot(t.period+2,t.share_inside,color=color,ls=ls,lw=1.5 if frac==.1 else .6,alpha=1 if frac==.1 else .3,label=layer+' (10%)' if frac==.1 else None)
    ax.set(xlabel='Window midpoint',ylabel='International mass within set',ylim=(0,1));legend(ax)
    db.use(name,'g','fixed_exposure_clubs','Countries chosen on 2010-2014 F strength and frozen; thin faint curves show 5% and 20% cutoffs; no outcome-selected club')
    db.save(fig,name,'E denotes exposure; B denotes context-dependent bridge credit; LQ=(B/E)/(world mean q). These are structural accounts, not scientific ability or quality. RCA uses the same P* output. Concentration uses one fixed unit universe. In f, solid lines are assembler roles and dashed lines are source roles. Trajectories use ratios of summed credits; network roles condition on a frozen baseline F partition. In g, thin faint curves show the 5% and 20% cutoffs, and thick curves show 10%. Current source stock and current paper assembly are different roles.')


def figure5(db):
    name='Fig5_Composition_Coverage';fig=db.new('Composition, reference history and measurement scope',164)
    axes=fig.subplot_mosaic('AB;CD;EF',gridspec_kw={'height_ratios':[1,1,1]})
    ax=axes['A'];title(ax,'a','Exact change accounting');a=db.read('change_accounting')
    value=a.value.to_numpy();cum=0
    for k,(label,v) in enumerate(zip(a.component,value)):
        ax.bar(k,v,bottom=cum,color=TEAL if v>=0 else ORANGE,width=.65);ax.plot([k-.32,k+.32],[cum+v,cum+v],color=NAVY,lw=.5);cum+=v
    ax.axhline(0,color=GRAY,lw=.5);ax.set_xticks(np.arange(len(a)),[x.capitalize() for x in a.component],rotation=25,ha='right');ax.set_ylabel('Change in mean q');ax.text(.03,.98,f'Total: {cum:+.4f}',transform=ax.transAxes,va='top',fontsize=6.5)
    db.use(name,'a','change_accounting','2010-2014 vs 2020-2024; exact symmetric decomposition, not causal attribution')
    ax=axes['B'];title(ax,'b','Source cohorts and bridge intensity');t=db.read('cohort_intensity')
    table=t.pivot(index='citing_period',columns='source_cohort',values='intensity');expo=t.pivot(index='citing_period',columns='source_cohort',values='E')
    table=table.where(expo>=db.c['min_cell_mass']);im=ax.imshow(table,aspect='auto',origin='lower',cmap='viridis',vmin=0,vmax=1)
    ax.set_yticks(np.arange(len(table)),[f'{p}-{str(p+4)[-2:]}' for p in table.index]);cols=table.columns.to_numpy();ticks=np.unique(np.r_[0,np.linspace(0,len(cols)-1,5,dtype=int)])
    ax.set_xticks(ticks,['<1950' if cols[x]==1945 else str(cols[x]) for x in ticks],rotation=45,ha='right');ax.set(xlabel='Source publication cohort',ylabel='Citing window');fig.colorbar(im,ax=ax,fraction=.035,pad=.025,label='B / E')
    db.use(name,'b','cohort_intensity','All source ages retained; pre-1950 pooled only for plotting; intensity is sum(b)/sum(1/m)')
    ax=axes['C'];title(ax,'c','Observed and reference compositions');a=db.read('network_annual')
    for column,den,label,color,ls in [('H_model','N_model','Observed model subset',TEAL,'-'),('H0','N_model','Label-mixing reference',ORANGE,'--'),('H_context','N','Context mean (all P*)',NAVY,':')]:
        v=np.divide(a[column],a[den],out=np.full(len(a),np.nan),where=a[den]>0);ax.plot(a.year,v,label=label,color=color,ls=ls)
    ax.set(xlabel='Citing year',ylabel='Mean bridge intensity');legend(ax,loc='best');db.use(name,'c','network_annual','Observed model subset and H0 share denominator; model coverage separately reported; context uses all P*')
    ax=axes['D'];title(ax,'d','Definition and support sensitivity');a=db.read('definition_sensitivity')
    labels={'main':'Main','at_least_5':'>=5 joint references','coverage_ge_08':'>=80% core coverage','primary_same_refs':'Primary Fields, same refs'}
    for k,(variant,t) in enumerate(a.groupby('variant')):ax.plot(t.year,t['mean'],label=labels[variant],color=COLORS[k],lw=.9)
    fixed=db.read('fixed_support_mean')
    if len(fixed):ax.plot(fixed.year,fixed['mean'],label='Fixed baseline cells',color=GRAY,ls='--',lw=.8)
    ax.set(xlabel='Citing year',ylabel='Mean joint co-use');legend(ax,loc='best',fontsize=5.5);db.use(name,'d','definition_sensitivity; fixed_support_mean','Each curve has its own explicit support; fixed weights valid only from configured support start')
    ax=axes['E'];title(ax,'e','Completion ranges within core references');a=db.read('paper_annual')
    ax.fill_between(a.year,a.completion_lower_mean,a.completion_upper_mean,color=TEAL,alpha=.2,label='Completion range')
    ax.plot(a.year,a.joint_cross_mean,color=TEAL,label='Observed joint set');ax.set(xlabel='Citing year',ylabel='Mean joint co-use',ylim=(0,1));legend(ax)
    db.use(name,'e','paper_annual','Conditional P*: unknown classifications among core references only; ranges are not confidence intervals')
    ax=axes['F'];joint=db.read('joint_semantic')
    if len(joint)>=3:
        title(ax,'f','Same-reference semantic cross-check');hb=ax.hexbin(joint.q_same_refs,joint.cosine_pair_distance,gridsize=28,mincnt=1,cmap='YlGnBu',linewidths=0)
        cb=fig.colorbar(hb,ax=ax,fraction=.035,pad=.03);cb.set_label('Papers',fontsize=5.5)
        ax.set(xlabel='Joint co-use (same refs)',ylabel='Mean cosine pair distance');db.use(name,'f','joint_semantic','Exact same country/Fields/embedding reference set; probability sample')
    else:
        title(ax,'f','Broader-scope semantic association');s=pd.read_parquet(db.d['data']/'plot_sample.parquet');s=s[s.semantic_existing.notna()&s.semantic_coverage.ge(.5)]
        if len(s)>=3:
            hb=ax.hexbin(s.joint_cross,s.semantic_existing,gridsize=28,mincnt=1,cmap='YlGnBu',linewidths=0)
            cb=fig.colorbar(hb,ax=ax,fraction=.035,pad=.03);cb.set_label('Papers',fontsize=5.5)
            ax.set(xlabel='Joint co-use (joint refs)',ylabel='Semantic distance\n(broader reference set)')
        else:empty(ax,'Embedding matrix not configured. Exact same-reference semantic check is available but has not run.')
        db.use(name,'f','plot_sample.parquet','Existing semantic metric has a different broader reference scope; not exact same-reference validation')
    semantic_scope=('Panel f uses identical country/Fields/embedding-valid references for both metrics.'
                    if len(joint)>=3 else 'Panel f uses different reference sets: joint co-use uses jointly country/Fields-known references; the existing semantic distance uses a broader reference set. This is not same-reference validation.')
    db.save(fig,name,'Decomposition is descriptive accounting. Label-mixing pools condition on citing year, receiving Fields, reference age and source type and leave out the entire focal paper; they do not preserve individual cited-work degrees or establish exogenous availability. Missing-classification ranges concern the observed core graph only. '+semantic_scope+' No causal interpretation is made.')


def build(c):
    db=Data(c)
    for fn in (figure1,figure2,figure3,figure4,figure5):
        print('Rendering '+fn.__name__,flush=True);fn(db)
    outputs=sorted(p.name for p in db.d['figures'].glob('*') if p.is_file())
    json_write({'files':outputs,'main_figures':5,'conceptual_panels':32,
        'formats':['editable PDF','editable SVG','450 dpi PNG'],
        'layout':'NHB compact; 180 mm width; no in-image titles or footer captions',
        'figure_heights_mm':[154,168,170,168,164],
        'legend_location':'Fig*_caption.txt (figure title, panel descriptions and explanatory legend)',
        'synthetic':bool(c.get('synthetic',False))},db.d['status']/'figure_outputs.json')
    return dict(figures=5,panels=32,files=len(outputs))
