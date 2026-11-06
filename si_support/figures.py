"""Numbered SI figures, exact plotted data, legends and a vector PDF."""
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from .core import MissingInput,divide

TEAL='#008B83';NAVY='#254B69';ORANGE='#D18436';PURPLE='#7D71A7';GRAY='#73808A'
PALETTE=[TEAL,ORANGE,NAVY,PURPLE,'#AD5362','#6C985E','#67A6BF']
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','Liberation Sans','DejaVu Sans'],
    'font.size':6.3,'axes.labelsize':6.3,'axes.titlesize':7,'xtick.labelsize':5.5,'ytick.labelsize':5.5,
    'legend.fontsize':5.5,'axes.linewidth':.5,'lines.linewidth':1.,'axes.spines.top':False,
    'axes.spines.right':False,'axes.labelpad':2,'xtick.major.pad':2,'ytick.major.pad':2,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white',
    'savefig.facecolor':'white','figure.constrained_layout.h_pad':.03,'figure.constrained_layout.w_pad':.03,
    'figure.constrained_layout.hspace':.06,'figure.constrained_layout.wspace':.07})


def new(rows=2,cols=2,height=155):
    fig,axes=plt.subplots(rows,cols,figsize=(180/25.4,height/25.4),layout='constrained',squeeze=False)
    for letter,ax in zip('abcdefghijklmnop',axes.flat):ax.set_title(letter,loc='left',fontsize=7,fontweight='bold',pad=2)
    return fig,list(axes.flat)


def key(ax,**kwargs):
    return ax.legend(frameon=False,handlelength=1.5,**kwargs)


def heat(fig,ax,table,label,cmap='viridis',vmin=None,vmax=None,center=None,fmt=None):
    if table.empty:raise MissingInput('No supported values for heatmap '+label)
    cmap=plt.get_cmap(cmap).copy();cmap.set_bad('#E3E6E8')
    values=table.to_numpy(float)
    options=dict(cmap=cmap,aspect='auto',interpolation='nearest')
    if center is not None:options['norm']=TwoSlopeNorm(vmin=vmin,vcenter=center,vmax=vmax)
    else:options.update(vmin=vmin,vmax=vmax)
    image=ax.imshow(values,**options)
    ticks=np.arange(len(table.columns)) if len(table.columns)<=12 else np.unique(np.linspace(0,len(table.columns)-1,5,dtype=int))
    ax.set_xticks(ticks,[str(table.columns[i]) for i in ticks],rotation=45,ha='right')
    ax.set_yticks(np.arange(len(table)),list(map(str,table.index)),fontsize=5.2)
    if fmt:
        for (i,j),v in np.ndenumerate(values):
            if np.isfinite(v):ax.text(j,i,format(v,fmt),ha='center',va='center',fontsize=5.2,color='white' if v<(vmin or 0)+.5*((vmax or np.nanmax(values))-(vmin or 0)) else 'black')
    cb=fig.colorbar(image,ax=ax,fraction=.035,pad=.02);cb.set_label(label,fontsize=5.5)
    return image


def latest_countries(ctx,z):
    return z[z.period.eq(ctx.current)].nlargest(ctx.top,'E_asm').country.to_list()


class Exporter:
    def __init__(self,ctx,book=True):
        self.ctx=ctx
        self.pdf=PdfPages(ctx.out/'Supplementary_Information.pdf') if book else None

    def save(self,fig,number,slug,title,caption,panels):
        ctx=self.ctx;name=f'Supplementary_Fig_S{number:02d}_{slug}'
        fig.canvas.draw();fig.set_layout_engine('none')
        watermark=None
        if ctx.synthetic:watermark=fig.text(.5,.5,'SYNTHETIC / LAYOUT CHECK',fontsize=24,rotation=30,color='#B04C4C',alpha=.10,ha='center',va='center')
        paths=[]
        for ext in ('pdf','svg','png'):
            path=ctx.d['figures']/(name+'.'+ext);fig.savefig(path,dpi=450);paths.append(str(path))
        legend=f'Supplementary Fig. S{number} | {title}.\n\n{caption}\n'
        (ctx.d['figures']/(name+'_caption.txt')).write_text(legend,encoding='utf-8')
        sources=[ctx.source(name,panel,data) for panel,data in panels]
        ctx.figures.append(dict(number=f'Supplementary Fig. S{number}',title=title,caption=caption,files=paths,source_data=sources))
        if self.pdf:
            width,height=fig.get_size_inches();positions=[ax.get_position().frozen() for ax in fig.axes]
            page_w,page_h=210/25.4,297/25.4;left=15/25.4;bottom=page_h-30/25.4-height
            fig.set_size_inches(page_w,page_h)
            for ax,pos in zip(fig.axes,positions):ax.set_position([(left+pos.x0*width)/page_w,(bottom+pos.y0*height)/page_h,pos.width*width/page_w,pos.height*height/page_h])
            if watermark:watermark.set_position((.5,(bottom+height/2)/page_h))
            fig.text(15/210,281/297,f'Supplementary Fig. S{number} | {title}',fontsize=10,ha='left',va='top')
            wrapped=textwrap.fill(caption,width=112,break_long_words=False)
            lines=wrapped.count('\n')+1
            available=(bottom-14/25.4)*72
            if lines*10.5>available:raise ValueError(f'{name}: legend is too long for its A4 SI page; shorten it or reduce figure height.')
            fig.text(15/210,(bottom-6/25.4)/page_h,wrapped,fontsize=8,linespacing=1.3,ha='left',va='top')
            fig.text(.93,.025,str(self.pdf.get_pagecount()+1),fontsize=7,ha='right')
            self.pdf.savefig(fig)
        plt.close(fig)

    def tables(self):
        if not self.pdf:return
        ctx=self.ctx
        specs=[
          ('S01_population','Annual population and coverage',['year','one','country_known','eligible','eligible_fraction']),
          ('S04_endpoints','Alternative observation windows',['start_year','end_year','N','mean_q']),
          ('S05_reference_comparison','Annual model-subset reference comparison',['year','N_model','q_model','q0','gap','observed_to_reference']),
          ('S11_country_roles','Recent country roles; largest assembly exposures',['country_code','E_src','E_asm','LQ_src','LQ_asm','LQ_gap']),
          ('S16_resolution_similarity','Recent cross-layer agreement by resolution',['gamma','layers','AMI','VI_normalized','n_shared']),
          ('S21_change_by_field','Exact change contributions by field',['field_label','within','composition','entry','exit','total']),
          ('S25_regression_coefficients','Reported regression coefficients',['model','term','estimate','conf_low','conf_high','n']),
          ('S26_regression_scope','Regression sample counts and covariance',['model','unit','n_export','n_complete','n_model','covariance']),
          ('S32_semantic_summary','Semantic associations and reference scope',['scope','n','rho'])]
        table_number=0
        for name,title,columns in specs:
            try:data=ctx.table(name)
            except MissingInput:continue
            if 'period' in data:data=data[data.period.eq(ctx.current)]
            if name=='S11_country_roles':data=data.nlargest(ctx.top,'E_asm')
            columns=[x for x in columns if x in data];data=data[columns]
            if data.empty:continue
            table_number+=1
            for start in range(0,len(data),24):
                part=data.iloc[start:start+24];fig=plt.figure(figsize=(210/25.4,297/25.4))
                ax=fig.add_axes([.06,.15,.88,.72]);ax.set_axis_off()
                def display(x,col):
                    if pd.isna(x):return 'NA'
                    if col in ['year','period','start_year','end_year']:return str(int(x))
                    if col in ['n','N','N_model','n_export','n_complete','n_model','one','country_known','eligible','n_shared']:return f'{x:,.0f}'
                    if isinstance(x,(int,np.integer)):return f'{x:,}'
                    if col.startswith('E_') and isinstance(x,(float,np.floating)):return f'{x:,.2f}'
                    if isinstance(x,(float,np.floating)):return f'{x:.5g}'
                    x={'same_reference_set':'Same reference set',
                       'different_broader_reference_set':'Different, broader reference set'}.get(str(x),str(x))
                    return textwrap.fill(str(x),width=24)
                values=[[display(v,col) for v,col in zip(row,columns)] for row in part.itertuples(index=False,name=None)]
                header_names={'one':'Core papers','eligible':'P* papers','country_known':'Country-known papers',
                              'eligible_fraction':'P* / core','N_model':'Model subset N','q_model':'Observed q',
                              'q0':'Label reference q0','gap':'Observed - reference'}
                headers=[textwrap.fill(header_names.get(x,x.replace('_',' ')),width=18) for x in columns]
                tab=ax.table(cellText=values,colLabels=headers,loc='upper center',cellLoc='left',colLoc='left')
                tab.auto_set_font_size(False);tab.set_fontsize(6.3)
                line_rows=max([max(str(v).count('\n')+1 for v in row) for row in values]+[2])
                for (r,c),cell in tab.get_celld().items():
                    cell.set_height(max(.037,.018*line_rows));cell.set_linewidth(.25);cell.set_edgecolor('#CCD7DC')
                    if r==0:cell.set_facecolor('#E0ECEE');cell.set_text_props(weight='bold')
                suffix=f' (rows {start+1}-{start+len(part)})' if len(data)>24 else ''
                fig.text(.065,.95,f'Supplementary Table S{table_number} | {title}{suffix}',fontsize=10,va='top')
                scope=ctx.inventory.get(name,{}).get('scope','See the full CSV and SI table inventory for the unit and scope.')
                note=f'Full numerical table: {name}.csv. NA denotes undefined or unavailable values. '+scope
                if name=='S11_country_roles':note+=' This PDF preview is restricted by assembly exposure; the CSV retains every country.'
                fig.text(.065,.10,textwrap.fill(note,112),fontsize=8,va='top',linespacing=1.3)
                if ctx.synthetic:fig.text(.5,.5,'SYNTHETIC / LAYOUT CHECK',ha='center',fontsize=23,rotation=30,color='#B04C4C',alpha=.12)
                fig.text(.93,.025,str(self.pdf.get_pagecount()+1),fontsize=7,ha='right')
                self.pdf.savefig(fig);plt.close(fig)

    def close(self):
        if self.pdf:self.pdf.close()


def fig01(ctx,e):
    z=ctx.table('S01_population');fig,ax=new()
    for col,label,color in [('one','Core',NAVY),('country_known','Country known',GRAY),('eligible','P*',TEAL)]:ax[0].plot(z.year,z[col]/1e6,label=label,color=color)
    ax[0].set(xlabel='Publication year',ylabel='Papers (million)');key(ax[0])
    for col,label in [('country_known_fraction','Country known'),('field_known_fraction','Fields known'),('eligible_fraction','P* / core')]:ax[1].plot(z.year,z[col],label=label)
    ax[1].set(xlabel='Publication year',ylabel='Fraction of dated core',ylim=(0,1));key(ax[1])
    ax[2].plot(z.year,z.eligible_given_country_known,color=TEAL);ax[2].set(xlabel='Publication year',ylabel='P* / country-known papers',ylim=(0,1))
    recent=z.tail(6)
    for k,(col,label) in enumerate([('country_known_fraction','Country known'),('eligible_fraction','P*')]):ax[3].bar(np.arange(len(recent))+(k-.5)*.35,recent[col],width=.35,label=label)
    ax[3].set_xticks(np.arange(len(recent)),recent.year.astype(int),rotation=45);ax[3].set(ylabel='Fraction of recent core',ylim=(0,1));key(ax[3])
    e.save(fig,1,'Population_coverage','Population support and annual coverage','a, Observed core, country-known and P* counts. b, Metadata and main-population coverage. c, P* as a fraction of country-known papers. d, Recent coverage diagnostic. Denominators are dated core papers in the stated year; the core induced graph does not cover all published references. Counts and proportions are descriptive.',[('a_d',z)])


def fig02(ctx,e):
    d=ctx.table('S27_sample_distributions');y=ctx.table('S28_sample_year');bins=ctx.table('S29_sample_coverage');s=ctx.sample();fig,ax=new(2,3,160)
    for a,metric in zip(ax[:3],['known_fields','known_countries','joint_refs']):
        z=d[d.metric.eq(metric)]
        if metric=='joint_refs':a.plot(z.value,z.fraction,'o-',ms=2,color=TEAL);a.set_xscale('log')
        else:
            counts=z.assign(category=np.minimum(z.value.astype(int),5)).groupby('category').fraction.sum()
            a.bar(np.arange(len(counts)),counts.to_numpy(),color=TEAL)
            a.set_xticks(np.arange(len(counts)),['5+' if v==5 else str(int(v)) for v in counts.index]);a.set_ylim(0,1)
        a.set(xlabel=metric.replace('_',' '),ylabel='Fraction of selected sample')
    ax[3].hist(s.reference_coverage.dropna(),bins=np.linspace(0,1,21),weights=np.full(s.reference_coverage.notna().sum(),1/max(1,len(s))),color=NAVY)
    ax[3].set(xlabel='Joint / observed core references',ylabel='Fraction of selected sample')
    ax[4].plot(bins.mean_coverage,bins.mean_q,'o-',color=TEAL,ms=2);ax[4].set(xlabel='Mean coverage within bin',ylabel='Mean joint co-use')
    ax[5].plot(y.year,y.n_sample,color=NAVY);ax[5].set(xlabel='Publication year',ylabel='Selected sample papers')
    n=len(s)
    e.save(fig,2,'Sample_metadata','Reference and classification diagnostics in the plot sample',f'a-c, Observed Fields, countries and usable reference counts; a-b pool counts of five or more for display, with all counts retained numerically. d, Joint-classification coverage within observed core references. e, Co-use by coverage bin. f, Sample support over time. N={n:,}. {ctx.sample_scope} These are sample descriptions; they exclude papers outside P* and do not correct metadata selection. A one-label distribution is not evidence of successful multilabel measurement.',[('a_c',d),('d',s[['reference_coverage']]),('e',bins),('f',y)])


def fig03(ctx,e):
    d=ctx.table('S02_definition_sensitivity');fixed=ctx.table('S03_fixed_support');end=ctx.table('S04_endpoints');fig,ax=new()
    for k,(name,z) in enumerate(d.groupby('variant')):
        ax[0].plot(z.year,z['mean'],label=name.replace('_',' '),color=PALETTE[k%7])
        ax[1].plot(z.year,z.retained_fraction,label=name.replace('_',' '),color=PALETTE[k%7])
    ax[0].set(xlabel='Citing year',ylabel='Mean joint co-use');key(ax[0],fontsize=5.2)
    ax[1].set(xlabel='Citing year',ylabel='Retained P* fraction',ylim=(0,1.02));key(ax[1],fontsize=5.2)
    main=d[d.variant.eq('main')];ax[2].plot(main.year,main['mean'],label='Observed main',color=NAVY)
    if len(fixed):ax[2].plot(fixed.year,fixed['mean'],label='Fixed baseline cells',color=TEAL)
    ax[2].set(xlabel='Citing year',ylabel='Mean joint co-use');key(ax[2])
    ax[3].bar(np.arange(len(end)),end.mean_q,color=[NAVY,ORANGE,TEAL][:len(end)])
    ax[3].set_xticks(np.arange(len(end)),[f'{int(a)}-{int(b)}' for a,b in zip(end.start_year,end.end_year)],rotation=25,ha='right');ax[3].set(ylabel='Paper-weighted mean q')
    e.save(fig,3,'Definition_endpoints','Definitions, fixed support and endpoint windows','a, Annual means for each sensitivity definition. b, Retained paper fractions for each definition. c, Fixed baseline-weight curve on its eligible years. d, Period means using alternative observation windows. Each curve uses its stated definition and support. Retained fixed-support baseline mass and cell counts are supplied in the numerical table.',[('a_b',d),('c',fixed),('d',end)])


def fig04(ctx,e):
    z=ctx.table('S05_reference_comparison');fig,ax=new()
    for col,label,color,ls in [('q_model','Observed model subset',TEAL,'-'),('q0','Label-mixing reference',ORANGE,'--'),('q_context','Empirical context mean',NAVY,':')]:ax[0].plot(z.year,z[col],label=label,color=color,ls=ls)
    ax[0].set(xlabel='Citing year',ylabel='Mean joint co-use');key(ax[0])
    ax[1].plot(z.year,z.gap,color=TEAL);ax[1].axhline(0,color=GRAY,lw=.5);ax[1].set(xlabel='Citing year',ylabel='Observed minus label reference')
    ax[2].plot(z.year,z.observed_to_reference,color=NAVY);ax[2].axhline(1,color=GRAY,lw=.5);ax[2].set(xlabel='Citing year',ylabel='Observed / label reference')
    ax[3].plot(z.year,z.model_coverage,color=TEAL);ax[3].set(xlabel='Citing year',ylabel='Reference-defined / P* papers',ylim=(0,1.02))
    e.save(fig,4,'Reference_gap','Observed co-use relative to label-mixing references','a, Observed, label-mixing and empirical-context summaries. b-c, Difference and ratio for H_model and H0 on the identical model-defined subset. d, Subset coverage. The empirical-context mean uses all P*. The label model leaves out the focal paper but preserves neither individual cited-work degrees nor an exogenous opportunity set; gaps are descriptive.',[('a_d',z)])


def fig05(ctx,e):
    z=ctx.table('S06_field_year');fig,ax=new(height=178)
    for a,metric in zip(ax,['geo_cross','field_cross','joint_cross','coupling']):
        t=z[z.metric.eq(metric)].pivot(index='field',columns='year',values='mean').reindex(sorted(ctx.field));t.index=t.index.map(ctx.field)
        if metric=='coupling':limit=max(.02,float(np.nanquantile(np.abs(t.to_numpy()),.95)));heat(fig,a,t,'q - geography x Fields','PuOr_r',-limit,limit,0)
        else:heat(fig,a,t,metric.replace('_',' '),vmin=0,vmax=1)
        a.set_xlabel('Citing year')
    e.save(fig,5,'Field_trends','Field-specific boundary metrics through time','a-c, Geographic, Fields and joint co-use means by receiving field and citing year. d, Mean within-paper coupling q-g*d, computed within papers before aggregation. Fractional field masses are retained. Coupling is a descriptive dependence measure, not cognitive integration or a causal interaction. Undefined cells are grey; all numerical means and fractional masses accompany the figure.',[('a_d',z)])


def fig06(ctx,e):
    z=ctx.table('S08_distribution_cdf');fig,ax=new(3,2,178)
    for r,metric in enumerate(['geo_cross','field_cross','joint_cross']):
        for group in [0,1]:
            a=ax[r*2+group];t=z[(z.metric==metric)&(z.international==group)]
            for k,(period,b) in enumerate(t.groupby('period')):
                b=b.sort_values('bin');a.step(np.r_[0,b.hi],np.r_[b.zero_fraction.iloc[0],b.cdf],where='post',color=PALETTE[k%7],label=f'{period}-{str(period+4)[-2:]}')
            a.set(xlabel=metric.replace('_',' '),ylabel='CDF: '+('one country' if group==0 else '2+ countries'),xlim=(0,1),ylim=(0,1))
            if r==0 and group==0:key(a,fontsize=5.2,ncol=2)
    e.save(fig,6,'Full_distributions','Full-population boundary distributions by period and team geography','Rows show geography, Fields and joint co-use; columns show one versus two or more observed institutional countries. CDFs combine the zero atom with positive histogram counts for all P* papers. Zero metadata are counted once per year/group, not once per bin. The companion quantile table reports bin-edge approximations rather than exact raw-data quantiles.',[('a_f',z)])


def fig07(ctx,e):
    z=ctx.table('S11_country_roles');countries=latest_countries(ctx,z);t=z[(z.period==ctx.current)&z.country.isin(countries)].set_index('country').reindex(countries).reset_index();fig,ax=new(height=165)
    positions=np.arange(len(t));names=t.country_code
    ax[0].barh(positions-.17,t.LQ_src,height=.32,label='Source',color=NAVY);ax[0].barh(positions+.17,t.LQ_asm,height=.32,label='Assembler',color=TEAL)
    ax[0].set_yticks(positions,names);ax[0].invert_yaxis();ax[0].axvline(1,color=GRAY,lw=.5);ax[0].set_xlabel('Role location quotient');key(ax[0])
    ax[1].scatter(t.LQ_src,t.LQ_asm,s=8+35*np.sqrt(t.E_asm/max(t.E_asm.max(),1)),color=TEAL,alpha=.7)
    ax[1].axhline(1,color=GRAY,lw=.5);ax[1].axvline(1,color=GRAY,lw=.5);ax[1].set(xlabel='Source LQ',ylabel='Assembler LQ')
    for k,country in enumerate(countries[:6]):
        a=z[z.country==country].sort_values('period');ax[2].plot(a.period+2,a.LQ_gap,label=ctx.country[country],color=PALETTE[k%7])
    ax[2].axhline(0,color=GRAY,lw=.5);ax[2].set(xlabel='Window midpoint',ylabel='Assembler minus source LQ');key(ax[2],ncol=3)
    if {'model_gap_src','model_gap_asm'}<=set(t):
        ax[3].scatter(t.model_gap_src,t.model_gap_asm,s=12,color=NAVY);ax[3].axvline(0,color=GRAY,lw=.5);ax[3].axhline(0,color=GRAY,lw=.5)
        ax[3].set(xlabel='Source observed - reference',ylabel='Assembler observed - reference')
    else:
        ax[3].scatter(t.E_src,t.E_asm,s=12,color=NAVY);ax[3].set(xscale='log',yscale='log',xlabel='Source exposure',ylabel='Assembler exposure')
    scope=('same-subset observed-minus-reference intensities' if {'model_gap_src','model_gap_asm'}<=set(t) else 'source and assembler exposures')
    e.save(fig,7,'Country_roles','Country roles and their systematic differences',f'a, Source and assembler LQs for the {len(t)} countries selected by recent assembly exposure. b, Their paired role intensities; size denotes assembly exposure. c, Continuous role gaps for the six highest-exposure countries. d, {scope}. Country rates are ratios of summed credits to exposures. Historical source stocks and contemporary assembly are distinct accounts. Complete country data are exported.',[('a_b_d',t),('c',z[z.country.isin(countries[:6])])])


def fig08(ctx,e):
    z=ctx.table('S10_country_field_roles');c=ctx.table('S11_country_roles');countries=latest_countries(ctx,c)
    z=z[(z.period==ctx.current)&z.country.isin(countries)].copy();fig,ax=new(height=178)
    joint_values=z[['LQ_src','LQ_asm']].to_numpy();finite=joint_values[np.isfinite(joint_values)]
    common_max=max(1.5,float(np.quantile(finite,.95))) if len(finite) else 1.5
    for a,col,label in zip(ax,['LQ_src','LQ_asm','LQ_gap','E_asm'],['Source LQ','Assembler LQ','Assembler - source LQ','log10(1 + assembly exposure)']):
        support=z.both_roles_supported if col=='LQ_gap' else (z['E_src' if col=='LQ_src' else 'E_asm']>=ctx.minimum)
        values=z[col].where(support)
        if col=='E_asm':values=np.log10(1+values)
        t=z.assign(value=values).pivot(index='country',columns='field',values='value').reindex(countries)
        t.index=t.index.map(ctx.country);t.columns=t.columns.map(ctx.field)
        if col=='LQ_gap':
            finite=t.to_numpy()[np.isfinite(t.to_numpy())];limit=max(.2,float(np.quantile(np.abs(finite),.95))) if len(finite) else .2
            heat(fig,a,t,label,'BrBG',-limit,limit,0)
        elif col.startswith('LQ'):heat(fig,a,t,label,'BrBG',0,common_max,1)
        else:heat(fig,a,t,label)
        a.set_xticks(np.arange(len(t.columns)),t.columns,rotation=70,ha='right',fontsize=5.0)
    e.save(fig,8,'Country_field_atlas','Source and assembler roles across country-field units',f'a-b, Source and assembler LQs. c, Their paired gap, requiring exposure at least {ctx.minimum:g} in both roles. d, Assembly exposure. Countries are selected by recent E_asm, and the same country/field ordering is used throughout. Individual role panels require their own minimum exposure. Grey means undefined or below threshold; display color limits may clip extremes, with untruncated values retained in Source Data.',[('a_d',z)])


def fig09(ctx,e):
    z=ctx.table('S13_concentration');fig,ax=new(2,3,155)
    for r,scope in enumerate(['country','country_field']):
        for j,metric in enumerate(['Gini','top5_share','top10pct_share']):
            a=ax[r*3+j]
            for role,color in [('src',NAVY),('asm',TEAL)]:
                for quantity,ls in [('E','--'),('B','-')]:
                    t=z[(z.scope==scope)&(z.role==role)&(z.quantity==quantity)]
                    a.plot(t.period+2,t[metric],label=role+' '+quantity,color=color,ls=ls)
            a.set(xlabel='Window midpoint',ylabel=scope.replace('_',' x ')+': '+metric.replace('_',' '),ylim=(0,1))
            if r==0 and j==0:key(a,fontsize=5.2)
    e.save(fig,9,'Concentration','Exposure and credit concentration on fixed unit universes','Rows compare country and country-field units. Columns show Gini, top-five-unit share and top-ten-percent share. Solid curves represent bridge credit B and dashed curves exposure E; navy is source and teal assembler. All four distributions within a scope retain the same fixed union of active units and period-specific zeros. Curves show deterministic concentration summaries.',[('a_f',z)])


def fig10(ctx,e):
    z=ctx.table('S16_resolution_similarity');s=ctx.table('S14_optimisation_stability');fig,ax=new(2,3,155)
    for a,pair in zip(ax[:3],['C-F','C-H','F-H']):
        t=z[z.layers==pair].pivot(index='period',columns='gamma',values='AMI');t.index=[f'{p}-{str(p+4)[-2:]}' for p in t.index]
        heat(fig,a,t,pair+' AMI',vmin=-.05,vmax=1);a.set_xlabel('Resolution gamma')
    for a,layer in zip(ax[3:],['C','F','H']):
        for k,(period,t) in enumerate(s[s.layer==layer].groupby('period')):
            a.plot(t.gamma,t.mean_ami_to_best,'o-',ms=2,color=PALETTE[k%7],label=str(int(period)))
        a.set(xlabel='Resolution gamma',ylabel=layer+': AMI to best run',ylim=(0,1.02))
        if layer=='C':key(a,ncol=2,fontsize=5.2)
    e.save(fig,10,'Community_resolution','Community agreement and optimisation stability across resolutions','a-c, Cross-layer AMI at every stored period and resolution, using the stored best repeats. d-f, Mean AMI of repeated optimisation to the best run for C, F and H. AMI equally weights common assigned countries. Non-diagonal F-exposure coverage, assigned-country counts and community counts are exported separately. Repeated-run AMI describes optimisation stability; the primary resolution follows the main-analysis settings.',[('a_c',z),('d_f',s)])


def fig11(ctx,e):
    mass=ctx.table('S17_network_mass');nodes=ctx.table('S18_network_nodes');edges=ctx.table('S19_network_edges_current');country=ctx.table('S11_country_roles')
    fig,ax=new(2,3,165)
    for layer,color in [('F',NAVY),('H',TEAL)]:
        z=mass[mass.layer==layer];ax[0].plot(z.period+2,z.domestic_fraction,label=layer,color=color)
    ax[0].set(xlabel='Window midpoint',ylabel='Domestic fraction of total mass',ylim=(0,1));key(ax[0])
    plotted=edges[(edges.F>0)&(edges.H>0)].copy()
    if len(plotted):
        ax[1].scatter(plotted.F,plotted.H,s=1.5,alpha=.15,color=TEAL,rasterized=True);limits=[plotted.F.min(),plotted.F.max()];ax[1].plot(limits,limits,color=GRAY,lw=.6,ls='--')
    ax[1].set(xscale='log',yscale='log',xlabel='International F edge weight',ylabel='International H edge weight')
    h=nodes[(nodes.period==ctx.current)&(nodes.layer=='H')].merge(country[country.period==ctx.current],on=['period','country','country_code'],validate='one_to_one')
    ax[2].scatter(h.participation_out,h.LQ_src,s=9,color=NAVY,alpha=.7);ax[2].set(xlabel='H outgoing participation',ylabel='Source LQ')
    ax[3].scatter(h.participation_in,h.LQ_asm,s=9,color=TEAL,alpha=.7);ax[3].set(xlabel='H incoming participation',ylabel='Assembler LQ')
    top=latest_countries(ctx,country)
    table=edges.pivot(index='source',columns='target',values='H_over_F').reindex(index=top,columns=top)
    table.index=table.index.map(ctx.country);table.columns=table.columns.map(ctx.country);heat(fig,ax[4],table,'H / F',vmin=0,vmax=1)
    ax[4].set(xlabel='Receiver country',ylabel='Source country')
    f=nodes[(nodes.period==ctx.current)&(nodes.layer=='F')];paired=f.merge(h[['country','participation_in']],on='country',suffixes=('_F','_H'),validate='one_to_one')
    ax[5].scatter(paired.participation_in_F,paired.participation_in_H,s=9,color=TEAL,alpha=.7);ax[5].plot([0,1],[0,1],color=GRAY,lw=.6,ls='--');ax[5].set(xlabel='F incoming participation',ylabel='H incoming participation',xlim=(0,1),ylim=(0,1))
    e.save(fig,11,'Network_direction','Domestic mass and directed knowledge-use positions','a, Domestic diagonal fractions of F and H. b, International edge weights; zero-H edges remain in the numerical table but are excluded from logarithmic axes. c-d, Source roles against outgoing and assembler roles against incoming H participation. e, H/F for high-assembly-exposure countries; undefined edges are grey. f, Incoming F/H participation. Participation uses non-diagonal strengths and the frozen baseline F partition. Total and international strengths are explicitly separated.',[('a',mass),('b_e',edges),('c_d',h),('f',paired)])


def diverging_bars(ax,z,label):
    y=np.arange(len(z));pos=np.zeros(len(z));neg=np.zeros(len(z))
    for col,color in zip(['within','composition','entry','exit'],[TEAL,ORANGE,PURPLE,GRAY]):
        v=z[col].to_numpy();ax.barh(y,np.maximum(v,0),left=pos,color=color,label=col);ax.barh(y,np.minimum(v,0),left=neg,color=color)
        pos+=np.maximum(v,0);neg+=np.minimum(v,0)
    ax.scatter(z.total,y,marker='x',s=10,color='black');ax.set_yticks(y,z[label],fontsize=5.2);ax.invert_yaxis();ax.axvline(0,color=GRAY,lw=.5);ax.set_xlabel('Contribution to global mean q change')


def fig12(ctx,e):
    f=ctx.table('S21_change_by_field');c=ctx.table('S21_change_by_country');u=ctx.table('S20_change_by_unit');fig,ax=new(height=178)
    f=f.sort_values('total',ascending=False);c=c.assign(abs_total=c.total.abs()).nlargest(min(20,len(c)),'abs_total').sort_values('total',ascending=False)
    diverging_bars(ax[0],f,'field_label');key(ax[0],fontsize=5.2)
    diverging_bars(ax[1],c,'country_label')
    valid=u[(u.E0>0)&(u.E1>0)];ax[2].scatter(valid.within,valid.composition,s=2,alpha=.3,color=TEAL,rasterized=True);ax[2].axhline(0,color=GRAY,lw=.5);ax[2].axvline(0,color=GRAY,lw=.5);ax[2].set(xlabel='Within contribution',ylabel='Composition contribution')
    values=np.sort(np.abs(u.total.to_numpy()))[::-1];shares=np.cumsum(values)/values.sum() if values.sum()>0 else np.zeros_like(values)
    ax[3].plot(np.arange(1,len(values)+1)/max(1,len(values)),shares,color=NAVY);ax[3].set(xlabel='Fraction of units, largest absolute first',ylabel='Cumulative absolute contribution',xlim=(0,1),ylim=(0,1))
    e.save(fig,12,'Composition_details','Country and field contributions to the change in joint co-use','a, Every field contribution, with components separated around zero and net totals marked by crosses. b, Up to 20 countries with the largest absolute net contributions; all countries remain in the full table. c, Common-unit within and composition terms. d, Concentration of absolute unit contributions. Components sum exactly to the global mean difference for the configured baseline and comparison windows. Components describe accounting within the configured country-field units.',[('a',f),('b',c),('c_d',u)])


def fig13(ctx,e):
    z=ctx.table('S22_source_cohorts');a=ctx.table('S23_reference_age');fig,ax=new()
    for plot,metric,label in [(ax[0],'intensity','Bridge credit / exposure'),(ax[1],'exposure_share','Fraction of exposure')]:
        table=z.pivot(index='period',columns='cohort',values=metric);table=table.where(z.pivot(index='period',columns='cohort',values='E')>=ctx.minimum)
        table.index=[f'{p}-{str(p+4)[-2:]}' for p in table.index];table.columns=['<1950' if int(v)==1945 else str(int(v)) for v in table.columns]
        heat(fig,plot,table,label,vmin=0,vmax=1);plot.set(xlabel='Source publication cohort',ylabel='Citing period')
    bands=['0','1-2','3-5','6-10','11-20','21-40','41+']
    for k,(period,t) in enumerate(a.groupby('period')):
        t=t.set_index('age_band').reindex(bands)
        ax[2].plot(np.arange(7),t.exposure_share,'o-',ms=2,color=PALETTE[k%7],label=str(int(period)))
        ax[3].plot(np.arange(7),t.intensity,'o-',ms=2,color=PALETTE[k%7])
    for plot,ylabel in [(ax[2],'Fraction of reference exposure'),(ax[3],'Bridge credit / exposure')]:plot.set_xticks(np.arange(7),bands,rotation=45);plot.set(xlabel='Exact source age band (years)',ylabel=ylabel,ylim=(0,1))
    key(ax[2],ncol=2,fontsize=5.2)
    e.save(fig,13,'Source_history','Reference history, source ages and bridge intensity','a, Source-cohort bridge intensity. b, Cohort share of paper-normalised reference exposure. c-d, Exposure share and intensity by exact reference age band for each citing period. Age is computed from citing and cited years before aggregation, retaining all observed reference ages. Pre-1950 sources are pooled only for display. Masked cohort cells have insufficient exposure; rates are ratios of summed credits and exposures.',[('a_b',z),('c_d',a)])


def fig14(ctx,e):
    summary=ctx.table('S32_semantic_summary');plots=[]
    for name,x,y,label in [('S30_joint_semantic','q_same_refs','cosine_pair_distance','Same reference set'),('S31_broader_semantic','joint_cross','semantic_existing','Different, broader semantic refs')]:
        try:z=ctx.table(name)
        except MissingInput:continue
        if len(z)>=3:plots.append((z,x,y,label))
    if not plots:raise MissingInput('At least three measured semantic records are needed.')
    fig,axes=new(1,len(plots),110);sources=[]
    for k,(a,(z,x,y,label)) in enumerate(zip(axes,plots)):
        h=a.hexbin(z[x],z[y],gridsize=30,mincnt=1,bins='log',cmap='YlGnBu',linewidths=0)
        cb=fig.colorbar(h,ax=a,fraction=.035,pad=.025);cb.set_label('Papers',fontsize=5.5)
        a.set(xlabel='Joint co-use',ylabel='Semantic distance\n'+label,xlim=(0,1));sources.append((chr(97+k),z[[v for v in ['paper_id','year',x,y,'n_joint_embedding_refs','joint_ref_coverage','semantic_coverage'] if v in z]]))
    e.save(fig,14,'Semantic_scope','Semantic associations under explicitly separated reference scopes','Each vertical axis identifies the reference scope of its measured semantic indicator. Same-reference and broader-reference associations occupy separate panels. Hexagon colour gives paper counts on a logarithmic scale. Sample sizes and unweighted Spearman correlations are exported for each scope.',sources+[('summary',summary)])


def fig15(ctx,e):
    z=ctx.table('S25_regression_coefficients');models=list(z.model.unique())
    if not models:raise MissingInput('No regression coefficients.')
    nr=(len(models)+1)//2;fig=plt.figure(figsize=(180/25.4,min(180,56*nr)/25.4),layout='constrained');grid=fig.add_gridspec(nr,2);axes=[]
    names={'international':'2+ observed countries','log_team':'Log team size','log_refs':'Log joint references','mean_ref_age_joint':'Mean reference age','ai_high':'AIProb > 0.5','log1p(RCA)':'Log(1 + RCA)','log1p(E_asm)':'Log(1 + exposure)','known_fields':'Observed Fields count'}
    for k,model in enumerate(models):
        a=fig.add_subplot(grid[k//2,:] if len(models)%2 and k==len(models)-1 else grid[k//2,k%2]);axes.append(a);a.set_title(chr(97+k),loc='left',fontweight='bold',fontsize=7,pad=2)
        t=z[z.model==model];y=np.arange(len(t));a.errorbar(t.estimate,y,xerr=np.vstack([t.estimate-t.conf_low,t.conf_high-t.estimate]),fmt='o',ms=2.5,color=TEAL,lw=.8,capsize=1.5)
        a.set_yticks(y,[names.get(x,x) for x in t.term]);a.invert_yaxis();a.axvline(0,color=GRAY,lw=.5,ls='--');a.set(xlabel='Coefficient with reported 95% CI',ylabel=model.replace('paper_','').replace('_',' '))
    e.save(fig,15,'Reported_regressions','Reported regression coefficients and model-specific uncertainty','Each panel represents one stored model, identified on the vertical axis. Points are reported coefficients and bars are the corresponding 95% confidence intervals. Paper and country-field models retain their own units, weighting, fixed effects and covariance estimators. Exact sample counts, complete-case definitions and covariance choices are exported. AI results describe the classified subset.',[('all_models',z)])


def fig16(ctx,e):
    labels=ctx.table('S33_full_label_diagnostics');coverage=ctx.table('S34_full_coverage_diagnostics')
    try:types=ctx.table('S35_source_type_diagnostics')
    except MissingInput:types=pd.DataFrame()
    fig,ax=new(3 if len(types) else 2,2,178 if len(types) else 150)
    for a,metric in zip(ax[:2],['known_fields','known_countries']):
        z=labels[labels.metric==metric].copy();z['category']=np.minimum(z.value.astype(int),5);z=z.groupby('category',as_index=False).n.sum()
        a.bar(np.arange(len(z)),z.n/max(1,z.n.sum()),color=TEAL);a.set_xticks(np.arange(len(z)),['5+' if v==5 else str(int(v)) for v in z.category]);a.set(xlabel=metric.replace('_',' '),ylabel='Fraction of dated core',ylim=(0,1))
    t=coverage.pivot(index='coverage_bin',columns='year',values='fraction_within_year');t.index=[f'{int(k)/10:.1f}-{(int(k)+1)/10:.1f}' for k in t.index]
    heat(fig,ax[2],t,'Fraction of P*',vmin=0,vmax=1);ax[2].set(xlabel='Citing year',ylabel='Observed core coverage bin')
    for k,(b,z) in enumerate(coverage.groupby('coverage_bin')):ax[3].plot(z.year,z.mean_q,color=plt.get_cmap('viridis')(k/10),label=f'{int(b)/10:.1f}-{(int(b)+1)/10:.1f}')
    ax[3].set(xlabel='Citing year',ylabel='Full-P* mean q by coverage');key(ax[3],fontsize=5.2,ncol=2)
    if len(types):
        for k,(kind,z) in enumerate(types.groupby('kind_label')):
            ax[4].plot(z.year,z.share,label=kind,color=PALETTE[k]);ax[5].plot(z.year,z.mean_q,label=kind,color=PALETTE[k])
        ax[4].set(xlabel='Citing year',ylabel='Full-P* source-type share',ylim=(0,1));ax[5].set(xlabel='Citing year',ylabel='Full-P* mean q by source type');key(ax[4])
    e.save(fig,16,'Full_population_diagnostics','Optional streaming population diagnostics','a-b, Fields and country membership counts for all dated core papers, including papers outside P*; counts of five or more are pooled for display, with full counts in the table. c-d, Reference-classification coverage and mean co-use within P*. The optional e-f panels show source-type shares and means. Population diagnostics stream the existing assembly table. Source kinds use exact cached paper-ID alignment; reference coverage describes the observed core graph.',[('a_b',labels),('c_d',coverage)]+([('e_f',types)] if len(types) else []))


FIGURES=[fig01,fig02,fig03,fig04,fig05,fig06,fig07,fig08,fig09,fig10,fig11,fig12,fig13,fig14,fig15,fig16]


def render_all(ctx,book=True):
    ctx.validate_table_reuse()
    for folder in ['figures','source_data']:
        for path in ctx.d[folder].glob('Supplementary_Fig_S[0-9][0-9]_*'):
            if path.is_file():path.unlink()
    exporter=Exporter(ctx,book=book)
    try:
        for fn in FIGURES:
            try:fn(ctx,exporter)
            except MissingInput as exc:ctx.event('figures',fn.__name__,'skipped',exc)
            else:ctx.event('figures',fn.__name__,'complete')
        exporter.tables()
    finally:
        exporter.close();ctx.finish();plt.close('all')
    return ctx
