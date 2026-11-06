from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import numpy as np
import pandas as pd


class MissingInput(FileNotFoundError):
    pass


def divide(a,b):
    a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
    return np.divide(a,b,out=np.full(a.shape,np.nan),where=b>0)


def require(frame,columns,source):
    absent=set(columns)-set(frame)
    if absent:raise ValueError(f'{source}: missing columns {sorted(absent)}')
    return frame


class Context:
    def __init__(self,config='config.json',results_dir=None,output_dir=None,
                 sample_file=None,min_exposure=None,top_countries=24,synthetic=False):
        self.config_path=Path(config).expanduser().resolve()
        self.c=json.loads(self.config_path.read_text(encoding='utf-8-sig'))
        result=Path(results_dir or self.c.get('output_dir','results')).expanduser()
        self.results=(result if result.is_absolute() else self.config_path.parent/result).resolve()
        self.out=Path(output_dir).expanduser().resolve() if output_dir else self.results/'si'
        if self.out==self.results:raise ValueError('SI output_dir must be a separate directory, not the main results root.')
        self.d={k:self.out/k for k in ('tables','figures','source_data','logs')}
        for p in self.d.values():p.mkdir(parents=True,exist_ok=True)
        self.sample_path=Path(sample_file).expanduser().resolve() if sample_file else self.results/'data/plot_sample.parquet'
        self.sample_scope=('User-specified sample; inclusion design must be verified.' if sample_file else
                           'Stage-05 uniform deterministic bottom-k hash sample of P*; not a stratified sample.')
        self.minimum=float(min_exposure if min_exposure is not None else self.c.get('min_cell_mass',25))
        self.top=int(top_countries)
        if self.minimum<0 or self.top<1:raise ValueError('min_exposure must be nonnegative and top_countries positive.')
        self.synthetic=bool(synthetic or self.c.get('synthetic',False))
        self.start=int(self.c.get('trend_start',1990));self.end=int(self.c.get('end_year',2024))
        self.current=int(self.c.get('comparison_years',[2020,2024])[0])
        self.base=int(self.c.get('baseline_years',[2010,2014])[0])
        old_path=self.out/'SI_manifest.json'
        self.previous=json.loads(old_path.read_text(encoding='utf-8')) if old_path.is_file() else None
        if self.previous and Path(self.previous['results_dir'])!=self.results:
            raise ValueError('This SI output directory belongs to a different results root; choose another output_dir.')
        self.cache={};self.inventory={r['name']:r for r in self.previous.get('tables',[])} if self.previous else {}
        self.inputs={};self.events=[r for r in self.previous.get('events',[]) if r['stage']=='tables'] if self.previous else []
        self.figures=[];self.book=None;self.tables_rebuilt=False
        self.countries=self.raw('countries',['country','country_code'])
        self.fields=self.raw('fields',['field','field_label'])
        for table,key in ((self.countries,'country'),(self.fields,'field')):
            if not np.array_equal(np.sort(table[key].to_numpy()),np.arange(len(table))):
                raise ValueError(f'{key} codes must be unique contiguous zero-based integers.')
        self.country=dict(zip(self.countries.country,self.countries.country_code))
        self.field=dict(zip(self.fields.field,self.fields.field_label))

    def track(self,path):
        path=Path(path);stat=path.stat()
        row={'path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        if stat.st_size<=25*1024**2:row['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        self.inputs[str(path)]=row

    def raw(self,name,columns=(),folder='tables'):
        key=(folder,name)
        if key not in self.cache:
            path=self.results/folder/(name+'.csv')
            if not path.is_file():raise MissingInput(str(path))
            self.track(path);self.cache[key]=pd.read_csv(path,encoding='utf-8-sig')
        return require(self.cache[key].copy(),columns,str(key))

    def sample(self):
        key=('sample',str(self.sample_path))
        if key not in self.cache:
            path=self.sample_path
            if not path.is_file():raise MissingInput(str(path))
            self.track(path)
            frame=pd.read_csv(path) if path.suffix.lower()=='.csv' else pd.read_parquet(path)
            require(frame,['year','joint_cross','geo_cross','field_cross','joint_refs','reference_coverage',
                           'known_fields','known_countries','team_size'],'plot sample')
            if 'paper_id' in frame and not frame.paper_id.is_unique:raise ValueError('Plot sample contains duplicate paper IDs.')
            self.cache[key]=frame.loc[frame.year.between(self.start,self.end)].copy()
        return self.cache[key].copy()

    def networks(self):
        key=('network','windows')
        if key not in self.cache:
            path=self.results/'cache/window_networks.npz'
            if not path.is_file():raise MissingInput(str(path))
            self.track(path)
            with np.load(path,allow_pickle=False) as data:
                net=data['network'].copy();periods=data['periods'].copy()
            n=len(self.countries)
            if net.ndim!=4 or net.shape[0]!=len(periods) or net.shape[1]<3 or net.shape[2:]!=(n,n):
                raise ValueError('Unexpected window_networks shape or country order.')
            if np.any(~np.isfinite(net)) or np.any(net<0):raise ValueError('Network weights must be finite and nonnegative.')
            self.cache[key]=(net,periods)
        return self.cache[key]

    def write(self,name,frame,title,scope):
        path=self.d['tables']/(name+'.csv')
        frame.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.12g')
        self.inventory[name]={'name':name,'title':title,'scope':scope,'rows':len(frame),
                              'columns':';'.join(map(str,frame.columns)),'path':str(path)}
        self.cache[('si',name)]=frame.copy()
        return frame

    def table(self,name,columns=()):
        key=('si',name)
        if key not in self.cache:
            path=self.d['tables']/(name+'.csv')
            if not path.is_file():raise MissingInput('SI table '+str(path))
            self.cache[key]=pd.read_csv(path,encoding='utf-8-sig')
        return require(self.cache[key].copy(),columns,name)

    def source(self,figure,panel,frame):
        path=self.d['source_data']/f'{figure}_{panel}.csv'
        frame.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.12g')
        return str(path)

    def event(self,stage,item,status,reason=''):
        self.events.append(dict(stage=stage,item=item,status=status,reason=str(reason)))
        print(f'[{stage}] {item}: {status}'+(f' ({reason})' if reason else ''),flush=True)

    def begin_tables(self):
        # Remove only this add-on's managed derived tables, preventing stale
        # optional results from leaking into a later run with missing inputs.
        for path in self.d['tables'].glob('S[0-9][0-9]_*.csv'):path.unlink()
        self.cache={};self.inputs={}
        self.countries=self.raw('countries',['country','country_code']);self.fields=self.raw('fields',['field','field_label'])
        for table,col in [(self.countries,'country'),(self.fields,'field')]:
            if not np.array_equal(np.sort(table[col].to_numpy()),np.arange(len(table))):raise ValueError('Code mapping changed to a non-contiguous index.')
        self.country=dict(zip(self.countries.country,self.countries.country_code));self.field=dict(zip(self.fields.field,self.fields.field_label))
        self.inventory={};self.events=[];self.tables_rebuilt=True

    def validate_table_reuse(self):
        if self.tables_rebuilt:return
        if not self.previous:raise MissingInput('No previous SI table manifest; run the tables or all stage first.')
        expected=self.definition()
        if self.previous.get('table_definition')!=expected:
            raise ValueError('SI thresholds/windows/sample settings changed; rebuild SI tables before figures.')
        current=self.code_hashes()
        for name in ['core.py','analysis.py']:
            old=self.previous.get('code_sha256',{}).get(name)
            if old and old!=current.get(name):raise ValueError('SI analysis implementation changed; rebuild SI tables before figures.')
        for row in self.previous.get('inputs',[]):
            path=Path(row['path'])
            if not path.is_file() or path.stat().st_size!=row['bytes'] or path.stat().st_mtime_ns!=row['mtime_ns']:
                raise ValueError(f'SI input changed or disappeared: {path}. Rebuild SI tables.')
        self.inputs.update({r['path']:r for r in self.previous.get('inputs',[])})

    def definition(self):
        return dict(start=self.start,end=self.end,baseline=self.base,comparison=self.current,
                    minimum=self.minimum,sample_path=str(self.sample_path),synthetic=self.synthetic)

    def code_hashes(self):
        return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}

    def finish(self):
        import scipy
        import matplotlib
        status={}
        for name in ['metadata','references','metrics','networks','descriptions','semantic','interpretation','reference_definitions']:
            path=self.results/'status'/(name+'.json')
            if path.is_file():status[name]=json.loads(path.read_text(encoding='utf-8-sig'))
        meta={'created_utc':datetime.now(timezone.utc).isoformat(),'synthetic':self.synthetic,
              'results_dir':str(self.results),'minimum_exposure_each_role':self.minimum,
              'top_country_selection':'Recent assembly exposure E_asm; never highest role score.',
              'sample_scope':self.sample_scope,'table_definition':self.definition(),'inputs':list(self.inputs.values()),
              'software':{'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
              'code_sha256':self.code_hashes(),'main_pipeline_status':status,
              'study_config':{k:self.c.get(k) for k in ['trend_start','end_year','field_mode','field_weighting','field_level','field_ids','baseline_years','comparison_years','fixed_support_start','min_cell_mass','min_null_pool','seed','r_export_fraction']},
              'tables':list(self.inventory.values()),'figures':self.figures,'events':self.events,
              'interpretation':'Descriptive accounting and conditional association based on the supplied main-analysis results.'}
        (self.out/'SI_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        pd.DataFrame(self.events).to_csv(self.out/'SI_run_status.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(self.inventory.values()).to_csv(self.out/'SI_table_inventory.csv',index=False,encoding='utf-8-sig')
        lines=[]
        for f in self.figures:lines.extend([f['number']+' | '+f['title'],f['caption'],''])
        (self.out/'SI_figure_legends.txt').write_text('\n'.join(lines),encoding='utf-8')
        return meta
