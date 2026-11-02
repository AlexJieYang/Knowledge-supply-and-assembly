"""Rebuild English notebooks without saved outputs or local execution metadata."""
from pathlib import Path
import hashlib
import json


ROOT = Path(__file__).resolve().parent
HEADER = """# Select the project Python kernel before running this cell.
from pathlib import Path
import os, sys, json
candidates = [Path.cwd(), *Path.cwd().parents]
PROJECT = next((p for p in candidates if (p / 'assembly' / 'runtime.py').is_file()), None)
if PROJECT is None:
    raise FileNotFoundError('Open this notebook from the repository root or a subdirectory.')
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
CONFIG_PATH = Path(os.environ.get('GKA_CONFIG', str(PROJECT / 'config.json'))).resolve()
settings = json.loads(CONFIG_PATH.read_text(encoding='utf-8-sig'))
os.environ['ASSEMBLY_THREADS'] = str(settings['threads'])
import bootstrap
bootstrap.configure(settings['threads'])
from assembly.runtime import config, run, dirs
C = config(CONFIG_PATH)
D = dirs(C)
print('Project:', PROJECT)
print('Python:', sys.executable)
print('Output:', D['root'])
"""


def make(filename, title, body, cells, language='python'):
    notebook = {'cells': [], 'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5}
    entries = [('markdown', '# ' + title + '\n\n' + body), *cells]
    for index, (cell_type, source) in enumerate(entries):
        cell = {'cell_type': cell_type, 'metadata': {},
                'id': hashlib.sha256(f'{filename}:{index}:{source}'.encode()).hexdigest()[:12],
                'source': source.splitlines(keepends=True)}
        if cell_type == 'code':
            cell.update(execution_count=None, outputs=[])
        notebook['cells'].append(cell)
    notebook['metadata']['kernelspec'] = {
        'display_name': ('Python' if language == 'python' else 'R') + ' - Global Knowledge Assembly',
        'language': language,
        'name': 'global-knowledge-python' if language == 'python' else 'global-knowledge-r'}
    notebook['metadata']['language_info'] = {'name': language}
    destination = ROOT / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def main():
    make('00_Python_Start.ipynb', 'Knowledge supply and assembly in global science',
         'Configure external inputs, inspect their schemas and run stages in order. '
         'The primary population contains focal papers from 1990-2024 with country and field '
         'information and at least two jointly classified references. Historical sources remain eligible. '
         'See README.md for installation, resource settings and the Python/R kernel sequence.', [
             ('code', HEADER),
             ('code', 'print(json.dumps(settings, indent=2))'),
             ('markdown', 'Edit config.json or set GKA_CONFIG before creating the configuration. '
              'Relative directories are resolved against the configuration file. '
              'Select explicit field columns when automatic detection is ambiguous.'),
             ('code', 'from assembly.metadata import preflight\ncatalog = preflight(C)\ndisplay(catalog)'),
             ('markdown', 'Run 01-05, use the R kernel for 06, then generate figures in 07. '
              'Close large-data kernels between stages. `python run_python.py` runs Python '
              'stages in separate processes.')])

    stages = [
        ('01_Python_Metadata.ipynb', '01 Core metadata and fractional attribution', 'metadata',
         'Build core metadata and compact country/field memberships. Validate country weights and '
         'use the configured field weighting. Optional attributes do not define the primary population.',
         "population = json.loads((D['status'] / 'population.json').read_text(encoding='utf-8'))\n"
         "print(json.dumps(population, indent=2))",
         'metadata.parquet is the consolidated metadata output. Unknown publication years have '
         'a separate status. Upstream files remain read-only.'),
        ('02_Python_Reference_Graph.ipynb', '02 Compact core citation graph', 'references',
         'Scan edges twice to build a shared CSR index. Deduplicate references within citing papers, '
         'remove self-links and invalid dates, and retain same-year citations and historical sources.',
         "summary = json.loads((D['status'] / 'references.json').read_text(encoding='utf-8'))['result']\n"
         "print(json.dumps(summary, indent=2))",
         'ref_ptr, ref_count and ref_index form one aligned index. Keep these caches together. '
         'The sorted lookup option reduces hash-index memory use.'),
        ('03_Python_Assembly_Metrics.ipynb', '03 Reference contexts and contextual credit', 'metrics',
         'Calculate r(j,k)=(1-a_j.a_k)(1-t_j.t_k), source-specific sigma, paper mean q and '
         'source credit b=sigma/m using aggregate moments and explicit self-pair subtraction.',
         "import pandas as pd\ndisplay(pd.read_csv(D['tables'] / 'population_year.csv').tail(8))\n"
         "print(json.loads((D['status'] / 'metrics.json').read_text(encoding='utf-8'))['result'])",
         'Geographic, disciplinary and joint measures use identical usable references. Fewer than '
         'two usable references give an undefined measure. This stage also calculates primary-field '
         'sensitivity and conditional completion bounds.'),
        ('04_Python_Networks_References.ipynb', '04 Networks and composition benchmark', 'networks',
         'Construct C/F/H on the same P*. Directed links run from sources to receivers. '
         'The benchmark retains citing year, receiving field, source type and age while '
         'excluding all focal-paper slots.',
         "import pandas as pd\ndisplay(pd.read_csv(D['tables'] / 'network_annual.csv').tail(8))\n"
         "display(pd.read_csv(D['tables'] / 'community_similarity.csv').tail(9))\n"
         "display(pd.read_csv(D['tables'] / 'conservation_checks.csv').tail(9))",
         'Checks enforce sum(F)=N, sum(H)=sum(q), H<=F and matched role credit. Compare Leiden '
         'partitions across resolutions and repeats. Alluvial widths show common F exposure; '
         'temporal participation uses the baseline F partition.'),
        ('05_Python_Descriptions.ipynb', '05 Roles distributions and temporal change', 'descriptions',
         'Summarise roles, RCA, concentration, distributions, cohorts, reference support and '
         'symmetric decomposition. Primary summaries use all P*; plotting records use '
         'deterministic uniform bottom-k sampling.',
         "import pandas as pd\ndisplay(pd.read_csv(D['tables'] / 'sample_scope.csv'))\n"
         "display(pd.read_csv(D['tables'] / 'change_accounting.csv'))\n"
         "display(pd.read_csv(D['tables'] / 'role_correlations.csv').tail(6))",
         'Fixed-support estimates begin at fixed_support_start and use baseline weights. '
         'Endpoint summaries retain 2020-2023 and 2020-2024 separately. The R export leaves '
         'model-specific complete-case selection to the R stage.')]
    for filename, title, stage, body, view, note in stages:
        make(filename, title, body, [('code', HEADER), ('code', f"run(C, '{stage}')"),
                                    ('code', view), ('markdown', note)])

    make('06_R_Conditional_Associations.ipynb', '06 Conditional association models in R',
         'Use the R project kernel after setup_R.R. Paper models include field-year and '
         'publication-source fixed effects. Aggregate models use assembler-exposure weights. '
         'The original optional AI-profile specification remains available.', [
             ('code', "candidates <- c(Sys.getenv('GKA_PROJECT', unset = getwd()), getwd(), dirname(getwd()))\n"
              "valid <- vapply(candidates, function(p) file.exists(file.path(p, 'R', 'analysis.R')), logical(1))\n"
              "if (!any(valid)) stop('Set the working directory to the repository root.')\n"
              "PROJECT <- normalizePath(candidates[which(valid)[1]], winslash = '/', mustWork = TRUE)\n"
              "source(file.path(PROJECT, 'R', 'analysis.R'), encoding = 'UTF-8')\n"
              "models <- run_associations(PROJECT)"),
             ('markdown', 'results/r contains coefficients, intervals, model scope, tables, '
              'compact model objects and R session information. Main descriptive figures '
              'can be produced independently of the R stage.')], language='R')

    make('07_Python_Publication_Figures.ipynb', '07 Main publication figures',
         'Generate five main figures from completed results: role maps, aligned networks, '
         'distributions, role profiles and temporal comparisons. Export PDF, SVG and 450-dpi PNG '
         'with captions and panel-to-source mappings.', [
             ('code', HEADER), ('code', "run(C, 'figures')"),
             ('code', "from IPython.display import Image, display\n"
              "for image in sorted(D['figures'].glob('Fig*.png')):\n"
              "    print(image.name)\n    display(Image(filename=str(image), width=1000))"),
             ('markdown', 'Unsupported cells remain masked and response surfaces connect observed '
              'bin means. Use publication_redraw.py for the two revised figures based on the '
              'included aggregate tables.')])

    si_header = """from pathlib import Path
import os, sys
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
candidates = [Path.cwd(), *Path.cwd().parents]
PROJECT = next((p for p in candidates if (p / 'si_support' / 'core.py').is_file()), None)
if PROJECT is None:
    raise FileNotFoundError('Open this notebook from the repository root or its SI directory.')
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from si_support.core import Context, MissingInput
CONFIG_PATH = Path(os.environ.get('GKA_CONFIG', str(PROJECT / 'config.json'))).resolve()
CTX = Context(config=CONFIG_PATH)
print('Input:', CTX.results)
print('SI output:', CTX.out)
"""
    make('SI/08_Python_Supplementary_Information.ipynb', '08 Supplementary analyses',
         'Build supplementary tables, figures and source data from existing main results. '
         'Read SI_README.md for workflow numbering and optional inputs.', [
             ('code', si_header),
             ('markdown', '## 1 Build numerical tables\n\nReuse summaries, network caches and '
              'the plotting sample from stages 01-05. Existing R results are imported when available.'),
             ('code', 'from si_support.analysis import build_tables\nbuild_tables(CTX)'),
             ('markdown', '## 2 Optional full-population diagnostics\n\nDisabled by default. '
              'Stream assembly_papers.parquet; source-type diagnostics require aligned kind/ID caches.'),
             ('code', "RUN_DEEP_AUDIT = False\nif RUN_DEEP_AUDIT:\n"
              "    from si_support.analysis import deep_audit\n    try:\n"
              "        deep_audit(CTX, batch_rows=250000)\n    except MissingInput as exc:\n"
              "        CTX.event('tables', 'deep_audit', 'skipped', exc)\n    else:\n"
              "        CTX.event('tables', 'deep_audit', 'complete')"),
             ('markdown', '## 3 Render figures and the combined PDF\n\nExport figures, numerical '
              'source data, legends, the combined PDF and run status.'),
             ('code', 'from si_support.figures import render_all\nrender_all(CTX, book=True)'),
             ('markdown', '## 4 Inspect the inventory\n\nReview completed and skipped sections. '
              'S16 is skipped when full-population diagnostics are disabled.'),
             ('code', "import pandas as pd\ndisplay(pd.read_csv(CTX.out / 'SI_run_status.csv'))\n"
              "display(pd.read_csv(CTX.out / 'SI_table_inventory.csv'))\n"
              "print(CTX.out / 'Supplementary_Information.pdf')"),
             ('markdown', '## 5 Preview figures\n\nShow the first two figures to keep notebook '
              'output small. Adjust SHOW_N as needed.'),
             ('code', "from IPython.display import Image, display\nSHOW_N = 2\n"
              "for image_path in sorted(CTX.d['figures'].glob('Supplementary_Fig_*.png'))[:SHOW_N]:\n"
              "    print(image_path.name)\n    display(Image(filename=str(image_path), width=1000))")])
    print('Wrote 9 clean English notebooks: 8 Python, 1 R.')


if __name__ == '__main__':
    main()
