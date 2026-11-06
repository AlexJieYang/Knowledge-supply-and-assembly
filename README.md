# Knowledge supply and assembly in global science

Research code for **Knowledge supply and assembly in the global division of scientific labour**.

The project measures how publications combine knowledge across institutional countries and broad scientific fields. It allocates the same contextual reference credit to cited sources and to the papers assembling those sources, then compares role profiles, national networks, community boundaries, concentration and temporal change.

The repository contains the Python and R pipeline, clean Jupyter notebooks, plotting assets, tests and a small set of aggregated publication tables.

## Two ways to use this repository

| Workflow | Inputs | Entry point |
| --- | --- | --- |
| Redraw the two publication figures | Included `publication_tables/` | `python publication_redraw.py` |
| Recompute the complete analysis | Configured research directory | `python run_python.py` |

The first workflow redraws main Figure 1 and Supplementary Figure 7 from existing numerical results. It retains the four reported model specifications and their original coefficients and confidence intervals. It does not re-estimate the models.

## Installation

Use **64-bit Python 3.11 or 3.12** with the dependency constraints supplied in this repository. The core version pins are in `requirements-tested.txt`; supported ranges and additional runtime, notebook and testing dependencies are in `requirements.txt`.

From the repository root:

```bash
python -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Or activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-tested.txt
```

For the Python notebooks, register the project kernel:

```bash
python setup_kernel.py
```

This setup script installs the supplied dependency set and registers **Python - Global Knowledge Assembly**, including thread limits applied before numerical libraries are imported. Run it inside the virtual environment. A Jupyter-compatible frontend is required to open notebooks.

## Redraw figures from the included tables

```bash
python publication_redraw.py
```

Outputs are written to `publication_output/figures/` in PDF, SVG and PNG formats. The script also copies the numerical tables used for the redraw to `publication_output/tables/`.

Optional relative-directory overrides:

```bash
python publication_redraw.py --tables-dir publication_tables --output-dir publication_output
```

## Configure the full analysis

The default configuration uses repository-relative directories:

```json
{
  "research_dir": "data",
  "output_dir": "results"
}
```

These are two entries from `config.json`, rather than a replacement for the full configuration. Relative input and output roots are resolved against the directory containing the configuration file. Input files remain read-only; the output directory must be outside the input root.

Set `research_dir` and `paths` in your configuration. A private `config.local.json` is excluded from Git by default. Table schemas and column conventions are documented in [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md).

Country weights must sum to one within each classified publication, apart from small rounding differences. Field settings support multiple distinct labels at a specified level; the empirical classification used in the article contains one broad field label per classified record. The preset 19 field identifiers refer to broad Fields, rather than OpenAlex Topics.

Set ambiguous field columns explicitly in `field_columns`. `field_mode="multilabel"` reads the field-membership input. `field_mode="primary"` is an explicit alternative definition and should use a separate output directory. Optional AI scores never define the main analytical population.

## Run the Python pipeline

```bash
python run_python.py
```

Each stage runs in a fresh process. The command reuses compatible completed stages and does not run the R models automatically.

To use a private configuration:

```bash
python run_python.py --config config.local.json
```

To recompute one stage:

```bash
python run_python.py --stage metrics --force
```

If an upstream definition or input changes, rerun the affected stage and its downstream stages. To redraw all five main figures from a complete existing result directory:

```bash
python redraw_figures.py --results-dir results
```

### Notebook sequence

Run notebooks in order. Close each large-data kernel after finishing a stage so that multiple notebooks do not retain large indexes simultaneously.

| Notebook | Kernel | Purpose |
| --- | --- | --- |
| `00_Python_Start.ipynb` | Python | Configuration, resource settings and input preflight. |
| `01_Python_Metadata.ipynb` | Python | Core metadata and fractional country/field memberships. |
| `02_Python_Reference_Graph.ipynb` | Python | Compact citation graph, duplicate removal and date checks. |
| `03_Python_Assembly_Metrics.ipynb` | Python | Geographic, disciplinary and joint reference-context measures. |
| `04_Python_Networks_References.ipynb` | Python | C/F/H networks, composition benchmark and community comparisons. |
| `05_Python_Descriptions.ipynb` | Python | Role accounts, RCA, distributions, decomposition and model inputs. |
| `06_R_Conditional_Associations.ipynb` | R | Conditional association models. |
| `07_Python_Publication_Figures.ipynb` | Python | Five main figures and panel-to-source mappings. |
| `SI/08_Python_Supplementary_Information.ipynb` | Python | Supplementary tables, figures and source data. |

Python notebooks discover the repository root from the working directory or its parents. Set the `GKA_CONFIG` environment variable to select an alternative configuration for these notebooks. Rebuild the clean notebook wrappers with `python build_notebooks.py`.

## R models

Use **R 4.3 or later**. From an R session with the repository as its working directory:

```r
source("setup_R.R")
source("R/analysis.R")
models <- run_associations()
```

Alternatively, select **R - Global Knowledge Assembly** for notebook 06 after running `setup_R.R`, or run `Rscript run_R.R --config config.local.json`. Python and R accept `GKA_CONFIG` as the default configuration selector. With no selector, they read the repository's `config.json`. Relative input and output directories are resolved against the selected configuration file's directory. Use the same configuration for the Python and R stages.

The paper models include field-by-year and publication-source fixed effects, with publication-source-clustered standard errors. The country-field-year model uses assembler-exposure weights, country-field and field-year fixed effects, and covariance clustered by country and year. The optional AI-profile branch uses valid input scores. See [STUDY_DESIGN.md](STUDY_DESIGN.md) for definitions and interpretation.

## Supplementary analysis

After the main summaries have been produced:

```bash
python run_si.py
```

To build tables and figures separately:

```bash
python run_si.py --stage tables
python run_si.py --stage figures
```

See [SI_README.md](SI_README.md) for inputs, optional diagnostics, figure numbering and output definitions. Internal SI figure numbers follow the analytical workflow; the article uses a different presentation order, including workflow S15 as Supplementary Figure 7.

## Output structure

| Directory | Contents |
| --- | --- |
| `results/data/` | `metadata.parquet`, `assembly_papers.parquet`, `r_papers.parquet`, `plot_sample.parquet`. |
| `results/cache/` | Compact binary indexes, reference metrics and network arrays. |
| `results/tables/` | Aggregated analysis and plotting tables. |
| `results/figures/` | Main figures, captions and source mappings. |
| `results/r/` | Model estimates, confidence intervals, sample definitions and R session information. |
| `results/status/` | Current stage-completion records and reuse signatures. |
| `results/si/` | Supplementary tables, figures, source data and combined PDF. |

Generated data, caches, outputs, virtual environments and notebook checkpoints are excluded by `.gitignore`. Notebooks are distributed without execution outputs or local execution metadata.

## Resource settings

The full-corpus workflow is intended for a large-memory machine. The default guard limits resident memory to the smaller of 384 GiB and 80% of visible physical memory; it is a guard rather than an allocation guarantee. The default uses eight Numba threads, two Arrow threads and single-threaded BLAS.

Reduce `batch_rows` and `metric_batch_papers`, or use `lookup="sorted"`, when memory is constrained. Main aggregates use the full eligible population. `plot_sample_n` controls the deterministic plotting sample, while `r_export_fraction` controls the model-export sample separately.

## Tests and synthetic examples

```bash
python -m unittest discover -s tests -p test_paths.py -v
python -m pytest tests si_tests -q
python run_demo.py --n 900
python run_demo.py --n 6000 --figures --joint-semantic
python demo_si.py
```

The path-resolution checks use only Python's standard library. The full test suite uses the installed project dependencies. Synthetic examples use separate directories and labelled synthetic outputs. They exercise the implementation and do not reproduce the empirical findings. R model estimation uses the configured R environment.

## Repository contents

- `assembly/`: data preparation, reference metrics, networks, summaries and main figures.
- `si_support/`: supplementary analyses and plotting.
- `tests/`, `si_tests/`: numerical, I/O and synthetic pipeline tests.
- `R/`: model estimation.
- `publication_tables/`: aggregate data for the two publication figures.
- `assets/`: map geometry and its source notice.

## Citation and licensing

When citing the accompanying study, use its final published bibliographic record when available. The research code is available under the [MIT License](LICENSE). External bibliographic data and derived inputs retain their source licenses and access conditions. The bundled map data retain their separate source and public-domain notice in [assets/MAP_LICENSE.txt](assets/MAP_LICENSE.txt).

## Static source checks

```bash
python tools/check_sources.py
```

GitHub Actions checks Python syntax, JSON and notebook structure, and parses R scripts and R notebook cells with R 4.4. Notebooks contain no saved execution output or execution metadata. Source checks parse code without evaluating analysis stages.
