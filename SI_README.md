# Supplementary analyses

This workflow reads completed main-analysis results and produces supplementary numerical tables, multi-panel figures, source data and a combined PDF. It reuses stages 01-05 and imports R estimates when stage 06 has been completed.

## Run

```bash
python run_si.py
python run_si.py --stage tables
python run_si.py --stage figures
```

Alternatively, open `SI/08_Python_Supplementary_Information.ipynb` with the project Python kernel. Run cells in order.

Directory and presentation options:

```bash
python run_si.py --results-dir results --output-dir results/si
python run_si.py --min-exposure 50 --top-countries 20 --output-dir results/si_min50
python run_si.py --no-book
```

`--results-dir` is resolved against the configuration directory; explicitly supplied `--output-dir` and `--sample-file` values are resolved against the current working directory. The default output is `results/si/`.

Figure-only reuse checks the previous table definitions and input signatures. Rebuild tables after changing inputs, thresholds, windows or analysis code. The workflow manages its own derived files without replacing main-analysis outputs.

## Inputs

The workflow reads the mapping tables `results/tables/countries.csv` and `fields.csv`. Individual sections read relevant main summary tables, `results/data/plot_sample.parquet`, `results/cache/window_networks.npz` and, when available, `results/r/coefficients.csv` and `model_scope.csv`.

The default sample is the deterministic uniform bottom-k sample from stage 05. `--sample-file` accepts an alternative CSV or Parquet file, whose inclusion design is labelled separately.

## Analytical figure sequence

Workflow identifiers below differ from the article's presentation order.

| Workflow figure | Contents | Main inputs |
| --- | --- | --- |
| S01 | Population counts and coverage | `population_year` |
| S02 | Sample classifications and reference support | `plot_sample` |
| S03 | Reference definitions, fixed support and endpoints | `definition_sensitivity`, `fixed_support_mean`, `period_endpoint_comparison` |
| S04 | Observed and benchmark intensities | `network_annual` |
| S05 | Co-use by receiving field | `field_year_means` |
| S06 | Full-population cumulative distributions | `histograms` |
| S07 | National source/assembler profiles | `role_country_field_period` |
| S08 | Country-field role atlas | `role_country_field_period` |
| S09 | Credit and exposure concentration | `role_concentration` |
| S10 | Community agreement and optimisation stability | `community_runs`, `community_stability` |
| S11 | Domestic links and directed positions | `window_networks`, baseline F partition |
| S12 | Contributions to temporal change | `role_country_field_period` |
| S13 | Source cohorts and reference ages | `reference_cohorts` |
| S14 | Semantic breadth at available reference scopes | `joint_semantic`, plotting sample |
| S15 | R coefficients and confidence intervals | R estimates and model scope |
| S16 | Optional full-population coverage diagnostics | `assembly_papers`, aligned caches |

`publication_redraw.py` separately reproduces the four-model coefficient display as **Supplementary Figure 7**, using the included `publication_tables/Regression_Coefficients.csv`.

## Outputs

| Output | Contents |
| --- | --- |
| `figures/` | Individual PDF, SVG and 450-dpi PNG figures. |
| `tables/` | Numerical summaries, full profiles, distributions and model scope. |
| `source_data/` | Numerical CSV data used by each figure or panel group. |
| `Supplementary_Information.pdf` | Combined figures and readable tables. |
| `SI_figure_legends.txt` | English titles and legends. |
| `SI_manifest.json` | Input signatures, definitions, software versions and inventory. |
| `SI_run_status.csv` | Completed and skipped sections. |
| `SI_table_inventory.csv` | Table definitions and row counts. |

The manifest records the generated tables, figures and plotting sources.

## Measurement conventions

- Observed and benchmark means use identical supported papers; empirical-context means have a separate full-population definition.
- Role gaps require sufficient exposure in both roles; zero-exposure rates remain undefined.
- Distributions count the zero atom once. Approximate quantiles use upper-bin edges.
- Incoming and outgoing participation follow source-to-receiver direction and use international links.
- Temporal decomposition is an exact identity; country and field totals are alternative marginals.
- Semantic measures retain their reference-scope labels. R intervals retain their original covariance definitions.

See [STUDY_DESIGN.md](STUDY_DESIGN.md) for equations and [README.md](README.md) for data schemas and installation.

## Functional examples

```bash
python -m pytest si_tests -q
python demo_si.py
```

Examples use synthetic inputs and labelled outputs in a separate directory, without reading the full empirical corpus.
