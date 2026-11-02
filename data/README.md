# External research inputs

The default `config.json` reads bibliographic inputs from this directory. Large input files are excluded from version control.

Required relative locations:

```text
core/papers.parquet
extended/country_members.parquet
stage/fields/data.parquet
stage/edges/data.parquet
```

Optional relative locations:

```text
metrics/embedding_paper.parquet
extended/AI_papers.parquet
```

See the input-schema table in the repository [README](../README.md). Source files are read-only. Analysis outputs are written outside this directory, to `results/` by default.
