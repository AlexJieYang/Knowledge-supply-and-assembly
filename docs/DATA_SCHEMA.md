# Research table schema

Paths below are relative to `research_dir`.

| Input key | Default location within `research_dir` | Expected columns |
| --- | --- | --- |
| `papers` | `core/papers.parquet` | Required: `paper_id`, `year`, `primary_field_id`. Recommended: `source_id`, `team_size`, `is_conference`. |
| `countries` | `extended/country_members.parquet` | `paper_id`, `entity_id`, `weight`; country entities use ISO-style codes. |
| `fields` | `stage/fields/data.parquet` | Long-form `paper_id`, `field_id`; optional `level` and `weight`. Configurable aliases are supported. |
| `edges` | `stage/edges/data.parquet` | `citing_id`, `cited_id`. |
| `semantic` | `metrics/embedding_paper.parquet` | Optional: `paper_id`, `emb_ref_pair_mean_distance`, `emb_ref_coverage`. |
| `ai` | `extended/AI_papers.parquet` | Optional: `paper_id`, `AIProb`. |

