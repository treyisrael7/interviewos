## Retrieval Evals

This folder holds internal, developer-focused assets for retrieval evaluation.

The initial foundation is intentionally small:
- a typed schema in `schema.py`
- a JSON loader in `loader.py`
- a mode-aware runner in `runner.py`
- a metrics module in `metrics.py`
- editable datasets in `cases/`

No production routes import this package today. The goal is to keep eval inputs versioned, reviewable, and easy to expand into a full offline evaluation runner later.

### CI regression (Postgres + real retrieval)

`tests/evals/test_retrieval_eval_db_integration.py` seeds a **stable document UUID** (`ci_constants.PLATFORM_ENGINEER_JD_DOCUMENT_ID`, also listed in `cases/ci_fixture_map.json`) with chunk text aligned to `job_description_starter.json`. It mocks `embed_query` to a fixed vector (no `OPENAI_API_KEY` in CI) and asserts every case passes for **semantic**, **hybrid**, and **keyword** against the live SQL retrieval stack. This runs in the existing GitHub Actions `pytest` job.

To exercise the CLI against the same map after you have loaded that corpus into your DB (e.g. by running that test once and pausing before teardown, or by inserting equivalent rows), use `--fixture-map evals/retrieval/cases/ci_fixture_map.json`.

### Dataset format

Datasets are stored as JSON objects with this top-level shape:

```json
{
  "version": 1,
  "dataset": "job_description_starter",
  "description": "Starter internal retrieval eval cases for job-description documents.",
  "cases": [
    {
      "id": "jd-required-skills",
      "fixture_ref": "platform_engineer_jd",
      "query": "What skills and qualifications are required?",
      "expected_chunk_ids": [],
      "expected_content_substrings": ["Python", "PostgreSQL"],
      "expected_section_types": ["qualifications", "tools"],
      "expected_source_types": ["jd"],
      "top_k": 6,
      "notes": "Broad qualifications query."
    }
  ]
}
```

### Case fields

Each case supports:
- `id`: stable identifier for reporting
- `document_id`: optional concrete document UUID for ad hoc evals
- `fixture_ref`: optional fixture name for seeded/dev datasets
- `query`: retrieval query text
- `expected_chunk_ids`: optional exact chunk ids when you have stable chunk fixtures
- `expected_content_substrings`: optional loose content checks when chunk ids are not stable yet
- `expected_section_types`: optional expected JD section types such as `qualifications` or `compensation`
- `expected_source_types`: optional expected source types such as `jd`, `resume`, `company`, or `notes`
- `top_k`: retrieval depth to evaluate
- `notes`: optional human context

Validation rules:
- each case must set either `document_id` or `fixture_ref`
- each case must include at least one `expected_*` field so it is measurable
- case ids must be unique within a dataset

### Adding new cases

1. Pick the dataset file under `cases/` that best matches the flow or document family.
2. Add a new object to the `cases` array.
3. Prefer `fixture_ref` for reusable seeded corpora and `document_id` only for local ad hoc investigations.
4. Use `expected_content_substrings` first when chunk ids may change.
5. Add `expected_chunk_ids` only when the corpus is stable and deterministic.
6. Keep `notes` short and focused on why the case exists.

### Loading datasets

Use the helpers in `loader.py`:

```python
from evals.retrieval import get_builtin_dataset_path, load_eval_dataset

dataset = load_eval_dataset(get_builtin_dataset_path("job_description_starter"))
```

### Running evals

The main developer entry point is:

```bash
cd apps/api
python -m evals.retrieval.entrypoint --help
```

You can also use the repo-level Make targets:

```bash
make retrieval-eval ARGS="--dataset job_description_starter --mode hybrid --fixture-map fixture-map.json"
make retrieval-eval-compare ARGS="--dataset job_description_starter --fixture-map fixture-map.json --output-json retrieval-comparison.json"
```

Single-mode run example:

```bash
cd apps/api
python -m evals.retrieval.entrypoint --dataset job_description_starter --mode hybrid --fixture-map fixture-map.json
```

Useful CLI options:
- `--dataset` or `--dataset-path`: choose a built-in dataset or a JSON file
- `--mode`: run one retrieval mode (`semantic`, `hybrid`, `keyword`)
- `--compare`: compare multiple modes side by side
- `--modes`: override which modes to compare
- `--document-id`: only run cases for one concrete document
- `--fixture-ref`: only run cases for one fixture
- `--case-id`: only run one or more specific case ids
- `--output-json`: write the run/comparison result to disk

Supported modes:
- `hybrid`: semantic retrieval plus keyword augmentation
- `semantic`: vector retrieval only
- `keyword`: keyword retrieval only

If a dataset uses `fixture_ref`, pass a fixture map or the entry point will fail with a clear error telling you which fixture refs are missing.

### Comparing strategies

You can compare retrieval strategies for the same dataset in Python:

```python
from evals.retrieval import compare_eval_dataset, load_builtin_dataset

dataset = load_builtin_dataset("job_description_starter")
comparison = await compare_eval_dataset(
    db=db,
    dataset=dataset,
    modes=("semantic", "hybrid", "keyword"),
    fixture_resolver=my_fixture_resolver,
)
```

Comparison example:

```bash
cd apps/api
python -m evals.retrieval.entrypoint --dataset job_description_starter --compare --fixture-map fixture-map.json --output-json retrieval-comparison.json
```

Example fixture map:

```json
{
  "platform_engineer_jd": "11111111-1111-1111-1111-111111111111"
}
```

CLI behavior:
- prints a concise console summary by default
- optionally writes a full JSON report with `--output-json`
- supports custom datasets with `--dataset-path`
- supports selecting a subset of modes with `--modes semantic hybrid`
- supports filtering the run with `--document-id`, `--fixture-ref`, and repeated `--case-id`
- fails clearly when fixture mappings or filters are missing
- expands failed cases with query, expected evidence, returned chunks, metadata, scores, and failure reasons

### Metrics

The runner now returns:
- per-case expectation details
- per-case metrics
- aggregate summary metrics for the full dataset/mode run

Current first-pass metrics include:
- `hit_at_1`
- `hit_at_3`
- `hit_at_5`
- `recall_at_k`
- `mrr`
- `section_type_match_rate`
- `source_type_match_rate`

Relevance can be satisfied by:
- exact `expected_chunk_ids`
- `expected_content_substrings` appearing in returned text
- matching `expected_section_types`
- matching `expected_source_types`
