# DeepSearch

[![Parsed Surveys](https://img.shields.io/badge/Parsed%20Surveys-11.5K-1f6feb?style=flat-square)](https://huggingface.co/datasets/ShuhaoChen202401/deepsearch-data)
[![Trajectories](https://img.shields.io/badge/Trajectories-3K%2B-2da44e?style=flat-square)](https://huggingface.co/datasets/ShuhaoChen202401/deepsearch-data/tree/main/trajectories)
[![Filter Examples](https://img.shields.io/badge/Filter%20Examples-1.20M-8250df?style=flat-square)](https://huggingface.co/datasets/ShuhaoChen202401/deepsearch-data/tree/main/training)
[![Benchmark Topics](https://img.shields.io/badge/Benchmark%20Topics-300-f97316?style=flat-square)](https://huggingface.co/datasets/ShuhaoChen202401/deepsearch-data/tree/main/benchmark)
[![Checkpoints](https://img.shields.io/badge/Checkpoints-5-0ea5e9?style=flat-square)](#model-checkpoints)
[![Python](https://img.shields.io/badge/Python-3.10%2B-f7c948?style=flat-square)](./pyproject.toml)

**What Makes a Good Deep Search? Rethinking Trajectory Construction, Agent
Frameworks, and Evaluation Methodologies**

Official code for process-supervised, survey-oriented literature retrieval.
The system combines keyword search, citation expansion, a paper filter, and a
retrieval-action selector. It includes:

- a section-aware, compare-then-update trajectory constructor;
- SFT data builders for the paper filter and action selector;
- an inference agent with interchangeable retrieval backends and models;
- SFT and VERL-compatible selector-RL entry points;
- reference, section-coverage, temporal, influence, and composite metrics;
- an entirely offline example and unit tests.

The repository intentionally contains code and small synthetic examples only.
Research data and model checkpoints are released separately.

<p align="center">
  <img src="./main_figure.png" width="100%" alt="DeepSearch trajectory construction pipeline">
</p>

<p align="center"><em>Overview of the section-aware, compare-then-update trajectory construction pipeline.</em></p>

## Dataset Scale

The source pool contains 11,920 parsed expert-written surveys. After quality
control and cross-domain deduplication, the release contains **11,588
section-level survey records**, including section-to-reference mappings and
bibliographic metadata. From this pool, **3,152 retrieval trajectories** are
released for process supervision, together with approximately **1.20 million
topic-paper examples** for paper-filter training. Evaluation uses a
non-overlapping benchmark of **300 survey topics**. The code repository
contains only a small synthetic example; the released datasets and available
checkpoints are hosted on Hugging Face.

## 📦 Resources

- Code: https://github.com/Jacko-chen/deepsearch
- Dataset: https://huggingface.co/datasets/ShuhaoChen202401/deepsearch-data
- Default paper filter: https://huggingface.co/ShuhaoChen202401/deepsearch-paper-filter-sft
- Default retrieval selector: https://huggingface.co/ShuhaoChen202401/deepsearch-retrieval-selector-sft-rl

### Model checkpoints

| Checkpoint | Training strategy | Role |
|---|---|---|
| [Paper Filter SFT](https://huggingface.co/ShuhaoChen202401/deepsearch-paper-filter-sft) | SFT | Default paper filter |
| [Paper Filter RL](https://huggingface.co/ShuhaoChen202401/deepsearch-paper-filter-rl) | RL-only | Filter ablation |
| [Retrieval Selector SFT](https://huggingface.co/ShuhaoChen202401/deepsearch-retrieval-selector-sft) | SFT | Selector ablation |
| [Retrieval Selector RL](https://huggingface.co/ShuhaoChen202401/deepsearch-retrieval-selector-rl) | RL-only | Selector ablation |
| [Retrieval Selector SFT+RL](https://huggingface.co/ShuhaoChen202401/deepsearch-retrieval-selector-sft-rl) | SFT followed by RL | Default retrieval selector |

## ⚙️ Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/Jacko-chen/deepsearch.git
cd deepsearch
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install optional dependencies only when needed:

```bash
pip install -e ".[api]"    # live AMiner backend
pip install -e ".[train]"  # Qwen inference and SFT
pip install -e ".[dev]"    # development checks
```

## ⬇️ Downloading released artifacts

Transformers can load a checkpoint directly from its Hugging Face repository
ID, as shown in the inference example below. No manual model download is
required in that case; files are stored in the standard Hugging Face cache.

For local or offline use, run the following commands from the repository root
to download the dataset and the default checkpoint pair into the recommended
project directories:

```bash
pip install -U "huggingface_hub[hf_xet]"

hf download ShuhaoChen202401/deepsearch-data \
  --repo-type dataset \
  --local-dir data/deepsearch

hf download ShuhaoChen202401/deepsearch-paper-filter-sft \
  --local-dir checkpoints/paper-filter-sft

hf download ShuhaoChen202401/deepsearch-retrieval-selector-sft-rl \
  --local-dir checkpoints/retrieval-selector-sft-rl
```

The resulting layout is:

```text
deepsearch/
├── data/
│   └── deepsearch/
│       ├── surveys/
│       ├── trajectories/
│       ├── training/
│       ├── benchmark/
│       └── ablations/
└── checkpoints/
    ├── paper-filter-sft/
    └── retrieval-selector-sft-rl/
```

Use `training/paper_filter_sft/` or `training/retrieval_selector_sft/` for
supervised training data, the corresponding `*_rl/` directories for RL data,
`benchmark/` for evaluation, `trajectories/raw/` for constructed trajectories,
and `surveys/parsed_section_level/` for parsed survey records. These locations
are a recommended convention rather than hard-coded paths; pass alternative
locations explicitly to the relevant command or model constructor.

## 🚀 Five-minute offline example

No API key, external dataset, GPU, or model checkpoint is required:

```bash
deepsearch-demo --output-dir outputs/demo
```

This command constructs a supervised trajectory from
`examples/topic.json`, runs the inference agent over
`examples/corpus.json`, and writes:

```text
outputs/demo/
├── agent_result.json
├── constructed_trajectory.json
└── metrics.json
```

The example uses deterministic heuristic replacements for the learned filter
and selector. The retrieval pipeline and file interfaces are the same when
real checkpoints are supplied.

## 🔍 Pipeline

### Trajectory construction

At the first round, the constructor performs keyword search. At later rounds,
it evaluates keyword-search and citation-expansion branches, filters their
candidates, and chooses the update with higher section coverage and then
paper-level F1:

```bash
deepsearch-construct \
  --topic examples/topic.json \
  --corpus examples/corpus.json \
  --output outputs/trajectory.json \
  --max-steps 5
```

Survey references and section targets are construction-time supervision.
`DeepSearchAgent` does not access either at inference time.

### Inference with trained checkpoints

The default checkpoint pair can be loaded directly from Hugging Face:

```python
from deepsearch.agent import DeepSearchAgent
from deepsearch.backends.aminer import AMinerBackend
from deepsearch.models import TransformersActionSelector, TransformersPaperFilter

backend = AMinerBackend()  # reads AMINER_API_KEY
paper_filter = TransformersPaperFilter(
    "ShuhaoChen202401/deepsearch-paper-filter-sft",
)
selector = TransformersActionSelector(
    "ShuhaoChen202401/deepsearch-retrieval-selector-sft-rl",
)
result = DeepSearchAgent(backend, paper_filter, selector).retrieve(
    "Graph neural networks for recommender systems"
)
```

If the checkpoints were downloaded into the recommended local directories,
replace the two repository IDs with local paths:

```python
paper_filter = TransformersPaperFilter(
    "checkpoints/paper-filter-sft",
)
selector = TransformersActionSelector(
    "checkpoints/retrieval-selector-sft-rl",
)
```

Set credentials outside source code:

```bash
cp .env.example .env
export AMINER_API_KEY="your-key"
```

The [AMiner](https://www.aminer.cn/) service is external and its response
schema or access policy may change. `RetrievalBackend` can be implemented for
another paper-search provider without changing the constructor or agent.

## Preparing SFT data

### Paper-filter labels

Input is JSONL with one candidate per line. Reference papers are positive.
For candidates outside the bibliography, three independent judge decisions
are expected: unanimous positive votes create a supplementary positive,
whereas at least two negative votes create a negative. Remaining cases are
excluded. Similarity fields used for candidate organization are deliberately
removed from model inputs.

```bash
python scripts/build_filter_data.py \
  --input /path/to/filter_candidates.jsonl \
  --train-output data/filter_train.jsonl \
  --validation-output data/filter_validation.jsonl
```

### Selector trajectories

```bash
python scripts/build_selector_data.py \
  --input /path/to/trajectories.jsonl \
  --train-output data/selector_train.jsonl \
  --validation-output data/selector_validation.jsonl
```

Splits are group-based to avoid topic or trajectory leakage. The input and
output fields are summarized in the Data formats section below.

## 🧪 Training

The provided SFT trainer consumes JSONL records containing a `messages`
conversation. It masks prompt tokens and optimizes only assistant outputs.
LoRA is enabled by default.

```bash
python scripts/train_filter_sft.py \
  --model Qwen/Qwen3-8B \
  --train-file data/filter_train.jsonl \
  --validation-file data/filter_validation.jsonl \
  --output-dir checkpoints/filter

python scripts/train_selector_sft.py \
  --model Qwen/Qwen3-8B \
  --train-file data/selector_train.jsonl \
  --validation-file data/selector_validation.jsonl \
  --output-dir checkpoints/selector
```

For selector RL, install an upstream [VERL](https://github.com/volcengine/verl)
checkout and provide its path plus the prepared Parquet files:

```bash
export VERL_DIR=/path/to/verl
export MODEL_PATH=/path/to/sft-selector
export TRAIN_DATA=/path/to/train.parquet
export VAL_DATA=/path/to/validation.parquet
bash scripts/train_selector_rl.sh
```

Additional VERL overrides can be appended to the command. The custom reward
entry point is `src/deepsearch/training/verl_reward.py`. The environment
should attach final collection reward and trajectory state through
`extra_info`.

## 📊 Evaluation

```bash
deepsearch-evaluate \
  --topic examples/topic.json \
  --collection outputs/demo/agent_result.json
```

The local evaluator implements metrics that can be derived from released
metadata. LLMJudge requires separately configured model APIs; its
model-agnostic prompt is in `prompts/llm_judge.txt`.

## Repository layout

```text
src/deepsearch/
  agent.py          inference loop
  construction.py   compare-then-update trajectory construction
  filters.py        filter interface and candidate screening
  selectors.py      selector interface and output parser
  models.py         optional Transformers checkpoint adapters
  metrics.py        automatic evaluation metrics
  rewards.py        selector-RL format/action rewards
  backends/         in-memory and AMiner retrieval backends
  training/         SFT utilities and the VERL reward entry point
scripts/            data preparation and training entry points
prompts/            released prompt templates
examples/           small synthetic runnable inputs
tests/              offline unit and integration tests
```

## 🗂️ Data formats

The following schemas describe the normalized inputs used by the code. The
released dataset preserves additional provenance fields documented in its
Hugging Face Dataset Card. In particular, the raw filter-candidate schema
describes the expected input to the data builder.

### Topic

```json
{
  "topic_id": "stable-topic-id",
  "topic": "survey title or retrieval topic",
  "survey_year": 2025,
  "target_references": ["paper-id"],
  "section_targets": {"Section title": ["paper-id"]}
}
```

`target_references` and `section_targets` are used only for construction and
evaluation and are never passed to the inference agent.

### Paper

```json
{
  "id": "paper-id",
  "title": "Paper title",
  "abstract": "Abstract text",
  "year": 2024,
  "citation_count": 10,
  "venue": "Venue",
  "keywords": ["keyword"],
  "authors": ["Author"],
  "references": ["paper-id"],
  "cited_by": ["paper-id"]
}
```

Only `id` and `title` are required. Citation edges are used by the offline
backend; live backends retrieve them from their providers.

### Raw filter candidate

```json
{
  "topic_id": "topic-id",
  "topic": "survey topic",
  "in_reference": false,
  "judge_votes": ["yes", "yes", "yes"],
  "paper": {"id": "paper-id", "title": "Paper title"}
}
```

Source-reference papers are positive. Three positive votes create a
supplementary positive, at least two negative votes create a negative, and
the remaining out-of-reference candidates are excluded.

### Constructed trajectory

Each raw trajectory is stored as an ordered JSON list. The first two entries
contain topic-level statistics and target-reference IDs. The remaining
entries record the initialization search and subsequent compare-then-update
rounds, including retrieval results, the accumulated paper collection,
unmatched papers, and matched-reference information. These trajectories are
subsequently converted into selector state-action supervision.

### SFT records

Released SFT data are stored in Parquet format. Paper-filter records contain a
`messages` conversation whose assistant target is `yes` or `no`.
Retrieval-selector records contain `messages`, `enable_thinking`, and
`extra_info`; the assistant target is `search` with keywords, `citation`, or
`END`. The `extra_info` field records the source topic, trajectory step, and
other provenance needed to trace the example back to its constructed
trajectory.

## 🔒 Reproducibility and security

Large datasets, generated outputs, checkpoints, experiment logs, and secrets
are ignored by Git. Do not commit API keys. The code imported from research
prototypes was reorganized to remove machine-specific paths and embedded
credentials. If a credential was previously committed elsewhere, rotate it
before making that repository public.

Run all offline checks with:

```bash
make check
```

## 📚 Citation

Citation metadata will be updated after publication. See `CITATION.cff`.

## ⚖️ License

The source code is released under the [MIT License](LICENSE). Dataset and
checkpoint licenses are specified separately in their corresponding Hugging
Face repository cards.
