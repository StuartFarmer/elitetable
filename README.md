# EliteTable

Companion code for **EliteTable: Self-Improving Program Search for Executable Control Policies**. The importable algorithm lives in `elitetable.py`; `run.py` runs the six-environment experiment; `analyze.py` reconstructs the paper's results from recorded episode returns.

## Reproduce the paper results

Use Python 3.14. From this repository:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-analysis.txt
python analyze.py --check
```

This is offline after dependency installation: no API key, Docker, policy execution, or model calls. It recomputes the tables, bootstrap intervals and figures from the bundled evidence and checks them against the archived results. Outputs go to `outputs/`.

The evidence covers all 60 searches, 600 generations, and 60 valid first-to-final endpoint pairs. One intermediate MountainCar held-out evaluation is missing; complete billing is unknown for two requests. These gaps are preserved. See [data provenance](data/provenance.json) and [source provenance](PROVENANCE.md).

## Run new experiments

Install the search dependencies and start Docker:

```sh
pip install swig
pip install -e '.[run,analysis]'
docker build --platform linux/amd64 -t elitetable-worker:paper .
export OPENROUTER_API_KEY=your-key
python run.py --env CartPole-v1 --search-seed 0 \
  --image elitetable-worker:paper --output runs/cartpole-0 --dry-run
```

With `uv`, `uv sync --locked --extra run --extra analysis` installs the committed dependency lock instead of the two `pip install` commands.

Remove `--dry-run` to launch the search. New searches make paid OpenRouter calls. The paper settings are the defaults: ten generations, fifty candidate slots per generation, ten elites, five repairs per candidate, ten search seeds, one hundred held-out seeds, eight episode workers, twenty-five concurrent model operations, a 45-second episode limit and a ten-second policy-call limit. Search continues after target attainment.

Run all sixty searches sequentially, then analyze their recorded evidence:

```sh
python run.py --all --image elitetable-worker:paper --output runs/new-study --dry-run
# Remove --dry-run to execute the printed batch.
python analyze.py --runs runs/new-study --output outputs/new-study
```

Each search requires a new output directory. After interruption, resume its checkpoint explicitly:

```sh
python run.py --resume runs/cartpole-0
```

`study.json` preserves the actual paper configuration, including the original worker image ID. The runner defaults to that ID; pass `--image elitetable-worker:paper` for a rebuilt worker. The Dockerfile pins the recorded Python and package versions, but a rebuild is not the original image and platform/timing differences can affect results. The original Linux AMD64 image is not distributed in this repository. Hardware must accommodate the fixed execution deadlines.

**New searches do not guarantee identical policies or rewards.** Hosted model responses and provider routing are stochastic. Exact reproduction here means reconstructing the paper's numerical analysis from its recorded evidence; a new search replicates the documented procedure. A rerun with another worker image is recorded as such.

## Import EliteTable

```sh
pip install .
```

```python
from elitetable import EliteTable, Config, Measurement

# provider.acall(prompt) supplies model responses.
# evaluate(policies) returns {policy.id: Measurement({seed: reward, ...})}.
search = EliteTable(
    task="Maximize cumulative reward in my environment.",
    provider=provider,
    evaluate=evaluate,
    context="Describe the policy interface, environment and evaluation limits here.",
    config=Config(generations=10, population_size=50, elite_size=10,
                  max_repairs=5, generation_concurrency=25),
    seed=0,
)
elites = await search.run()
best_policy = search.best
```

The evaluator is injected; importing or constructing the class does not launch Docker or call a model. The library resolves its own packaged prompts without changing Slick's global template path. Generated policies implement `rsikit.Policy`. Use isolated execution for generated source, as the paper runner does. Infrastructure failures must raise; candidate failures return `Measurement(scores={}, failure="diagnostic")` and enter the bounded repair loop.

## Files and checks

- `elitetable.py` and `_elitetable/`: search procedure, records, exact-edit validation and prompts.
- `run.py`, `study.json`: experiment CLI and paper configuration.
- `rsikit/`, `Dockerfile`: retained execution and checkpoint dependencies.
- `analyze.py`, `data/`: portable evidence and offline paper reconstruction.
- `tests/`: offline algorithm, runner, and analysis regression checks.

```sh
python -m unittest discover -s tests
python analyze.py --check
```

No pilot runs, unrelated search algorithms, notebook framework, credentials or editable dependencies on another checkout are required.
