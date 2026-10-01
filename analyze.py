"""Recreate paper tables and figures offline, or analyze saved experiment folders."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
import sqlite3
import zipfile
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from statistics import fmean

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
POPULATION_FIELDS = ("id", "generation", "status", "score", "repairs")
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "cost")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def finite(value):
    return value is not None and math.isfinite(value)


def panel_mean(evidence, seeds):
    scores = evidence.get("scores", {})
    if (scores and not evidence.get("failure") and set(map(int, scores)) == set(seeds)
            and all(finite(v) for v in scores.values())):
        return fmean(scores.values())
    return None


def source_members(path):
    with zipfile.ZipFile(path) as archive:
        return {name: hashlib.sha256(archive.read(name)).hexdigest()
                for name in sorted(archive.namelist())
                if Path(name).suffix in (".py", ".j2", ".txt")}


def extract_run(path, frozen_source=None):
    """Read checkpoints only. Never import or execute an archived policy."""
    experiment = read_json(path / "experiment.json")
    if not experiment:
        raise ValueError(f"Missing experiment.json: {path}")
    attempts = read_json(path / "attempts.json", [])
    record = dict(run=path.name, experiment=experiment, attempts=attempts,
                  status=read_json(path / "status.json", {"status": "unknown"}),
                  heldout=read_json(path / "heldout.json", {}),
                  summary=read_json(path / "summary.json", {}))
    with closing(sqlite3.connect((path / "run.sqlite").resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        organisms = {r["id"]: dict(r) for r in db.execute("SELECT * FROM elitesearch_organism ORDER BY id")}
        record["population"] = [[r[k] for k in POPULATION_FIELDS] for r in organisms.values()]
        record["generations"] = [dict(r) for r in db.execute(
            "SELECT number, status, elite_ids FROM elitesearch_generation ORDER BY number")]
        record["policies"] = {}
        record["policy_identity_mismatches"] = []
        for r in organisms.values():
            if r["policy_id"] and hashlib.sha256((r["name"] + "\0" + r["implementation"]).encode()).hexdigest() != r["policy_id"]:
                record["policy_identity_mismatches"].append(r["id"])
        for generation in record["generations"]:
            generation["elite_ids"] = json.loads(generation["elite_ids"])
            ids = generation["elite_ids"]
            if generation["status"] != "completed" or not ids:
                continue
            winner = organisms[ids[0]]
            generation.update(policy_id=winner["policy_id"], search_mean=winner["score"])
            cached = db.execute("SELECT scores FROM policy WHERE id = ?", (winner["policy_id"],)).fetchone()
            record["policies"][winner["policy_id"]] = dict(
                name=winner["name"], implementation=winner["implementation"],
                cached_scores=json.loads(cached[0]) if cached else {})
    record["calls"] = []
    record["partial_call_log"] = False
    call_path = path / "llm_calls.jsonl"
    if call_path.exists():
        lines = call_path.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                call = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
                record["partial_call_log"] = True
                continue
            record["calls"].append({k: call.get(k) for k in
                                    ("id", "started_at", "model", "provider", "usage", "error", "elapsed_seconds")})
    else:
        record["partial_call_log"] = True
    snapshots = sorted({a.get("source", "source.zip") for a in attempts} | {"source.zip"})
    record["source_snapshots"] = {}
    for name in snapshots:
        try:
            record["source_snapshots"][name] = source_members(path / name)
        except (OSError, zipfile.BadZipFile):
            record["source_snapshots"][name] = None
    record["frozen_source_matches"] = None
    if frozen_source is not None:
        prefixes = ("rsikit/", "elitesearch/", "alphaevolve/", "examples/")
        record["frozen_source_matches"] = all(
            members is not None and {k: v for k, v in members.items() if k.startswith(prefixes)} == frozen_source
            for members in record["source_snapshots"].values())
    files = {"experiment.json", "attempts.json", "status.json", "heldout.json", "summary.json",
             "run.sqlite", "llm_calls.jsonl", "curves.csv", "context.txt"} | set(snapshots)
    record["file_sha256"] = {name: sha256(path / name) for name in sorted(files) if (path / name).exists()}
    return record


def load_evidence():
    provenance = read_json(DATA / "provenance.json")
    if sha256(DATA / "evidence.json.gz") != provenance["evidence_sha256"]:
        raise ValueError("Bundled evidence checksum mismatch")
    with gzip.open(DATA / "evidence.json.gz", "rt") as stream:
        return json.load(stream)


def trajectory(record):
    rows = []
    for generation in record["generations"]:
        if generation["status"] != "completed":
            continue
        identity = generation.get("policy_id")
        evidence = record["heldout"].get(identity, {})
        if not evidence and identity and identity == record["summary"].get("best_id"):
            evidence = record["summary"].get("heldout") or {}
        rows.append(dict(generation=generation["number"], policy_id=identity,
                         search_mean=generation.get("search_mean"),
                         heldout_mean=panel_mean(evidence, record["experiment"]["heldout_seeds"])))
    return rows


def protocol_reasons(record, study=None):
    experiment = record["experiment"]
    reasons = []
    if record["status"]["status"] != "completed":
        reasons.append("run:" + record["status"]["status"])
    expected = {k: experiment[k] for k in
                ("model", "image", "docker_image", "env", "search_seed", "seeds", "heldout_seeds",
                 "max_output_tokens", "no_early_stop", "environment", "reporting_target",
                 "policy_timeout", "episode_timeout", "generations", "concurrency", "generation_concurrency", "config")
                if k in experiment}
    if study is not None:
        env, seed = experiment["env"], experiment["search_seed"]
        expected = dict(study["arguments"], env=env, search_seed=seed,
                        seeds=study["episode_seeds"], heldout_seeds=study["heldout_seeds"],
                        no_early_stop=True, docker_image=study["arguments"]["image"],
                        config=study["config"], environment=study["environment_settings"][env],
                        reporting_target=study["reporting_targets"][env])
        reasons.extend("protocol:" + k for k, v in expected.items() if experiment.get(k) != v)
        if experiment.get("versions", {}).get("gymnasium") != "1.3.0":
            reasons.append("protocol:gymnasium")
        if record["frozen_source_matches"] is not True:
            reasons.append("protocol:frozen_source_snapshot")
    context = hashlib.sha256(experiment.get("instructions", "").encode()).hexdigest()
    for attempt in record["attempts"]:
        if any(attempt.get(k, v) != v for k, v in expected.items()):
            reasons.append("protocol:attempt")
        if attempt.get("context_sha256", context) != context:
            reasons.append("protocol:context_history")
    snapshots = record["source_snapshots"]
    initial = snapshots.get("source.zip")
    if not initial or any(m != initial for m in snapshots.values()):
        reasons.append("protocol:source_snapshot")
    if record["policy_identity_mismatches"]:
        reasons.append("protocol:policy_identity")
    for identity, policy in record["policies"].items():
        if hashlib.sha256((policy["name"] + "\0" + policy["implementation"]).encode()).hexdigest() != identity:
            reasons.append("protocol:policy_identity")
    expected_generations = list(range(1, experiment["generations"] + 1))
    if [r["generation"] for r in trajectory(record)] != expected_generations:
        reasons.append("incomplete_search_trajectory")
    return sorted(set(reasons))


def inventory(records, study=None):
    rows = []
    for record in records:
        experiment = record["experiment"]
        curves = trajectory(record)
        values = {r["generation"]: r["heldout_mean"] for r in curves}
        first, final = values.get(1), values.get(experiment["generations"])
        target = experiment.get("reporting_target")
        search = [r["search_mean"] for r in curves if finite(r["search_mean"])]
        reasons = protocol_reasons(record, study)
        rows.append(dict(run=record["run"], environment=experiment["env"], search_seed=experiment["search_seed"],
                         status=record["status"]["status"], eligible=not reasons, exclusion=";".join(reasons),
                         generations=len(curves), valid_heldout_generations=sum(finite(v) for v in values.values()),
                         first_heldout=first, final_heldout=final,
                         improvement=final-first if finite(first) and finite(final) else None,
                         complete_trajectory=all(finite(values.get(g)) for g in range(1, experiment["generations"] + 1)),
                         search_target_reached=any(v >= target for v in search) if search and target is not None else None,
                         heldout_target_reached=final >= target if finite(final) and target is not None else None))
    return rows


def summarize(rows):
    eligible = sorted((r for r in rows if r["eligible"]), key=lambda r: r["search_seed"])
    paired = [r for r in eligible if finite(r["improvement"])]
    delta = np.array([r["improvement"] for r in paired])
    center = float(np.median(delta)) if len(delta) else None
    low = high = None
    if len(delta) >= 2:
        rng = np.random.default_rng(20260925)
        medians = np.median(rng.choice(delta, size=(10000, len(delta))), axis=1)
        low, high = map(float, np.percentile(medians, [2.5, 97.5]))
    candidates = [r for r in paired if r["complete_trajectory"]]
    representative = min(candidates, key=lambda r: (abs(r["improvement"]-center), r["search_seed"])) if candidates else None
    final = [r["final_heldout"] for r in eligible if finite(r["final_heldout"])]
    return dict(environment=rows[0]["environment"], planned=len(rows),
                started=sum(r["status"] != "not_started" for r in rows),
                completed=sum(r["status"] == "completed" for r in rows), eligible=len(eligible),
                valid_final=len(final), paired_searches=len(paired),
                final_median=float(np.median(final)) if final else None,
                improvement_median=center, improvement_ci_low=low, improvement_ci_high=high,
                improved_searches=int((delta > 0).sum()),
                search_target_reached=sum(r["search_target_reached"] is True for r in eligible),
                heldout_target_reached=sum(r["heldout_target_reached"] is True for r in eligible),
                representative=representative["run"] if representative else None)


def resource_row(record):
    calls = record["calls"]
    totals = {}
    for field in USAGE_FIELDS:
        values = [(call.get("usage") or {}).get(field) for call in calls]
        totals[field] = sum(values) if values and all(v is not None for v in values) and not record["partial_call_log"] else "unknown"
    durations = [a.get("elapsed_seconds") for a in record["attempts"] or [record["status"]]]
    return dict(run=record["run"], logged_calls=len(calls), **totals,
                providers=";".join(sorted({c["provider"] for c in calls if c.get("provider")})) or "unknown",
                elapsed_seconds=sum(durations) if all(v is not None for v in durations) else "unknown/incomplete",
                repairs=sum(row[4] for row in record["population"]),
                discarded=sum(row[2] == "discarded" for row in record["population"]))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def population(record):
    completed = {g["number"] for g in record["generations"] if g["status"] == "completed"}
    return [(generation, score) for _, generation, status, score, _ in record["population"]
            if generation in completed and status == "evaluated" and finite(score)]


def quartiles(groups, generations):
    result = np.full((3, len(generations)), np.nan)
    for column, generation in enumerate(generations):
        values = np.asarray(groups.get(generation, []), dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            result[:, column] = np.percentile(values, [25, 50, 75])
    return result


def analyze(evidence, output):
    records, study = evidence["runs"], evidence["study"]
    paper = evidence["kind"] == "archived-paper"
    rows = inventory(records, study if paper else None)
    summaries, counts = [], []
    for env in study["environments"]:
        selected = [r for r in rows if r["environment"] == env]
        if not selected:
            continue
        summary = summarize(selected)
        eligible_names = {r["run"] for r in selected if r["eligible"]}
        eligible = [r for r in records if r["run"] in eligible_names]
        random_means = [panel_mean(r["heldout"].get("random", {}), r["experiment"]["heldout_seeds"]) for r in eligible]
        random_means = [v for v in random_means if finite(v)]
        summary.update(valid_random=len(random_means), random_median=float(np.median(random_means)) if random_means else None)
        summaries.append(summary)
        for metric in ("search_mean", "heldout_mean"):
            for generation in range(1, study["arguments"]["generations"] + 1):
                count = sum(finite(row[metric]) for r in eligible for row in trajectory(r) if row["generation"] == generation)
                counts.append(dict(environment=env, generation=generation, metric=metric,
                                   valid_searches=count, planned=summary["planned"]))
    resources = [resource_row(r) for r in records]
    output.mkdir(parents=True, exist_ok=True)
    for name, table in (("inventory", rows), ("results", summaries), ("valid-counts", counts), ("resources", resources)):
        write_csv(output / f"{name}.csv", table)
    totals = dict(evidence=evidence["kind"], searches=len(records), eligible=sum(r["eligible"] for r in rows),
                  heldout_panels=sum(r["valid_heldout_generations"] for r in rows),
                  paired_searches=sum(r["paired_searches"] for r in summaries),
                  candidates=sum(len(r["population"]) for r in records),
                  evaluated=sum(len(population(r)) for r in records),
                  logged_calls=sum(r["logged_calls"] for r in resources),
                  repairs=sum(r["repairs"] for r in resources), discarded=sum(r["discarded"] for r in resources),
                  unknown_usage_calls=sum(any((c.get("usage") or {}).get(f) is None for f in USAGE_FIELDS) for r in records for c in r["calls"]),
                  partial_call_logs=sum(r["partial_call_log"] for r in records))
    for field in USAGE_FIELDS:
        totals[field] = sum(r[field] for r in resources) if all(isinstance(r[field], (int, float)) for r in resources) else "unknown"
        totals["recorded_" + field] = sum((c.get("usage") or {}).get(field) or 0 for r in records for c in r["calls"])
    (output / "totals.json").write_text(json.dumps(totals, indent=2) + "\n")
    return rows, summaries, totals


def render(evidence, rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records, study = evidence["runs"], evidence["study"]
    generations = np.arange(1, study["arguments"]["generations"] + 1)
    environments = study["environments"]
    ncols = min(3, len(environments))

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output / f"{name}.{extension}", bbox_inches="tight", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(math.ceil(len(environments)/ncols), ncols, figsize=(5*ncols, 4*math.ceil(len(environments)/ncols)), squeeze=False)
    for ax, env in zip(axes.flat, environments):
        pooled = defaultdict(list)
        searches = []
        for record in records:
            if record["experiment"]["env"] == env:
                values = defaultdict(list)
                for generation, score in population(record):
                    pooled[generation].append(score)
                    values[generation].append(score)
                searches.append(values)
        positions = [g for g in generations if pooled[g]]
        distributions = [np.asarray(pooled[g], dtype=float) for g in positions]
        if distributions:
            violins = ax.violinplot(distributions, positions=positions, widths=0.85,
                                    showmeans=False, showmedians=False, showextrema=False)
            for body in violins["bodies"]:
                body.set_facecolor("#176b9c")
                body.set_edgecolor("none")
                body.set_alpha(0.18)
        for values in searches:
            ax.plot(generations, quartiles(values, generations)[1], color="#176b9c", alpha=0.2, linewidth=0.7)
        ax.plot(generations, quartiles(pooled, generations)[1], color="#176b9c", label="Pooled median")
        ax.set(title=env, xlabel="Generation", ylabel="Mean search episode return", xticks=generations)
        if env == "LunarLander-v3":
            outside = sum(np.count_nonzero((values < -1000) | (values > 350)) for values in distributions)
            ax.set(ylim=(-1000, 350), title=f"{env} (zoom; {outside} scores outside view)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(environments):]:
        ax.set_visible(False)
    fig.suptitle("Training population · completed batch" if evidence["kind"] == "archived-paper" else "Training population · saved runs", fontsize=14)
    save(fig, "population-distributions-detail")

    fig, axes = plt.subplots(math.ceil(len(environments)/ncols), ncols, figsize=(5*ncols, 4*math.ceil(len(environments)/ncols)), squeeze=False)
    for ax, env in zip(axes.flat, environments):
        eligible = {r["run"] for r in rows if r["environment"] == env and r["eligible"]}
        selected = [r for r in records if r["run"] in eligible]
        for metric in ("heldout_mean", "search_mean"):
            pooled = defaultdict(list)
            for record in selected:
                values = {r["generation"]: [r[metric]] for r in trajectory(record) if finite(r[metric])}
                for generation, scores in values.items():
                    pooled[generation].extend(scores)
                if metric == "heldout_mean":
                    ax.plot(generations, quartiles(values, generations)[1], color="#ff7f0e", alpha=0.2, linewidth=0.7)
            low, median, high = quartiles(pooled, generations)
            if metric == "heldout_mean":
                ax.fill_between(generations, low, high, color="#ff7f0e", alpha=0.18)
                ax.plot(generations, low, color="#ff7f0e", alpha=0.3, linewidth=0.7)
                ax.plot(generations, high, color="#ff7f0e", alpha=0.3, linewidth=0.7)
                ax.plot(generations, median, color="#ff7f0e", label="Held-out median / IQR")
            else:
                ax.plot(generations, median, "--", color="#176b9c", label="Search median", zorder=3)
        target = study["reporting_targets"].get(env)
        if target is not None:
            ax.axhline(target, color="#998888", linestyle=":", label="Reporting target")
        planned = sum(r["environment"] == env for r in rows)
        ax.set(title=f"{env} ({len(eligible)}/{planned} eligible)", xlabel="Generation", ylabel="Episode return", xticks=generations)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(environments):]:
        ax.set_visible(False)
    save(fig, "learning-curves")


def check_reference(output):
    for name in ("results", "inventory", "valid-counts", "resources"):
        with (DATA / f"reference-{name}.csv").open() as stream:
            expected = list(csv.DictReader(stream))
        with (output / f"{name}.csv").open() as stream:
            actual = list(csv.DictReader(stream))
        key = (lambda r: (r["environment"], r["metric"], int(r["generation"]))) if name == "valid-counts" else (lambda r: r.get("run", r.get("environment")))
        actual, expected = sorted(actual, key=key), sorted(expected, key=key)
        if len(actual) != len(expected):
            raise AssertionError(f"{name}: different row counts")
        for got, want in zip(actual, expected):
            for field, value in want.items():
                if got[field] == value:
                    continue
                # JSON/CSV decimal round-trips and platform float summation only.
                try:
                    same = math.isclose(float(got[field]), float(value), rel_tol=1e-12, abs_tol=1e-12)
                except ValueError:
                    same = False
                if not same:
                    raise AssertionError(f"{name} {key(want)} {field}: {got[field]!r} != {value!r}")
    expected = dict(searches=60, eligible=60, heldout_panels=599, paired_searches=60, candidates=30000,
                    evaluated=29959, logged_calls=38289, repairs=8287, discarded=41, unknown_usage_calls=2,
                    partial_call_logs=0, prompt_tokens="unknown", completion_tokens="unknown", cost="unknown")
    actual = read_json(output / "totals.json")
    for key, value in expected.items():
        if actual[key] != value:
            raise AssertionError(f"{key}: {actual[key]} != {value}")


def export_archive(release, output):
    study = read_json(release / "study.json")
    manifest = read_json(release / "CHECKSUMS.json")
    frozen_source = {name: digest for name, digest in manifest.items()
                     if name.startswith(("rsikit/", "elitesearch/", "alphaevolve/", "examples/"))
                     and Path(name).suffix in (".py", ".j2", ".txt")}
    paths = [release / "runs" / f"{study['group']}-{env}-{seed}"
             for env in study["environments"] for seed in study["search_seeds"]]
    evidence = dict(schema_version=1, kind="archived-paper", study=study,
                    population_fields=POPULATION_FIELDS, runs=[extract_run(p, frozen_source) for p in paths])
    output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(evidence, separators=(",", ":"), allow_nan=False).encode()
    (output / "evidence.json.gz").write_bytes(gzip.compress(payload, mtime=0))
    provenance = dict(source_release=release.parent.name + "/release", source_group=study["group"],
                      extraction_command="python analyze.py --export-archive /path/to/batch/release --output data",
                      evidence_sha256=sha256(output / "evidence.json.gz"),
                      source_files={name: sha256(release / name) for name in ("study.json", "CHECKSUMS.json", "SOURCE_PROVENANCE.json")},
                      frozen_source=frozen_source,
                      note="Per-run original file SHA-256 hashes and per-attempt source member hashes are in evidence.json.gz. No generated policy is executed.")
    for name in ("results", "inventory", "valid-counts", "resources"):
        source = release / f"figures/main/{name}.csv"
        shutil.copyfile(source, output / f"reference-{name}.csv")
        provenance["source_files"][f"figures/main/{name}.csv"] = sha256(source)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def load_custom(runs):
    paths = [runs] if (runs / "experiment.json").exists() else sorted(p.parent for p in runs.glob("*/experiment.json"))
    if not paths:
        raise ValueError(f"No experiment folders: {runs}")
    records = [extract_run(path) for path in paths]
    signatures, seeds = {}, defaultdict(set)
    for record in records:
        experiment = record["experiment"]
        env = experiment["env"]
        keys = ("model", "config", "environment", "seeds", "heldout_seeds", "max_output_tokens",
                "policy_timeout", "episode_timeout", "concurrency", "docker_image", "reporting_target", "generations")
        protocol = {k: experiment.get(k) for k in keys}
        protocol.update(instructions=experiment.get("instructions"),
                        gymnasium=experiment.get("versions", {}).get("gymnasium"),
                        source=record["source_snapshots"].get("source.zip"))
        signature = json.dumps(protocol, sort_keys=True)
        if env in signatures and signatures[env] != signature:
            raise ValueError(f"Mixed protocols for {env}; select one protocol")
        if experiment["search_seed"] in seeds[env]:
            raise ValueError(f"Duplicate search seed for {env}")
        signatures[env] = signature
        seeds[env].add(experiment["search_seed"])
    if len({r["experiment"]["generations"] for r in records}) != 1:
        raise ValueError("Mixed generation budgets; analyze separately")
    study = dict(environments=list(signatures), arguments={"generations": records[0]["experiment"]["generations"]},
                 reporting_targets={r["experiment"]["env"]: r["experiment"].get("reporting_target") for r in records})
    return dict(schema_version=1, kind="custom-runs", study=study, runs=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--runs", type=Path, help="Saved run or parent directory; analyze its own protocol, not paper provenance")
    sources.add_argument("--export-archive", type=Path, metavar="RELEASE", help="Re-extract compact paper evidence from the original release")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--check", action="store_true", help="Compare bundled paper results, counts and resources to archived tables")
    args = parser.parse_args()
    if args.check and (args.runs or args.export_archive):
        parser.error("--check applies only to bundled paper evidence")
    if args.export_archive:
        export_archive(args.export_archive, args.output)
        print(f"Exported compact evidence to {args.output}")
        return
    evidence = load_custom(args.runs) if args.runs else load_evidence()
    rows, _, totals = analyze(evidence, args.output)
    if args.check:
        check_reference(args.output)
        print("Archived tables and resource totals match (relative/absolute tolerance 1e-12).")
    render(evidence, rows, args.output)
    print(f"{totals['searches']} searches, {totals['heldout_panels']} held-out panels; tables and both figures: {args.output}")


if __name__ == "__main__":
    main()
