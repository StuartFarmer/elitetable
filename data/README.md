# Paper evidence

`evidence.json.gz` is a losslessly compressed JSON extraction from all 60 completed
`main-v2-amd64-episode45` searches. `python analyze.py --check` reads it offline and
recreates the tables and two figures under `outputs/`. It needs NumPy and
Matplotlib, with no Gymnasium, model API, Docker, or generated-policy execution.

The JSON contains the frozen study and, for each search:

- Original experiment, attempts and status metadata; source ZIP member hashes and
  original input-file SHA-256 hashes.
- Every generation's status, elite IDs, winning policy identity and search score.
- Every candidate's `[id, generation, status, score, repairs]`, in that order:
  30,000 rows, including 29,959 evaluated candidates and 41 discarded candidates.
- Winner name and implementation as inert text for identity verification; cached
  SQL episode scores and the authoritative held-out JSON panels with their errors.
- Per-seed held-out scores, random-policy panels and identity-matched summaries.
- Each logged API call's ID, timestamp, model, provider, usage, error and duration.
  Requests, responses and environment credentials are omitted.

`provenance.json` records the bundle hash, original release metadata/table hashes,
and the frozen source manifest. These hashes provide an audit trail back to the
full archive; the compact extraction is not a substitute for the original logs
when investigating omitted prompts, candidate implementations or worker failures.
The extractor checks all candidate policy IDs; bundled analysis rechecks winner
identities from their retained name and source. Source identities use the original
`sha256(name + "\0" + implementation)` rule.

Re-extract from the original archive with:

```sh
python analyze.py --export-archive /path/to/batch/release --output data
python analyze.py --check
```

`reference-*.csv` are unchanged archived tables, used only by `--check`, never as
analysis inputs. All result, inventory, valid-count and resource cells are compared
exactly when possible, otherwise with relative and absolute tolerance `1e-12` for
floating-point serialization/summation. The statistics use median paired
generation-10 minus generation-1 held-out means, with 10,000 bootstrap resamples
and a fresh NumPy `default_rng(20260925)` for each environment. The representative
search is the complete trajectory nearest the median paired improvement, with the
lower search seed breaking ties. Population figures pool individual candidate
means from completed generations; they do not average search-level medians.

There are 599 valid held-out panels out of 600 and all 60 valid endpoint pairs.
MountainCar seed 8 generation 6 failed: its 99 finite cached held-out scores remain
in the evidence, but never become a partial-panel mean. CarRacing seed 7 has two
`CancelledError` calls without usage. Complete tokens and cost remain `unknown`;
`recorded_*` totals sum only reported usage and are not complete billing totals.
The batch logged 38,289 calls and 8,287 repairs. The actual protocol used 45-second
episode limits, 10-second policy calls, 8 workers and 25 concurrent model operations.

`--runs DIR` analyzes a saved run or a directory of runs using their own recorded
protocol. It rejects duplicate seeds and mixed protocols, checks attempt contexts
and source continuity, and labels its outputs `custom-runs`. It does not claim new
code is identical to the archived source or allow `--check` against paper results.
Paper eligibility additionally requires the original frozen source and protocol.
