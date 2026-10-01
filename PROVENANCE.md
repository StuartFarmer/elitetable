# Provenance and reproducibility

This repository was extracted from the completed job `c05debadc3ea4013ab8f84769f859551`, group `main-v2-amd64-episode45`. [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) records the original source file hashes and transformations. [data/provenance.json](data/provenance.json) records the compact evidence checksum and original archive identifiers; the evidence also retains per-run hashes of the original databases, manifests, call logs and source snapshots.

`EliteTable` is the paper implementation formerly named `EliteSearch`, not the newer development runner. The operator schedule, parent sampling, repair allowance mechanics, retry behavior, tie-breaking and checkpoint restoration are preserved. The five prompt templates and edit validation are retained. A package-local renderer replaces Slick's mutable template path; tests check the original requests byte-for-byte. The runner sets the completed paper's parameters from `study.json` and removes unrelated UI/video machinery.

The retained `rsikit` code provides the paper's policy interface, score cache, evaluator and Docker worker. It is repository source, not a dependency on another checkout. Unrelated AlphaEvolve agents, pilot runs, custom environments and notebook tooling are omitted. The edit validator originated in `alphaevolve/edits.py`; no FunSearch island implementation is included. Third-party Python dependencies are installed from their distributions and retain their respective licenses. No new licensing grant for the author's source is inferred here.

## What the checks establish

- `python analyze.py --check` recomputes results from raw recorded scores, not hard-coded summary values. It compares all fields in the four archived reference CSV tables and the experiment counts, including missing evidence and unknown billing. Floating-point comparison tolerance is 1e-12; table results matched on the extraction environment.
- The default data contains 60 searches, all 30,000 candidate slots, 506 distinct within-run winner identities, and complete endpoint panels. Repeated generation winners reuse evidence. One MountainCar generation-six panel failed and is kept missing.
- Compact data includes per-seed returns, winner programs as inert text, candidate population scores/statuses, compact API usage/error metadata, and original hashes. Full prompts for each recorded call and non-winning candidate implementations remain in the original run archive; they are not necessary for the paper's numerical reconstruction and are not duplicated here.
- The included tests exercise operator allocation, ranking/ties, repair, retry, restoration, prompt equivalence, held-out separation, offline analysis and CLI configuration without paid requests.

The worker recipe pins the recorded runtime versions. Its rebuilt image has a new identifier and cannot be assumed numerically or operationally identical to the unavailable original AMD64 image. Full sixty-search generation has not been repeated during extraction. A separate smoke check ran one fixed-policy CartPole episode through the extracted host backend using an existing local ARM64 worker; that checks the transport path, not reproduction of AMD64 paper scores. Hosted LLM searches remain stochastic even with the same search seeds.

Use the read-only original evidence as the source of published measurements. Newly generated searches are separate replication results; their manifests preserve the actual image and attempt history.
