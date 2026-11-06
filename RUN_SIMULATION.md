# HTGL simulation and signed reference verifier

Use Python 3.10 or newer. Install the dependencies in an isolated environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python simUp.py --output-dir runs/publication
```

The default publication profile runs 10,000 daily samples per population, scenario and model, 1,000 temporal replications per group, 1,000 peak bootstrap resamples, 5,000 matched Latin-hypercube samples per population and 4,096 signed witness fixtures. It evaluates capacities 1, 10, 100, 500, 1,000, 5,000 and 10,000 and analytical closure windows 10, 30, 60, 300 and 900 seconds. The temporal comparison uses 60-second windows. The event-driven implementation also measures batch closure and waiting under each capacity and timeout.

`--profile verification` exercises every stage with 12 temporal replications, 100 bootstrap resamples and 37 signed witnesses. It is useful for development checks. The daily and uncertainty workloads retain their publication sample sizes. Keep verification results in a separate directory. Use `--seed` to set the deterministic statistical seed. A run without `--output-dir` uses the script directory, with generated files under `htgl_data/` and `paper_artifacts/`. The program replaces its generated outputs when rerun in the same directory.

Regenerate tables and figures from completed outputs with the same profile:

```bash
python simUp.py --figures-only --output-dir runs/publication
```

The manifest records the exact script and submitted manuscript hashes when the manuscript is supplied or found in the project, statistical seed, run profile and observed dependency versions. Statistical streams derive from the seed and experiment labels. Real signatures and secret openings use OS randomness. Portable witness bytes and artifact hashes that depend on those bytes vary across runs. Runtime timings depend on the host and filesystem.

## Verify a portable witness

The executed reference workflow creates signed origin evidence, anchors an independently signed erasure request, removes the synthetic payload and opening from the application store, creates a custodian declaration and checks the retained witness. The local SQLite anchor log uses role-bound Ed25519 signatures, three of four validator signatures, durable prepare locks and an append-only record history. The executed quorum protocol uses one view. The local timings measure this backend on the execution host. External network finality and throughput require a separate deployed backend experiment.

The generated target and trust registry are separate verifier inputs. For use beyond the fixture, obtain them from the auditor's trusted case and key records. The worked witness comes from the deliberate false-declaration boundary fixture. The test matrix also includes the cases with observed logical removal.

```bash
python simUp.py \
  --verify-witness runs/publication/htgl_data/worked_witness.json \
  --trust-registry runs/publication/htgl_data/public_trust_registry.json \
  --audit-target runs/publication/htgl_data/worked_audit_target.json
```

Inspection is repeatable. To admit the declaration as a new entry, add `--register-declaration --replay-db runs/auditor.sqlite --verifier-id 00000000000000000000000000000001`. The first valid admission succeeds. A subsequent admission with the same identity and contents returns `DUPLICATE_DECLARATION` and exit status 2. A changed declaration under the same identity returns `CONFLICTING_DECLARATION`. Restart preserves the namespace state. Independent verifier namespaces may each register the declaration.

Protocol v2 uses canonical framed binary records. The exported JSON wrapper carries these bytes as lowercase hexadecimal. The target pins the expected tenant, event, resource, policy, request, custodian, scope, both batch IDs and both leaf digests. Request verification checks the declared roster, authorized scope, and chronology from the origin to the declaration. The verifier checks signed timestamp order. The local fixture constructs both anchors before declaration. Case completeness is assessed for the roster in the authenticated request. The protocol specification and worked witness describe the v2 construction for the revised manuscript.

## Replay an operational trace

Supply a CSV with exactly these headers:

```csv
timestamp,event_id,event_class,parent_event_id
```

Use timezone-aware ISO-8601 timestamps and unique opaque event IDs. Valid classes are `Rc_Consent`, `Ra_Authorization`, `Rm_DER`, `Ri_Integrity`, `Re_Erasure`, `Rt_Tombstone` and `Rv_AuditVerification`. Root rows have a blank parent. Each descendant refers to an erasure request in the same trace with an equal or earlier timestamp. Use an inclusive start and exclusive end for the actual exposure window, including intervals with zero events. The window must cover at least two hours.

```bash
python simUp.py --output-dir runs/trace-study \
  --trace-csv /absolute/path/governance-events.csv \
  --trace-start 2026-01-01T00:00:00Z --trace-end 2026-01-02T00:00:00Z \
  --trace-population 10000 --trace-institutions 25 \
  --trace-source 'Describe the source, event granularity and exposure here'
```

The trace outputs contain exposure-normalized rates, parent delays, observed event-time batching, right censoring, analytical queue estimates and a comparison of held-out minutes with earlier training blocks. The program retains aggregate trace outputs and the source hash. The default simulation uses declared synthetic workloads. Its automatic trace test uses a labelled synthetic fixture.

## Read the results

| Output in `htgl_data/` | Contents |
|---|---|
| `reviewer_comment_traceability.csv` | Each reviewer point, its implementation and the corresponding evidence |
| `validation_report.csv` | Blocking checks for reproducibility, storage identities, workloads and protocol coverage |
| `formal_model_verification.csv` | Fault, replay and boundary test counts, outcomes, reason codes and measured timings |
| `protocol_conformance_checks.csv` | Canonical byte vectors, odd trees, distinct audit targets, custody coverage, concurrent admission, quorum faults and restart |
| `evidence_protocol_specification.json` and `canonical_test_vectors.json` | Exact serialization, domain bytes, trusted inputs and public test vectors |
| `record_size_field_breakdown.csv` and `record_size_totals.csv` | Legacy sizing assumptions and actual binary codec lengths, including the retained redundant leaf |
| `storage_detailed.csv` and `storage_policy_comparison.csv` | Path-inclusive legacy accounting and comparisons for the legacy and v2 record profiles |
| `retention_policy_measurements.csv` | Actual serialized paths, shared nodes, leaf-only and compressed repositories, with verified retrieval and measured generation cost |
| `event_class_retention_comparison.csv` | Class-specific retention, including the modeled hold on origins linked to longer-lived requests |
| `paper_batching_reconciliation.csv` | Unrounded deterministic anchor rates and separately calculated temporal means and peak estimates |
| `paper_uncertainty_statistics.csv` | Mean, median, central 90% and 95% empirical intervals, and p99 over declared assumption ranges |
| `workload_assumption_rationale.csv` and `burst_assumption_rationale.csv` | Design inputs and the hypotheses they represent |
| `retained_metadata_catalogue.csv` and `metadata_linkage_metrics.csv` | Retained fields and a bounded identifier-equality linkage experiment |
| `local_reference_pipeline_timings.csv` and `local_reference_verification_timings.csv` | Wall-clock measurements for the local backend |

The legacy 296/204/170/208-byte envelope remains available for comparison with Equation (16). The v2 profile measures serialized records for a stated actor, custodian and quorum count. The reference storage and uncertainty calculations use the same accounting function. Compact representations provide verified proofs when the retained repository is available. The storage calculations model serialized bytes and specified replica counts. Storage-engine amplification and external retrieval-index overhead are separate deployment costs.

The 3,500 TPS line marks the configuration-specific Fabric result. Other service rates are declared analytical envelopes. [Androulaki et al., EuroSys 2018](https://doi.org/10.1145/3190508.3190538)

The signature and persistence implementation uses the documented Ed25519 and SQLite interfaces. [Ed25519 API](https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ed25519/) · [SQLite API](https://docs.python.org/3/library/sqlite3.html)

The study evaluates conditional workloads, local reference execution, evidence consistency and declarations within the authenticated scope. Physical deletion, custodian truthfulness, undisclosed copies or custodians, anonymity, regulatory compliance and deployment performance remain outside those findings. The executed false-declaration fixture records this boundary explicitly.
