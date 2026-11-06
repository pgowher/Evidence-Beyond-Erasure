# Publication figure captions

## fig_01_population_scale: Population-scale workload envelope

Causal-model p99 daily demand. The 3,500 TPS line marks the configuration-specific Fabric literature comparison from Androulaki et al., EuroSys 2018 (https://doi.org/10.1145/3190508.3190538). The 8,000 TPS line is an assumed capacity envelope.

Source data: `htgl_data/daily_model_comparison.csv`.

## fig_02_causal_dependence: Causal dependence versus the Poisson null

Explicit descendant generation restores the required erasure–tombstone–verification dependence that is absent from the independent-Poisson comparator.

Source data: `htgl_data/workflow_dependence_metrics.csv`.

## fig_03_temporal_peaks: Correlated temporal demand at 450 million

Temporal peaks under the declared MMPP and diurnal stress parameters. The 3,500 TPS reference is the configuration-specific Fabric comparison from Androulaki et al. (https://doi.org/10.1145/3190508.3190538).

Source data: `htgl_data/temporal_scenario_summary.csv`.

## fig_04_batching_and_queue: Batching and queue feasibility

Reference Merkle batching reduces consensus arrivals by orders of magnitude. The queue panel separates that design from the conservative event-per-transaction counterfactual.

Source data: `htgl_data/temporal_batching_summary.csv;htgl_data/capacity_queue_summary.csv`.

## fig_05_uncertainty_and_prcc: Uncertainty envelope and parameter influence

Latin-hypercube samples propagate declared assumption ranges; PRCC identifies which inputs dominate 450-million logical demand.

Source data: `htgl_data/uncertainty_summary.csv;htgl_data/prcc_sensitivity.csv`.

## fig_06_storage_envelope: Retained-evidence storage envelope

Annual single-copy decimal-terabyte storage under the legacy 296-byte envelope, with requests, tombstones, receipts, one digest path per event, and anchor receipts.

Source data: `htgl_data/storage_detailed.csv`.

## fig_07_abstract_verifier: Signed post-erasure reference verification

Exact outcomes for the executed signed-reference cases, including deliberately accepted boundary cases. The data table includes counts and reasons for every case.

Source data: `htgl_data/formal_model_verification.csv`.

## fig_08_compact_retention: Compact witness retention policies

Protocol-v2 storage at 450 million represented participants under uniform 365-day retention. Shared and leaf-only representations derive proof bytes from the actual codecs. Compressed storage extrapolates measured serialized fixture costs.

Source data: `htgl_data/storage_policy_comparison.csv;htgl_data/retention_policy_measurements.csv`.
