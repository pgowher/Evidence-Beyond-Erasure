"""Reproduce the paired fixed-window comparison from the simulation generator.

Run from the project root with python -m CAISE_2027.paired_stream_analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import simUp as sim


POPULATION = 450_000_000
CAPACITY = sim.WORKFLOW_ASSUMPTIONS.reference_batch_size
WINDOW_SECONDS = sim.SECONDS_PER_MINUTE
RETENTION_DAYS = 365
COPIES = 3
DATA = Path(__file__).resolve().parents[1] / "runs" / "reviewer_revision" / "htgl_data"


def layout_costs(streams, independent, leaves):
    anchors = np.zeros_like(streams[0])
    independent_bytes = 0
    leaf_bytes = 0
    for counts in streams:
        full, partial = np.divmod(counts, CAPACITY)
        anchors += full + (partial > 0)
        independent_bytes += full.sum() * independent[CAPACITY] + independent[partial].sum()
        leaf_bytes += full.sum() * leaves[CAPACITY] + leaves[partial].sum()
    return anchors, independent_bytes, leaf_bytes


def main():
    archived = pd.read_csv(DATA / "paper_batching_reconciliation.csv")
    sizes = sim.record_size_profiles()["canonical_protocol_v2"]
    independent = sim.witness_cost_lookup(CAPACITY, "independent_encoded_paths")
    leaves = sim.witness_cost_lookup(CAPACITY, "leaves_on_demand")
    model = "causal_workflow"
    origin_columns = list(sim.COMPONENT_COLUMNS[:4])
    rows = []
    for scenario in sim.SCENARIOS:
        if scenario.name not in ("baseline", "combined_stress"):
            continue
        rates = sim.analytical_component_rates(POPULATION, scenario)
        for replication in range(1, sim.TEMPORAL_REPLICATIONS + 1):
            trace = sim.simulate_temporal_day(
                rates, scenario, sim.experiment_rng("temporal", POPULATION, scenario.name, model, replication),
                model=model,
            ).trace
            origins = trace[origin_columns].sum(axis=1).to_numpy()
            requests = trace["Re_Erasure"].to_numpy()
            record_bytes = (
                origins.sum() * sizes["generic"]
                + requests.sum() * sizes["request"]
                + trace["Rt_Tombstone"].sum() * sizes["tombstone"]
                + trace["Rv_AuditVerification"].sum() * sizes["receipt"]
            )
            for layout, streams in (
                ("all_events_pooled", (trace["Events"].to_numpy(),)),
                ("origins_and_requests_pooled", (origins + requests,)),
                ("origins_and_requests_separate", (origins, requests)),
            ):
                anchors, independent_bytes, leaf_bytes = layout_costs(streams, independent, leaves)
                common_bytes = record_bytes + sizes["anchor"] * anchors.sum()
                rows.append(
                    {
                        "scenario": scenario.name,
                        "layout": layout,
                        "replication": replication,
                        "daily_mean_anchors_per_second": anchors.mean() / WINDOW_SECONDS,
                        "daily_peak_anchors_per_second": anchors.max() / WINDOW_SECONDS,
                        "independent_annual_tb_at_3_copies": (common_bytes + independent_bytes) * RETENTION_DAYS * COPIES / 1e12,
                        "leaf_only_annual_tb_at_3_copies": (common_bytes + leaf_bytes) * RETENTION_DAYS * COPIES / 1e12,
                    }
                )

    measurements = pd.DataFrame(rows)
    summary = measurements.groupby(["scenario", "layout"], sort=False).agg(
        replications=("replication", "size"),
        mean_anchors_per_second=("daily_mean_anchors_per_second", "mean"),
        p99_of_daily_peak_anchors_per_second=("daily_peak_anchors_per_second", lambda x: x.quantile(0.99)),
        independent_annual_tb_at_3_copies=("independent_annual_tb_at_3_copies", "mean"),
        leaf_only_annual_tb_at_3_copies=("leaf_only_annual_tb_at_3_copies", "mean"),
    ).reset_index()
    for row in summary.itertuples(index=False):
        if row.layout == "all_events_pooled":
            reference = archived.loc[
                (archived.population == POPULATION) & (archived.scenario == row.scenario)
                & (archived.model == model) & (archived.batch_size == CAPACITY)
            ].iloc[0]
            np.testing.assert_allclose(
                (row.mean_anchors_per_second, row.p99_of_daily_peak_anchors_per_second),
                (reference.mean_anchor_tps, reference.p99_of_daily_max_anchor_tps),
                atol=1e-12, rtol=0,
            )
    summary.to_csv(Path(__file__).with_name("paired_stream_results.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
