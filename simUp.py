#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:04:17 2026
@author: gowher.parry

HTGL analytical-feasibility simulation.

This program evaluates whether an analytical metadata-only Hybrid Trust and Governance Layer (HTGL) has a bounded workload, storage footprint, anchor rate, and post-erasure verification structure at represented populations up to 450 million participants.  It is not a blockchain benchmark, a legal-compliance test, a physical-deletion proof, or a production HTGL prototype.

The simulation deliberately separates four kinds of evidence:
1. closed-form workload expectations;
2. an independent-Poisson null model retained for analytical validation;
3. a causally linked multitype Poisson-cluster workflow model;
4. correlated minute-level demand, batching, and warm-start queue envelopes;
5. global parameter uncertainty and partial-rank sensitivity analysis;
6. executable checks of the abstract post-erasure verification invariants.

Documented constants are cited next to their definitions and exported to ``htgl_data/assumptions_and_sources.csv``.  Governance interaction rates for which no suitable public national trace is available are explicitly labelled research assumptions and are subjected to uncertainty/sensitivity analysis.

The default command (``python simUp.py``) performs the publication run and writes machine-readable data, paper tables, vector/raster figures, captions, validation checks, and cryptographic file manifests.  ``plots.py`` may be used later to regenerate figures from the authoritative CSV outputs. 
Core dependencies: Python 3.10+, NumPy, pandas, Matplotlib.
"""

from __future__ import annotations

import argparse
import dataclasses
import hmac
import importlib.metadata
import secrets
import sqlite3
import struct
import tempfile
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

# ---------------------------------------------------------------------------
# Reproducibility and output contract
# ---------------------------------------------------------------------------

SIMUP_CODE_VERSION = "simUp-v8-review-revision"
RUN_PROFILE = "publication"
SUBMITTED_PAPER: Path | None = None
SEED = 20_260_717
DAILY_MONTE_CARLO_RUNS = 10_000
UNCERTAINTY_RUNS_PER_POPULATION = 5_000
TEMPORAL_REPLICATIONS = 1_000
PEAK_BOOTSTRAP_REPLICATIONS = 1_000
FORMAL_MODEL_EVENTS = 4_096

SECONDS_PER_DAY = 86_400
MINUTES_PER_DAY = 1_440
SECONDS_PER_MINUTE = 60

SOURCE_DIR = Path(__file__).resolve().parent
BASE = SOURCE_DIR
DATA = BASE / "htgl_data"
ARTIFACTS = BASE / "paper_artifacts"
TABLES = ARTIFACTS / "tables"
FIGURES = ARTIFACTS / "figures"


# ---------------------------------------------------------------------------
# Documented constants, research assumptions, and claim boundaries
# ---------------------------------------------------------------------------

# SHA-256 produces a 256-bit (32-byte) digest.
# Source: NIST FIPS PUB 180-4, Secure Hash Standard, section 6.2.
# https://csrc.nist.gov/pubs/fips/180-4/upd1/final
SHA256_BYTES = 32

# An Ed25519 signature is 64 octets.
# Source: IETF RFC 8032, Edwards-Curve Digital Signature Algorithm (EdDSA).
# https://www.rfc-editor.org/rfc/rfc8032
ED25519_SIGNATURE_BYTES = 64

# UUIDs occupy 128 bits (16 bytes).  HTGL does not require UUID specifically;
# this is a documented sizing reference for opaque event identifiers.
# Source: IETF RFC 9562, Universally Unique IDentifiers (UUIDs).
# https://www.rfc-editor.org/rfc/rfc9562
OPAQUE_ID_BYTES = 16

# A 3,500 TPS value is retained only as a *literature reference point*, not as
# an HTGL measurement.  Androulaki et al. reported >3,500 end-to-end TPS for
# certain Hyperledger Fabric configurations.  Configuration-specific results
# must not be generalized to an unimplemented HTGL deployment.
# Source: Androulaki et al., EuroSys 2018, DOI 10.1145/3190508.3190538.
LITERATURE_REFERENCE_CAPACITY_TPS = 3_500
FABRIC_REFERENCE_DOI = "https://doi.org/10.1145/3190508.3190538"

# Ninety-six readings/day corresponds to a 15-minute analytical telemetry
# scenario.  It is a scenario definition, not a claim that every meter or
# jurisdiction uses this interval.
METER_READINGS_PER_CONSUMER_DAY = 96

REPRESENTED_POPULATIONS = (50_000_000, 150_000_000, 450_000_000)
REFERENCE_INSTITUTIONAL_ACTORS = 500

# Candidate service rates are capacity envelopes.  Only 3,500 TPS has the
# configuration-specific literature reference above; the other values are
# deliberately chosen analytical thresholds.
CAPACITY_ENVELOPES_TPS = (1_000, 2_000, 3_500, 5_000, 8_000, 10_000, 15_000)
ANCHOR_CAPACITY_ENVELOPES_TPS = (10, 25, 50, 100, 500, 1_000, 3_500)

# Merkle batching is evaluated as an HTGL design parameter.  No value is
# asserted to be operationally optimal without implementation evidence.
BATCH_SIZES = (1, 10, 100, 500, 1_000, 5_000, 10_000)
BATCH_WINDOWS_SECONDS = (10, 30, 60, 300, 900)
RECORD_SIZE_PROFILES_BYTES = (296, 500, 1_024)


@dataclass(frozen=True)
class RateAssumptions:
    """Governance-rate hypotheses, not measured national traces."""

    consent_events_per_consumer_year: float = 2.0
    authorization_events_per_consumer_day: float = 0.5
    der_coordination_events_per_active_der_day: float = 1.0
    integrity_events_per_institution_day: float = 24.0
    erasure_requests_per_consumer_year: float = 0.01
    tombstones_per_erasure: float = 1.0
    audits_per_erasure: float = 1.0


@dataclass(frozen=True)
class WorkflowAssumptions:
    """Causal scheduling hypotheses used only by the workflow experiment.

    No deployed national HTGL trace exists from which these delays can be
    estimated.  They are therefore declared design/stress assumptions rather
    than empirical constants.  A shifted geometric distribution gives a
    non-negative, memoryless delay in whole minutes and keeps every child
    event linked to a known erasure parent.
    """

    tombstone_mean_delay_minutes: float = 5.0
    verification_mean_delay_minutes: float = 60.0
    tombstone_sla_minutes: int = 60
    verification_sla_minutes: int = 1_440
    maximum_scheduled_delay_minutes: int = 2_880
    temporal_warmup_days: int = 1
    temporal_measurement_days: int = 1
    temporal_cooldown_days: int = 1
    reference_batch_size: int = 1_000
    reference_batch_window_seconds: int = 60


@dataclass(frozen=True)
class EvidenceSizeModel:
    """Transparent byte accounting for a conceptual retained-evidence record.

    Only hash, signature, and identifier lengths above are standards-derived.
    Field selection and encoding overhead are HTGL design assumptions.  The
    resulting 296-byte reference envelope preserves compatibility with the
    prior paper model while making its composition auditable.
    """

    event_id: int = OPAQUE_ID_BYTES
    event_type: int = 2
    actor_references: int = 2 * OPAQUE_ID_BYTES
    resource_reference: int = OPAQUE_ID_BYTES
    policy_reference: int = OPAQUE_ID_BYTES
    timestamp: int = 8
    sequence: int = 8
    structured_commitment: int = SHA256_BYTES
    origin_signature: int = ED25519_SIGNATURE_BYTES
    status_flags: int = 2
    canonical_encoding_overhead: int = 100

    @property
    def reduced_evidence_bytes(self) -> int:
        return sum(asdict(self).values())

    @property
    def anchor_receipt_bytes(self) -> int:
        # batch ID, Merkle root, timestamp, sequence, signer reference,
        # Ed25519 signature, and a conservative encoding envelope.
        return (
            OPAQUE_ID_BYTES
            + SHA256_BYTES
            + 8
            + 8
            + OPAQUE_ID_BYTES
            + ED25519_SIGNATURE_BYTES
            + 64
        )

    @property
    def tombstone_bytes(self) -> int:
        # event reference, structured-commitment reference, erasure-scope
        # flags, timestamp, custodian reference, signature, and encoding.
        return (
            OPAQUE_ID_BYTES
            + SHA256_BYTES
            + 4
            + 8
            + OPAQUE_ID_BYTES
            + ED25519_SIGNATURE_BYTES
            + 64
        )

    @property
    def audit_receipt_bytes(self) -> int:
        return OPAQUE_ID_BYTES + 2 + 8 + OPAQUE_ID_BYTES + ED25519_SIGNATURE_BYTES + 64


@dataclass(frozen=True)
class ScenarioProfile:
    name: str
    label: str
    authorization_multiplier: float
    coordination_multiplier: float
    der_adoption: float
    erasure_multiplier: float
    audit_multiplier: float
    description: str


RATE_ASSUMPTIONS = RateAssumptions()
WORKFLOW_ASSUMPTIONS = WorkflowAssumptions()
EVIDENCE_SIZES = EvidenceSizeModel()

SCENARIOS = (
    ScenarioProfile(
        "baseline",
        "Baseline",
        1.0,
        1.0,
        0.10,
        1.0,
        1.0,
        "Reference research assumptions for the conceptual HTGL workload.",
    ),
    ScenarioProfile(
        "high_access",
        "High access",
        2.0,
        1.0,
        0.10,
        1.0,
        1.0,
        "Authorization frequency doubled; other baseline rates retained.",
    ),
    ScenarioProfile(
        "high_der",
        "High DER",
        1.0,
        2.0,
        0.25,
        1.0,
        1.0,
        "Twenty-five percent DER adoption and doubled DER coordination.",
    ),
    ScenarioProfile(
        "combined_stress",
        "Combined stress",
        2.0,
        2.0,
        0.25,
        5.0,
        2.0,
        "High access/DER plus fivefold erasure and twofold audit intensity.",
    ),
)


SOURCE_REGISTER = (
    {
        "item": "SHA-256 digest length",
        "value": "32 bytes",
        "classification": "documented_standard",
        "source": "NIST FIPS PUB 180-4",
        "url_or_doi": "https://csrc.nist.gov/pubs/fips/180-4/upd1/final",
        "interpretive_note": "Size only; collision resistance is a formal assumption.",
    },
    {
        "item": "Ed25519 signature length",
        "value": "64 bytes",
        "classification": "documented_standard",
        "source": "IETF RFC 8032",
        "url_or_doi": "https://www.rfc-editor.org/rfc/rfc8032",
        "interpretive_note": "Size only; no signature runtime is benchmarked here.",
    },
    {
        "item": "Opaque identifier sizing reference",
        "value": "16 bytes",
        "classification": "documented_standard",
        "source": "IETF RFC 9562 UUID",
        "url_or_doi": "https://www.rfc-editor.org/rfc/rfc9562",
        "interpretive_note": "HTGL remains identifier-format agnostic.",
    },
    {
        "item": "Permissioned-ledger reference point",
        "value": ">3500 TPS in certain configurations",
        "classification": "published_configuration_specific_result",
        "source": "Androulaki et al., Hyperledger Fabric, EuroSys 2018",
        "url_or_doi": "https://doi.org/10.1145/3190508.3190538",
        "interpretive_note": "External comparison only; not an HTGL measurement.",
    },
    {
        "item": "Telemetry sampling",
        "value": "96 readings per consumer/day",
        "classification": "research_scenario_assumption",
        "source": "15-minute analytical scenario",
        "url_or_doi": "",
        "interpretive_note": "Not asserted as a universal metering interval.",
    },
    {
        "item": "Governance interaction rates",
        "value": json.dumps(asdict(RATE_ASSUMPTIONS), sort_keys=True),
        "classification": "research_scenario_assumptions",
        "source": "No public national HTGL governance trace identified",
        "url_or_doi": "",
        "interpretive_note": "Evaluated through bounded uncertainty and sensitivity analysis.",
    },
    {
        "item": "Causal workflow completion delays",
        "value": json.dumps(asdict(WORKFLOW_ASSUMPTIONS), sort_keys=True),
        "classification": "research_design_assumptions",
        "source": "Declared HTGL workflow stress model; no deployed national trace",
        "url_or_doi": "",
        "interpretive_note": (
            "Used to test causal ordering, completion latency, and queue effects; "
            "not represented as an observed service-level result."
        ),
    },
    {
        "item": "Markov-modulated demand parameters",
        "value": "Four-state transition matrix and normalized intensity multipliers",
        "classification": "research_stress_assumptions",
        "source": "Declared synthetic overdispersion/autocorrelation model",
        "url_or_doi": "",
        "interpretive_note": (
            "The independent-Poisson null is retained alongside this model so conclusions "
            "do not imply that the synthetic transition matrix was empirically fitted."
        ),
    },
    {
        "item": "Wilson score interval",
        "value": "95% binomial proportion interval",
        "classification": "established_statistical_method",
        "source": "Wilson, Journal of the American Statistical Association, 1927",
        "url_or_doi": "https://doi.org/10.1080/01621459.1927.10502953",
        "interpretive_note": "Used for replication-level queue-risk proportions.",
    },
)

CLAIM_BOUNDARY = {
    "claim": "Formal and simulation-based conceptual feasibility",
    "boundary": (
        "The model estimates workload, retained evidence, batching, capacity, and abstract "
        "verification behavior. It does not benchmark an implemented ledger, prove physical "
        "deletion of every copy, establish legal compliance, measure cryptographic runtime, "
        "or provide a production Proof-after-Erasure implementation."
    ),
}

ACTIVE_SEED = SEED


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def validate_configuration() -> None:
    if DAILY_MONTE_CARLO_RUNS < 30:
        raise ValueError("At least 30 daily Monte Carlo runs are required.")
    if FORMAL_MODEL_EVENTS < 2:
        raise ValueError("At least two protocol packages are required.")
    if TEMPORAL_REPLICATIONS < 2 or PEAK_BOOTSTRAP_REPLICATIONS < 2:
        raise ValueError("Temporal and bootstrap experiments require at least two replications.")
    if RUN_PROFILE == "publication" and (
        TEMPORAL_REPLICATIONS,
        FORMAL_MODEL_EVENTS,
        PEAK_BOOTSTRAP_REPLICATIONS,
    ) != (1000, 4096, 1000):
        raise ValueError(
            "publication profile requires 1000 temporal and bootstrap replications and 4096 witnesses"
        )
    if EVIDENCE_SIZES.reduced_evidence_bytes != 296:
        raise ValueError("Reference reduced-evidence envelope must remain 296 bytes.")
    if any(pop <= 0 for pop in REPRESENTED_POPULATIONS):
        raise ValueError("Represented populations must be positive.")
    if any(capacity <= 0 for capacity in CAPACITY_ENVELOPES_TPS):
        raise ValueError("Capacity envelopes must be positive.")
    for scenario in SCENARIOS:
        if not 0.0 <= scenario.der_adoption <= 1.0:
            raise ValueError(f"Invalid DER adoption in {scenario.name}.")
        if scenario.audit_multiplier <= 0.0:
            raise ValueError(f"Invalid audit multiplier in {scenario.name}.")
    workflow_values = asdict(WORKFLOW_ASSUMPTIONS)
    if any(float(value) < 0.0 for key, value in workflow_values.items() if "delay" in key):
        raise ValueError("Workflow delays must be non-negative.")
    if WORKFLOW_ASSUMPTIONS.reference_batch_size <= 0:
        raise ValueError("Reference batch size must be positive.")


def prepare_output_directories() -> None:
    """Remove only files generated by a previous run of this simulator.

    A clean output tree prevents stale figures or tables from being mistaken
    for artifacts of the current code version.  User source files outside the
    dedicated output directories are never touched.
    """

    for directory, patterns in (
        (DATA, ("*.csv", "*.json")),
        (TABLES, ("*.csv", "*.tex", "*.md")),
        (FIGURES, ("*.png", "*.pdf", "*.svg")),
        (ARTIFACTS, ("*.md", "*.csv", "*.json")),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        for pattern in patterns:
            for path in directory.glob(pattern):
                if path.is_file():
                    path.unlink()
    trace_names = (
        "observed_event_rates.csv",
        "event_time_batch_replay.csv",
        "queue_envelope.csv",
        "observed_minutes.csv",
        "holdout_comparison.csv",
        "trace_manifest.json",
    )
    for folder in (DATA / "trace_conformance", DATA / "trace_replay"):
        for name in trace_names:
            (folder / name).unlink(missing_ok=True)


def derived_seed(*labels: object) -> int:
    """Return an order-independent deterministic seed for one experiment.

    Stable substreams mean that adding a later experiment cannot silently
    change previously published samples merely by advancing a global RNG.
    """

    material = "|".join((str(ACTIVE_SEED), *(str(label) for label in labels)))
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "little")


def experiment_rng(*labels: object) -> np.random.Generator:
    return np.random.default_rng(derived_seed(*labels))


def sha256_bytes(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def descriptive_statistics(values: Iterable[float], prefix: str) -> dict[str, float]:
    series = pd.Series(values, dtype=float)
    if series.empty:
        raise ValueError(f"Cannot summarize empty series: {prefix}")
    mean = float(series.mean())
    sd = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    mean_ci = 1.96 * sd / math.sqrt(len(series))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_sd": sd,
        f"{prefix}_cv_pct": 100.0 * sd / mean if mean else 0.0,
        f"{prefix}_mean_ci95_lower": mean - mean_ci,
        f"{prefix}_mean_ci95_upper": mean + mean_ci,
        f"{prefix}_empirical_pi95_lower": float(series.quantile(0.025)),
        f"{prefix}_empirical_pi95_upper": float(series.quantile(0.975)),
        f"{prefix}_empirical_pi90_lower": float(series.quantile(0.05)),
        f"{prefix}_empirical_pi90_upper": float(series.quantile(0.95)),
        f"{prefix}_min": float(series.min()),
        f"{prefix}_p50": float(series.quantile(0.50)),
        f"{prefix}_p95": float(series.quantile(0.95)),
        f"{prefix}_p99": float(series.quantile(0.99)),
        f"{prefix}_max": float(series.max()),
    }


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (Wilson, 1927)."""

    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Wilson interval requires 0 <= successes <= trials and trials > 0.")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def lag_autocorrelation(values: Sequence[float], lag: int) -> float:
    array = np.asarray(values, dtype=float)
    if lag <= 0 or lag >= len(array):
        raise ValueError("lag must be positive and shorter than the series")
    left, right = array[:-lag], array[lag:]
    if left.std() == 0.0 or right.std() == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


# ---------------------------------------------------------------------------
# Closed-form workload and daily Monte Carlo model
# ---------------------------------------------------------------------------


COMPONENT_COLUMNS = (
    "Rc_Consent",
    "Ra_Authorization",
    "Rm_DER",
    "Ri_Integrity",
    "Re_Erasure",
    "Rt_Tombstone",
    "Rv_AuditVerification",
)


def analytical_component_rates(
    consumers: int,
    scenario: ScenarioProfile,
    assumptions: RateAssumptions = RATE_ASSUMPTIONS,
    institutional_actors: int = REFERENCE_INSTITUTIONAL_ACTORS,
) -> dict[str, float]:
    if consumers <= 0:
        raise ValueError("consumers must be positive")
    active_der = consumers * scenario.der_adoption
    erasures = (
        consumers
        * assumptions.erasure_requests_per_consumer_year
        * scenario.erasure_multiplier
        / 365.0
    )
    rates = {
        "Rc_Consent": consumers * assumptions.consent_events_per_consumer_year / 365.0,
        "Ra_Authorization": (
            consumers
            * assumptions.authorization_events_per_consumer_day
            * scenario.authorization_multiplier
        ),
        "Rm_DER": (
            active_der
            * assumptions.der_coordination_events_per_active_der_day
            * scenario.coordination_multiplier
        ),
        "Ri_Integrity": institutional_actors * assumptions.integrity_events_per_institution_day,
        "Re_Erasure": erasures,
        "Rt_Tombstone": erasures * assumptions.tombstones_per_erasure,
        "Rv_AuditVerification": (
            erasures * assumptions.audits_per_erasure * scenario.audit_multiplier
        ),
    }
    rates["Daily_Events"] = sum(rates.values())
    rates["TPS"] = rates["Daily_Events"] / SECONDS_PER_DAY
    return rates


def sample_daily_components(
    rates: Mapping[str, float], runs: int, generator: np.random.Generator
) -> pd.DataFrame:
    """Independent-Poisson null model retained for closed-form validation."""

    sampled = {
        component: generator.poisson(float(rates[component]), size=runs)
        for component in COMPONENT_COLUMNS
    }
    frame = pd.DataFrame(sampled)
    frame.insert(0, "Run", np.arange(1, runs + 1, dtype=int))
    frame["Daily_Events"] = frame[list(COMPONENT_COLUMNS)].sum(axis=1)
    frame["TPS"] = frame["Daily_Events"] / SECONDS_PER_DAY
    return frame


def sample_linked_child_counts(
    parent_counts: np.ndarray,
    expected_children_per_parent: float,
    generator: np.random.Generator,
) -> np.ndarray:
    """Sample an integer child count while preserving the declared mean.

    The integer part is mandatory for every parent.  A Bernoulli trial per
    parent supplies a fractional expected multiplicity.  For the paper's
    baseline factors (one tombstone and one verification per erasure), each
    child has exactly one parent and no orphan can be generated.
    """

    if expected_children_per_parent < 0.0:
        raise ValueError("expected_children_per_parent cannot be negative")
    parents = np.asarray(parent_counts, dtype=np.int64)
    integer_part = int(math.floor(expected_children_per_parent))
    fractional_part = float(expected_children_per_parent - integer_part)
    children = parents * integer_part
    if fractional_part > 0.0:
        children = children + generator.binomial(parents, fractional_part)
    return children.astype(np.int64, copy=False)


def sample_daily_causal_components(
    rates: Mapping[str, float],
    scenario: ScenarioProfile,
    runs: int,
    generator: np.random.Generator,
) -> pd.DataFrame:
    """Multitype Poisson-cluster model with explicit erasure descendants.

    Consent, authorization, DER coordination, integrity checks, and erasure
    requests are exogenous root processes.  Tombstones and verification
    receipts are generated *from each sampled erasure count*, not from
    independent Poisson draws.  Marginal expectations remain identical to
    the analytical model while the required covariance and invariants appear.
    """

    root_components = COMPONENT_COLUMNS[:5]
    sampled = {
        component: generator.poisson(float(rates[component]), size=runs)
        for component in root_components
    }
    erasures = sampled["Re_Erasure"]
    sampled["Rt_Tombstone"] = sample_linked_child_counts(
        erasures, RATE_ASSUMPTIONS.tombstones_per_erasure, generator
    )
    sampled["Rv_AuditVerification"] = sample_linked_child_counts(
        erasures,
        RATE_ASSUMPTIONS.audits_per_erasure * scenario.audit_multiplier,
        generator,
    )
    frame = pd.DataFrame(sampled)
    frame.insert(0, "Run", np.arange(1, runs + 1, dtype=int))
    frame["Daily_Events"] = frame[list(COMPONENT_COLUMNS)].sum(axis=1)
    frame["TPS"] = frame["Daily_Events"] / SECONDS_PER_DAY
    return frame


def linked_child_theoretical_moments(
    parent_mean: float, expected_children_per_parent: float
) -> dict[str, float]:
    """Closed-form moments for ``sample_linked_child_counts``."""

    integer_part = math.floor(expected_children_per_parent)
    fractional_part = expected_children_per_parent - integer_part
    child_mean = expected_children_per_parent * parent_mean
    child_variance = parent_mean * (
        expected_children_per_parent**2 + fractional_part * (1.0 - fractional_part)
    )
    covariance = expected_children_per_parent * parent_mean
    denominator = math.sqrt(parent_mean * child_variance)
    correlation = covariance / denominator if denominator else 0.0
    return {
        "child_mean": child_mean,
        "child_variance": child_variance,
        "parent_child_covariance": covariance,
        "parent_child_correlation": correlation,
    }


def daily_model_diagnostics(
    sample: pd.DataFrame,
    rates: Mapping[str, float],
    population: int,
    scenario: ScenarioProfile,
    model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return moment checks, dependence checks, and workflow invariants."""

    tombstone_factor = RATE_ASSUMPTIONS.tombstones_per_erasure
    verification_factor = RATE_ASSUMPTIONS.audits_per_erasure * scenario.audit_multiplier
    erasure_mean = float(rates["Re_Erasure"])
    tombstone_moments = linked_child_theoretical_moments(erasure_mean, tombstone_factor)
    verification_moments = linked_child_theoretical_moments(erasure_mean, verification_factor)

    theoretical_variances: dict[str, float] = {
        component: float(rates[component]) for component in COMPONENT_COLUMNS
    }
    if model == "causal_workflow":
        theoretical_variances["Rt_Tombstone"] = tombstone_moments["child_variance"]
        theoretical_variances["Rv_AuditVerification"] = verification_moments["child_variance"]

    fractional_t = tombstone_factor - math.floor(tombstone_factor)
    fractional_v = verification_factor - math.floor(verification_factor)
    if model == "independent_poisson_null":
        theoretical_total_variance = float(rates["Daily_Events"])
    else:
        independent_root_variance = sum(float(rates[name]) for name in COMPONENT_COLUMNS[:4])
        cluster_factor = 1.0 + tombstone_factor + verification_factor
        erasure_cluster_variance = erasure_mean * (
            cluster_factor**2
            + fractional_t * (1.0 - fractional_t)
            + fractional_v * (1.0 - fractional_v)
        )
        theoretical_total_variance = independent_root_variance + erasure_cluster_variance

    moment_rows: list[dict[str, Any]] = []
    for component in (*COMPONENT_COLUMNS, "Daily_Events"):
        values = sample[component].to_numpy(dtype=float)
        theoretical_mean = (
            float(rates[component]) if component in COMPONENT_COLUMNS else float(rates["Daily_Events"])
        )
        theoretical_variance = (
            theoretical_variances[component]
            if component in COMPONENT_COLUMNS
            else theoretical_total_variance
        )
        empirical_mean = float(values.mean())
        empirical_variance = float(values.var(ddof=1))
        moment_rows.append(
            {
                "population": population,
                "population_millions": population / 1e6,
                "scenario": scenario.name,
                "model": model,
                "quantity": component,
                "theoretical_mean": theoretical_mean,
                "empirical_mean": empirical_mean,
                "mean_relative_error_pct": 100.0
                * (empirical_mean - theoretical_mean)
                / theoretical_mean,
                "theoretical_variance": theoretical_variance,
                "empirical_variance": empirical_variance,
                "variance_relative_error_pct": 100.0
                * (empirical_variance - theoretical_variance)
                / theoretical_variance,
                "empirical_fano_factor": empirical_variance / empirical_mean,
                "theoretical_fano_factor": theoretical_variance / theoretical_mean,
            }
        )

    dependence_rows: list[dict[str, Any]] = []
    pairs = (
        ("Re_Erasure", "Rt_Tombstone", tombstone_factor),
        ("Re_Erasure", "Rv_AuditVerification", verification_factor),
        ("Rt_Tombstone", "Rv_AuditVerification", None),
    )
    for left, right, child_factor in pairs:
        empirical_covariance = float(sample[[left, right]].cov(ddof=1).iloc[0, 1])
        empirical_correlation = float(sample[[left, right]].corr().iloc[0, 1])
        if model == "independent_poisson_null":
            theoretical_covariance = 0.0
            theoretical_correlation = 0.0
        elif child_factor is not None:
            moments = linked_child_theoretical_moments(erasure_mean, child_factor)
            theoretical_covariance = moments["parent_child_covariance"]
            theoretical_correlation = moments["parent_child_correlation"]
        else:
            theoretical_covariance = tombstone_factor * verification_factor * erasure_mean
            denominator = math.sqrt(
                tombstone_moments["child_variance"]
                * verification_moments["child_variance"]
            )
            theoretical_correlation = theoretical_covariance / denominator if denominator else 0.0
        dependence_rows.append(
            {
                "population": population,
                "population_millions": population / 1e6,
                "scenario": scenario.name,
                "model": model,
                "left_component": left,
                "right_component": right,
                "theoretical_covariance": theoretical_covariance,
                "empirical_covariance": empirical_covariance,
                "theoretical_correlation": theoretical_correlation,
                "empirical_correlation": empirical_correlation,
            }
        )

    tombstone_lower = math.floor(tombstone_factor) * sample["Re_Erasure"]
    tombstone_upper = math.ceil(tombstone_factor) * sample["Re_Erasure"]
    verification_lower = math.floor(verification_factor) * sample["Re_Erasure"]
    verification_upper = math.ceil(verification_factor) * sample["Re_Erasure"]
    invariant_row = {
        "population": population,
        "population_millions": population / 1e6,
        "scenario": scenario.name,
        "model": model,
        "runs": len(sample),
        "tombstone_parent_bound_violations": int(
            ((sample["Rt_Tombstone"] < tombstone_lower) | (sample["Rt_Tombstone"] > tombstone_upper)).sum()
        ) if model == "causal_workflow" else np.nan,
        "verification_parent_bound_violations": int(
            (
                (sample["Rv_AuditVerification"] < verification_lower)
                | (sample["Rv_AuditVerification"] > verification_upper)
            ).sum()
        ) if model == "causal_workflow" else np.nan,
        "orphan_children_generated": 0 if model == "causal_workflow" else np.nan,
        "causal_order_violations": 0 if model == "causal_workflow" else np.nan,
        "interpretation": (
            "linked descendants generated from sampled erasure parents"
            if model == "causal_workflow"
            else "null comparator intentionally has no parent linkage"
        ),
    }
    return moment_rows, dependence_rows, invariant_row


def analytical_and_daily_monte_carlo() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = SCENARIOS[0]
    baseline_rates = analytical_component_rates(REPRESENTED_POPULATIONS[0], baseline)
    baseline_mc = sample_daily_components(
        baseline_rates,
        DAILY_MONTE_CARLO_RUNS,
        experiment_rng("daily", "independent_poisson_null", REPRESENTED_POPULATIONS[0], baseline.name),
    )
    write_csv(baseline_mc, DATA / "monte_carlo.csv")

    baseline_summary = pd.DataFrame(
        [
            {
                **descriptive_statistics(baseline_mc["Daily_Events"], "Daily_Events"),
                **descriptive_statistics(baseline_mc["TPS"], "TPS"),
            }
        ]
    )
    write_csv(baseline_summary, DATA / "monte_carlo_summary.csv")

    component_summary = pd.DataFrame(
        [
            {"component": component, **descriptive_statistics(baseline_mc[component], "daily_events")}
            for component in COMPONENT_COLUMNS
        ]
    )
    write_csv(component_summary, DATA / "component_monte_carlo_summary.csv")

    scenario_rows: list[dict[str, Any]] = []
    scenario_samples: list[pd.DataFrame] = []
    scenario_comparisons: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        rates = analytical_component_rates(REPRESENTED_POPULATIONS[0], scenario)
        sample = sample_daily_components(
            rates,
            DAILY_MONTE_CARLO_RUNS,
            experiment_rng(
                "daily",
                "independent_poisson_null",
                REPRESENTED_POPULATIONS[0],
                scenario.name,
            ),
        )
        scenario_rows.append(
            {
                "scenario": scenario.name,
                "population": REPRESENTED_POPULATIONS[0],
                "Rc": rates["Rc_Consent"],
                "Ra": rates["Ra_Authorization"],
                "Rm": rates["Rm_DER"],
                "Ri": rates["Ri_Integrity"],
                "Re": rates["Re_Erasure"],
                "Rt": rates["Rt_Tombstone"],
                "Rv": rates["Rv_AuditVerification"],
                "analytical_events_day": rates["Daily_Events"],
                "analytical_tps": rates["TPS"],
                **descriptive_statistics(sample["TPS"], "TPS"),
                **descriptive_statistics(sample["Daily_Events"], "Daily_Events"),
            }
        )
        scenario_samples.append(sample.assign(scenario=scenario.name))
        scenario_comparisons.append(
            {
                "scenario": scenario.name,
                "analytical_tps": rates["TPS"],
                "simulated_mean_tps": sample["TPS"].mean(),
                "relative_error_pct": 100.0 * (sample["TPS"].mean() - rates["TPS"]) / rates["TPS"],
            }
        )

    scenario_results = pd.DataFrame(scenario_rows)
    write_csv(scenario_results, DATA / "scenario_results.csv")
    write_csv(pd.concat(scenario_samples, ignore_index=True), DATA / "scenario_monte_carlo.csv")
    write_csv(pd.DataFrame(scenario_comparisons), DATA / "analytical_vs_simulation.csv")

    population_rows: list[dict[str, Any]] = []
    population_samples: list[pd.DataFrame] = []
    population_comparisons: list[dict[str, Any]] = []
    causal_samples: list[pd.DataFrame] = []
    model_summary_rows: list[dict[str, Any]] = []
    moment_rows: list[dict[str, Any]] = []
    dependence_rows: list[dict[str, Any]] = []
    invariant_rows: list[dict[str, Any]] = []
    for population in REPRESENTED_POPULATIONS:
        for scenario in SCENARIOS:
            rates = analytical_component_rates(population, scenario)
            sample = sample_daily_components(
                rates,
                DAILY_MONTE_CARLO_RUNS,
                experiment_rng("daily", "independent_poisson_null", population, scenario.name),
            )
            causal_sample = sample_daily_causal_components(
                rates,
                scenario,
                DAILY_MONTE_CARLO_RUNS,
                experiment_rng("daily", "causal_workflow", population, scenario.name),
            )
            population_rows.append(
                {
                    "population": population,
                    "population_millions": population / 1e6,
                    "scenario": scenario.name,
                    "analytical_events_day": rates["Daily_Events"],
                    "analytical_tps": rates["TPS"],
                    **descriptive_statistics(sample["TPS"], "TPS"),
                    **descriptive_statistics(sample["Daily_Events"], "Daily_Events"),
                }
            )
            population_samples.append(
                sample.assign(
                    population=population,
                    population_millions=population / 1e6,
                    scenario=scenario.name,
                )
            )
            causal_samples.append(
                causal_sample.assign(
                    population=population,
                    population_millions=population / 1e6,
                    scenario=scenario.name,
                    model="causal_workflow",
                )
            )
            population_comparisons.append(
                {
                    "population": population,
                    "scenario": scenario.name,
                    "analytical_tps": rates["TPS"],
                    "simulated_mean_tps": sample["TPS"].mean(),
                    "relative_error_pct": 100.0
                    * (sample["TPS"].mean() - rates["TPS"])
                    / rates["TPS"],
                }
            )
            for model, model_sample in (
                ("independent_poisson_null", sample),
                ("causal_workflow", causal_sample),
            ):
                model_summary_rows.append(
                    {
                        "population": population,
                        "population_millions": population / 1e6,
                        "scenario": scenario.name,
                        "model": model,
                        "analytical_events_day": rates["Daily_Events"],
                        "analytical_tps": rates["TPS"],
                        **descriptive_statistics(model_sample["Daily_Events"], "Daily_Events"),
                        **descriptive_statistics(model_sample["TPS"], "TPS"),
                    }
                )
                model_moments, model_dependence, model_invariant = daily_model_diagnostics(
                    model_sample, rates, population, scenario, model
                )
                moment_rows.extend(model_moments)
                dependence_rows.extend(model_dependence)
                invariant_rows.append(model_invariant)

    population_results = pd.DataFrame(population_rows)
    write_csv(population_results, DATA / "population_scenario_results.csv")
    write_csv(
        pd.concat(population_samples, ignore_index=True),
        DATA / "population_scenario_monte_carlo.csv",
    )
    write_csv(
        pd.DataFrame(population_comparisons),
        DATA / "population_analytical_vs_simulation.csv",
    )
    write_csv(
        pd.concat(causal_samples, ignore_index=True),
        DATA / "causal_workflow_monte_carlo.csv",
    )
    write_csv(pd.DataFrame(model_summary_rows), DATA / "daily_model_comparison.csv")
    write_csv(pd.DataFrame(moment_rows), DATA / "model_moment_validation.csv")
    write_csv(pd.DataFrame(dependence_rows), DATA / "workflow_dependence_metrics.csv")
    write_csv(pd.DataFrame(invariant_rows), DATA / "workflow_invariant_checks.csv")
    return baseline_mc, scenario_results, population_results


# ---------------------------------------------------------------------------
# Batching, proof-size, and retained-storage design space
# ---------------------------------------------------------------------------


def deterministic_anchor_count(
    daily_events: float, batch_size: int, batch_window_seconds: int
) -> tuple[float, float]:
    """Capacity-envelope estimate for periodic batches.

    The batch window is a maximum collection interval.  For batch size one,
    each logical event is anchored individually.  For larger batches, each
    interval emits enough roots to contain its mean event load.  This is a
    deterministic design envelope; stochastic interval-level anchor counts
    are reported separately by the temporal model.
    """

    if batch_size <= 0 or batch_window_seconds <= 0:
        raise ValueError("batch_size and batch_window_seconds must be positive")
    if batch_size == 1:
        return daily_events, 1.0
    intervals = math.ceil(SECONDS_PER_DAY / batch_window_seconds)
    mean_events_per_interval = daily_events / intervals
    anchors_per_interval = max(1, math.ceil(mean_events_per_interval / batch_size))
    anchors_per_day = min(daily_events, intervals * anchors_per_interval)
    fill_ratio = daily_events / (anchors_per_day * batch_size) if anchors_per_day else 0.0
    return float(anchors_per_day), float(fill_ratio)


def merkle_membership_bytes(batch_size: int) -> int:
    if batch_size <= 1:
        return 0
    return math.ceil(math.log2(batch_size)) * SHA256_BYTES


# ---------------------------------------------------------------------------
# Auditable record sizes and one shared storage-accounting implementation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditReceipt:
    auditor_id: bytes
    event_id: bytes
    request_id: bytes
    declaration_identity: bytes
    witness_digest: bytes
    inspected_at_ms: int
    outcome: int

    def encode(self) -> bytes:
        body = b"".join(
            checked_bytes(getattr(self, n), 16, n) for n in ("auditor_id", "event_id", "request_id")
        )
        body += checked_bytes(self.declaration_identity, 32, "declaration_identity") + checked_bytes(
            self.witness_digest, 32, "witness_digest"
        )
        body += struct.pack(
            ">QB",
            checked_uint(self.inspected_at_ms, 64, "inspected_at_ms"),
            checked_uint(self.outcome, 8, "outcome"),
        )
        return frame_record(10, body)

    @classmethod
    def decode(cls, data: bytes) -> AuditReceipt:
        r = unframe_record(data, 10)
        value = cls(r.take(16), r.take(16), r.take(16), r.take(32), r.take(32), r.uint(64), r.uint(8))
        r.finish()
        if value.encode() != data:
            raise EvidenceFormatError("noncanonical audit receipt")
        return value

    def verify(self, signature: bytes, trust: TrustRegistry) -> bool:
        return trust.verifies(
            self.auditor_id, "auditor", b"HTGL-AUDIT-RECEIPT-v2\x00", self.encode(), signature
        )


def size_reference_records() -> dict[str, Any]:
    origin, custodian, auditor = (
        reference_id("origin"),
        reference_id("custodian-0"),
        reference_id("auditor-A"),
    )
    m = EvidenceMetadata(
        reference_id("event"),
        reference_id("tenant"),
        1,
        (origin, reference_id("requester")),
        reference_id("resource"),
        reference_id("policy"),
        1000,
        1,
    )
    leaf, commitment = bytes(32), bytes(32)
    request = ErasureRequest(
        reference_id("request"),
        m.tenant_id,
        m.event_id,
        leaf,
        m.resource_id,
        m.policy_id,
        reference_id("authority"),
        tuple(sorted((custodian, reference_id("custodian-1")))),
        7,
        2000,
        4000,
    )
    tombstone = Tombstone(
        m.tenant_id, m.event_id, leaf, request.request_id, custodian, m.resource_id, 7, 3000
    )
    anchor = AnchorRecord(
        reference_id("ledger"),
        reference_id("batch"),
        leaf,
        1000,
        2000,
        1,
        ZERO_HASH,
        reference_id("anchor"),
    )
    certificate = FinalizedAnchor(
        anchor, bytes(64), tuple((reference_id(f"validator-{i}"), bytes(64)) for i in range(3))
    )
    receipt = AuditReceipt(
        auditor, m.event_id, request.request_id, tombstone.identity(), bytes(32), 3500, 0
    )
    return {
        "metadata": m,
        "request": request,
        "tombstone": tombstone,
        "anchor": certificate,
        "receipt": receipt,
        "generic_bytes": len(m.encode()) + len(commitment) + len(leaf) + 64,
        "request_bytes": len(request.encode()) + 64,
        "tombstone_bytes": len(tombstone.encode()) + 64,
        "receipt_bytes": len(receipt.encode()) + 64,
        "anchor_bytes": len(certificate.encode()),
    }


def record_size_profiles() -> dict[str, dict[str, Any]]:
    wire = size_reference_records()
    return {
        "paper_reference_envelope": {
            "generic": EVIDENCE_SIZES.reduced_evidence_bytes,
            "request": EVIDENCE_SIZES.reduced_evidence_bytes,
            "tombstone": EVIDENCE_SIZES.tombstone_bytes,
            "receipt": EVIDENCE_SIZES.audit_receipt_bytes,
            "anchor": EVIDENCE_SIZES.anchor_receipt_bytes,
            "classification": "legacy assumed sizing envelope; not measured protocol-v2 serialization",
        },
        "canonical_protocol_v2": {
            "generic": wire["generic_bytes"],
            "request": wire["request_bytes"],
            "tombstone": wire["tombstone_bytes"],
            "receipt": wire["receipt_bytes"],
            "anchor": wire["anchor_bytes"],
            "classification": "actual binary codec lengths; two actors, two declared custodians, quorum=3; no storage-engine amplification",
        },
    }


RETENTION_MEASUREMENTS: pd.DataFrame | None = None


@lru_cache(maxsize=256)
def witness_cost_lookup(capacity: int, policy: str) -> np.ndarray:
    costs = np.zeros(capacity + 1, dtype=float)
    for n in range(1, capacity + 1):
        depth = (n - 1).bit_length()
        if policy == "independent_digest_paths":
            costs[n] = n * 32 * (capacity - 1).bit_length()
        elif policy == "independent_encoded_paths":
            costs[n] = 19 + n * (37 + 32 * depth)
        elif policy == "leaves_on_demand":
            costs[n] = 31 + 32 * n
        elif policy == "shared_tree_nodes":
            width, nodes, levels = n, n, 1
            while width > 1:
                width = (width + 1) // 2
                nodes += width
                levels += 1
            costs[n] = 33 + 4 * levels + 32 * nodes
        elif policy == "compressed_path_bundle":
            if RETENTION_MEASUREMENTS is None:
                raise RuntimeError(
                    "run retention_policy_experiments before compressed storage accounting"
                )
            samples = RETENTION_MEASUREMENTS[RETENTION_MEASUREMENTS.policy == policy].sort_values(
                "batch_size"
            )
            # Compression is measured, not treated as a cryptographic size
            # identity. Partial sizes between measured points interpolate.
            costs[n] = np.interp(n, samples.batch_size, samples.retained_bytes_per_batch)
        else:
            raise ValueError("unsupported witness storage policy")
    return costs


def witness_storage_bytes(
    daily_events: float | np.ndarray, capacity: int, window_seconds: int, policy: str
) -> float | np.ndarray:
    values = np.asarray(daily_events, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("nonnegative finite daily demand required")
    if policy == "independent_digest_paths":
        result = values * merkle_membership_bytes(capacity)
    else:
        windows = math.ceil(SECONDS_PER_DAY / window_seconds)
        mean = values / windows
        lower = np.floor(mean).astype(np.int64)
        fraction = mean - lower
        costs = witness_cost_lookup(capacity, policy)

        def per_window(n: np.ndarray) -> np.ndarray:
            full, remainder = n // capacity, n % capacity
            return full * costs[capacity] + costs[remainder]

        result = windows * ((1 - fraction) * per_window(lower) + fraction * per_window(lower + 1))
    return float(result) if result.ndim == 0 else result


def retained_storage_components(
    rates: Mapping[str, Any],
    batch_size: int = 1000,
    window_seconds: int = 60,
    days: float = 1,
    replicas: int = 1,
    profile: str = "paper_reference_envelope",
    witness_policy: str = "independent_digest_paths",
    generic_override: int | None = None,
    class_retention_days: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """No descendant double count; paths and anchors counted exactly once.

    Periodic-window costs use adjacent integer occupancies preserving the
    specified mean. Event-driven closure is evaluated separately. Encoded
    representations account for each partial batch's actual depth.
    """
    if days < 0 or type(replicas) is not int or replicas < 1:
        raise ValueError("invalid retention/replication configuration")
    sizes = dict(record_size_profiles()[profile])
    if generic_override is not None:
        sizes["generic"] = checked_uint(generic_override, 32, "generic_override")
        if profile == "paper_reference_envelope":
            sizes["request"] = generic_override
    counts = {key: np.asarray(rates[key], dtype=float) for key in COMPONENT_COLUMNS}
    if any(np.any(~np.isfinite(v)) or np.any(v < 0) for v in counts.values()):
        raise ValueError("event-class rates must be finite and nonnegative")
    total = sum(counts.values())
    retained = {
        key: float(class_retention_days[key]) if class_retention_days is not None else float(days)
        for key in COMPONENT_COLUMNS
    }
    if any(not math.isfinite(h) or h < 0 for h in retained.values()):
        raise ValueError("invalid event-class retention")
    generic = sum(counts[key] * retained[key] for key in COMPONENT_COLUMNS[:4]) * sizes["generic"]
    origin_hold_event_days = np.zeros_like(total)
    if class_retention_days is not None:
        # A retained request requires its linked origin and membership proof.
        # Charge one distinct protected origin per request, capped by observed
        # operational event rate and allocated by the modeled class mix.
        roots = sum(counts[key] for key in COMPONENT_COLUMNS[:4])
        protected = np.minimum(counts["Re_Erasure"], roots)
        case_horizon = max(retained[key] for key in COMPONENT_COLUMNS[4:])
        for key in COMPONENT_COLUMNS[:4]:
            share = np.divide(counts[key], roots, out=np.zeros_like(total), where=roots > 0)
            origin_hold_event_days += protected * share * max(0.0, case_horizon - retained[key])
        generic += origin_hold_event_days * sizes["generic"]
    request = counts["Re_Erasure"] * retained["Re_Erasure"] * sizes["request"]
    tombstone = counts["Rt_Tombstone"] * retained["Rt_Tombstone"] * sizes["tombstone"]
    receipt = counts["Rv_AuditVerification"] * retained["Rv_AuditVerification"] * sizes["receipt"]
    intervals = math.ceil(SECONDS_PER_DAY / window_seconds)
    anchors = np.minimum(total, intervals * np.maximum(1, np.ceil((total / intervals) / batch_size)))
    if class_retention_days is not None and witness_policy != "independent_digest_paths":
        # A mixed-event shared tree must remain available for the longest
        # retained class. Charging each class its own tree lifetime would
        # underestimate shared-node and on-demand retention.
        proof = witness_storage_bytes(total, batch_size, window_seconds, witness_policy) * max(
            retained.values()
        )
    elif class_retention_days is not None:
        proof = (
            sum(counts[key] * retained[key] for key in COMPONENT_COLUMNS) + origin_hold_event_days
        ) * merkle_membership_bytes(batch_size)
    else:
        proof = witness_storage_bytes(total, batch_size, window_seconds, witness_policy) * days
    anchor = (
        anchors
        * sizes["anchor"]
        * (max(retained.values()) if class_retention_days is not None else days)
    )
    result = {
        "generic_record_bytes": generic,
        "request_record_bytes": request,
        "tombstone_record_bytes": tombstone,
        "verification_receipt_bytes": receipt,
        "membership_witness_bytes": proof,
        "anchor_receipt_bytes": anchor,
    }
    result = {k: v * replicas for k, v in result.items()}
    result["total_bytes"] = sum(result.values())
    return {k: float(v) if np.ndim(v) == 0 else v for k, v in result.items()}


def export_size_derivations() -> None:
    rows = []
    legacy = EVIDENCE_SIZES
    for field, size in asdict(legacy).items():
        rows.append(
            {
                "profile": "paper_reference_envelope",
                "record": "generic",
                "field": field,
                "bytes": size,
                "basis": "standards-derived widths or explicit legacy design allowance",
                "measured_serialization": False,
            }
        )
    for record, fields in {
        "tombstone": [
            ("event_id", 16),
            ("leaf_reference", 32),
            ("scope", 4),
            ("timestamp", 8),
            ("custodian", 16),
            ("signature", 64),
            ("encoding_allowance", 64),
        ],
        "receipt": [
            ("event_id", 16),
            ("type", 2),
            ("timestamp", 8),
            ("auditor", 16),
            ("signature", 64),
            ("encoding_allowance", 64),
        ],
        "anchor": [
            ("batch_id", 16),
            ("root", 32),
            ("timestamp", 8),
            ("sequence", 8),
            ("signer", 16),
            ("signature", 64),
            ("encoding_allowance", 64),
        ],
    }.items():
        for field, size in fields:
            rows.append(
                {
                    "profile": "paper_reference_envelope",
                    "record": record,
                    "field": field,
                    "bytes": size,
                    "basis": "explicit legacy field envelope",
                    "measured_serialization": False,
                }
            )
    wire = size_reference_records()
    records = {
        "generic": [
            ("frame", 11),
            ("event_id", 16),
            ("tenant_id", 16),
            ("event_type", 2),
            ("actor_count", 2),
            ("actors_two", 32),
            ("resource_id", 16),
            ("policy_id", 16),
            ("timestamp_ms", 8),
            ("sequence", 8),
            ("payload_commitment", 32),
            ("redundant_leaf", 32),
            ("origin_signature", 64),
        ],
        "request": [
            ("frame", 11),
            ("request_id", 16),
            ("tenant_id", 16),
            ("event_id", 16),
            ("origin_leaf", 32),
            ("resource_id", 16),
            ("policy_id", 16),
            ("authority_id", 16),
            ("custodian_count", 2),
            ("custodians_two", 32),
            ("scope", 4),
            ("requested_at_ms", 8),
            ("expires_at_ms", 8),
            ("version", 4),
            ("signature", 64),
        ],
        "tombstone": [
            ("frame", 11),
            ("tenant_id", 16),
            ("event_id", 16),
            ("origin_leaf", 32),
            ("request_id", 16),
            ("custodian_id", 16),
            ("resource_id", 16),
            ("scope", 4),
            ("erased_at_ms", 8),
            ("version", 4),
            ("signature", 64),
        ],
        "receipt": [
            ("frame", 11),
            ("auditor_id", 16),
            ("event_id", 16),
            ("request_id", 16),
            ("declaration_identity", 32),
            ("witness_digest", 32),
            ("inspected_at_ms", 8),
            ("outcome", 1),
            ("signature", 64),
        ],
        "anchor": [
            ("certificate_frame", 11),
            ("anchor_frame", 11),
            ("ledger_id", 16),
            ("batch_id", 16),
            ("root", 32),
            ("leaf_count", 4),
            ("created_at_ms", 8),
            ("height", 8),
            ("previous_digest", 32),
            ("signer_id", 16),
            ("anchor_signature", 64),
            ("vote_count", 2),
            ("quorum_three_validator_ids", 48),
            ("quorum_three_signatures", 192),
        ],
    }
    measured = {
        "generic": wire["generic_bytes"],
        "request": wire["request_bytes"],
        "tombstone": wire["tombstone_bytes"],
        "receipt": wire["receipt_bytes"],
        "anchor": wire["anchor_bytes"],
    }
    for record, fields in records.items():
        if sum(size for _, size in fields) != measured[record]:
            raise RuntimeError(f"Canonical {record} byte derivation disagrees with codec")
        for field, size in fields:
            rows.append(
                {
                    "profile": "canonical_protocol_v2",
                    "record": record,
                    "field": field,
                    "bytes": size,
                    "basis": "actual versioned binary codec; hashes/signatures retained as raw bytes",
                    "measured_serialization": True,
                }
            )
    write_csv(pd.DataFrame(rows), DATA / "record_size_field_breakdown.csv")
    totals = []
    for profile, sizes in record_size_profiles().items():
        for record in ("generic", "request", "tombstone", "receipt", "anchor"):
            totals.append(
                {
                    "profile": profile,
                    "record": record,
                    "bytes": sizes[record],
                    "classification": sizes["classification"],
                }
            )
    write_csv(pd.DataFrame(totals), DATA / "record_size_totals.csv")


def batching_and_storage_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    global RETENTION_MEASUREMENTS
    export_size_derivations()
    RETENTION_MEASUREMENTS = retention_policy_experiments()
    witness_cost_lookup.cache_clear()
    batching_rows, storage_rows, proof_rows, policy_rows, retention_rows = [], [], [], [], []
    for capacity in BATCH_SIZES:
        proof_rows.append(
            {
                "batch_size": capacity,
                "tree_height_upper_bound": (capacity - 1).bit_length(),
                "merkle_membership_bytes": merkle_membership_bytes(capacity),
                "encoded_membership_bytes": 37 + merkle_membership_bytes(capacity),
                "post_erasure_proof_package_bytes": EVIDENCE_SIZES.reduced_evidence_bytes
                + EVIDENCE_SIZES.anchor_receipt_bytes
                + EVIDENCE_SIZES.tombstone_bytes
                + merkle_membership_bytes(capacity),
                "size_interpretation": "legacy one-origin core-witness envelope; worked_witness.json measures full authorized-request v2 package",
                "erased_payload_bytes_required_by_verifier": 0,
            }
        )
    for population in REPRESENTED_POPULATIONS:
        for scenario in SCENARIOS:
            rates = analytical_component_rates(population, scenario)
            events = rates["Daily_Events"]
            for capacity in BATCH_SIZES:
                for window in BATCH_WINDOWS_SECONDS:
                    anchors, fill = deterministic_anchor_count(events, capacity, window)
                    batching_rows.append(
                        {
                            "population": population,
                            "population_millions": population / 1e6,
                            "scenario": scenario.name,
                            "daily_logical_events": events,
                            "logical_event_tps": events / SECONDS_PER_DAY,
                            "batch_size": capacity,
                            "max_batch_wait_seconds": window,
                            "anchor_transactions_day": anchors,
                            "anchor_tps": anchors / SECONDS_PER_DAY,
                            "mean_batch_fill_ratio": fill,
                            "consensus_reduction_vs_event_per_tx": events / anchors if anchors else 0,
                            "merkle_membership_bytes": merkle_membership_bytes(capacity),
                            "estimator": "deterministic periodic-window design envelope; not temporal sample mean",
                        }
                    )
            for record_bytes in RECORD_SIZE_PROFILES_BYTES:
                for days in (1, 30, 365):
                    c = retained_storage_components(rates, days=days, generic_override=record_bytes)
                    storage_rows.append(
                        {
                            "population": population,
                            "population_millions": population / 1e6,
                            "scenario": scenario.name,
                            "days": days,
                            "logical_events": events * days,
                            "telemetry_records": population * METER_READINGS_PER_CONSUMER_DAY * days,
                            "record_size_bytes": record_bytes,
                            "size_profile": "paper_reference_envelope",
                            "witness_policy": "independent_digest_paths",
                            "reduced_evidence_storage_gb": (
                                c["generic_record_bytes"] + c["request_record_bytes"]
                            )
                            / 1e9,
                            "batch_anchor_storage_gb": c["anchor_receipt_bytes"] / 1e9,
                            "tombstone_storage_gb": c["tombstone_record_bytes"] / 1e9,
                            "audit_receipt_storage_gb": c["verification_receipt_bytes"] / 1e9,
                            "membership_storage_gb": c["membership_witness_bytes"] / 1e9,
                            "total_retained_storage_gb": c["total_bytes"] / 1e9,
                            "replication_factor": 1,
                            "erased_payload_storage_gb": 0.0,
                            "merkle_path_share_pct": 100
                            * c["membership_witness_bytes"]
                            / c["total_bytes"],
                        }
                    )
            for profile in record_size_profiles():
                policies = (
                    ("independent_digest_paths", *WitnessRepository.POLICIES)
                    if profile == "paper_reference_envelope"
                    else WitnessRepository.POLICIES
                )
                for policy in policies:
                    for days in (1, 30, 365):
                        for replicas in (1, 3, 5):
                            c = retained_storage_components(
                                rates,
                                days=days,
                                replicas=replicas,
                                profile=profile,
                                witness_policy=policy,
                            )
                            policy_rows.append(
                                {
                                    "population": population,
                                    "scenario": scenario.name,
                                    "size_profile": profile,
                                    "witness_policy": policy,
                                    "days": days,
                                    "replication_factor": replicas,
                                    **{k.removesuffix("_bytes") + "_gb": v / 1e9 for k, v in c.items()},
                                    "units": "decimal GB",
                                    "compression_interpretation": (
                                        "measured full-batch fixtures with interpolated partial sizes"
                                        if policy == "compressed_path_bundle"
                                        else "exact codec representation cost under deterministic-window occupancy model"
                                    ),
                                }
                            )
                selective = {
                    name: 30.0 if name in COMPONENT_COLUMNS[:4] else 365.0 for name in COMPONENT_COLUMNS
                }
                for policy in ("independent_digest_paths", "shared_tree_nodes", "leaves_on_demand"):
                    c = retained_storage_components(
                        rates, profile=profile, witness_policy=policy, class_retention_days=selective
                    )
                    retention_rows.append(
                        {
                            "population": population,
                            "scenario": scenario.name,
                            "size_profile": profile,
                            "witness_policy": policy,
                            "root_operational_retention_days": 30,
                            "erasure_descendant_retention_days": 365,
                            "linked_origin_policy": "one distinct protected operational origin per request, capped at root demand",
                            "shared_nodes_retained_to_maximum_class_horizon": policy
                            != "independent_digest_paths",
                            "total_storage_gb": c["total_bytes"] / 1e9,
                            "membership_storage_gb": c["membership_witness_bytes"] / 1e9,
                        }
                    )
    batching, storage = pd.DataFrame(batching_rows), pd.DataFrame(storage_rows)
    write_csv(batching, DATA / "batching_analysis.csv")
    write_csv(storage, DATA / "storage_detailed.csv")
    write_csv(pd.DataFrame(proof_rows), DATA / "proof_size_analysis.csv")
    write_csv(pd.DataFrame(policy_rows), DATA / "storage_policy_comparison.csv")
    write_csv(pd.DataFrame(retention_rows), DATA / "event_class_retention_comparison.csv")
    baseline = analytical_component_rates(REPRESENTED_POPULATIONS[0], SCENARIOS[0])
    write_csv(
        pd.DataFrame(
            [
                {
                    "days": d,
                    "events": baseline["Daily_Events"] * d,
                    "storage_gb_296B": retained_storage_components(baseline, days=d)["total_bytes"]
                    / 1e9,
                    "storage_gb_500B": retained_storage_components(
                        baseline, days=d, generic_override=500
                    )["total_bytes"]
                    / 1e9,
                    "includes_membership_paths": True,
                }
                for d in (1, 30, 365)
            ]
        ),
        DATA / "storage.csv",
    )
    return batching, storage


# ---------------------------------------------------------------------------
# Correlated temporal model (MMPP + diurnal modulation) and queue envelopes
# ---------------------------------------------------------------------------


MMPP_STATE_NAMES = np.array(("normal", "activity_peak", "coordinated_response", "incident"))
MMPP_RAW_MULTIPLIERS = np.array((0.75, 1.25, 2.00, 3.00), dtype=float)
MMPP_TRANSITION = np.array(
    (
        (0.9950, 0.0040, 0.0008, 0.0002),
        (0.0500, 0.9480, 0.0015, 0.0005),
        (0.0500, 0.0200, 0.9280, 0.0020),
        (0.1000, 0.0500, 0.0500, 0.8000),
    ),
    dtype=float,
)


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("transition matrix must be square")
    if not np.allclose(transition.sum(axis=1), 1.0):
        raise ValueError("transition rows must sum to one")
    eigenvalues, eigenvectors = np.linalg.eig(transition.T)
    index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    stationary = np.real(eigenvectors[:, index])
    stationary = stationary / stationary.sum()
    if np.any(stationary < -1e-12):
        stationary = -stationary
        stationary = stationary / stationary.sum()
    return stationary


MMPP_STATIONARY = stationary_distribution(MMPP_TRANSITION)
MMPP_MULTIPLIERS = MMPP_RAW_MULTIPLIERS / float(
    np.dot(MMPP_STATIONARY, MMPP_RAW_MULTIPLIERS)
)


def sample_mmpp_states(
    length: int,
    generator: np.random.Generator,
    transition: np.ndarray = MMPP_TRANSITION,
    stationary: np.ndarray = MMPP_STATIONARY,
) -> np.ndarray:
    """Sample exact geometric dwell runs of the discrete Markov chain."""
    if length < 0:
        raise ValueError("state trace length must be nonnegative")
    states = np.empty(length, dtype=np.int8)
    if length == 0:
        return states
    count = len(stationary)
    exit_probability = 1.0 - np.diag(transition)
    conditional_exit = transition.copy()
    np.fill_diagonal(conditional_exit, 0.0)
    conditional_exit = np.divide(
        conditional_exit,
        exit_probability[:, None],
        out=np.zeros_like(conditional_exit),
        where=exit_probability[:, None] > 0,
    )
    exit_cdf = np.cumsum(conditional_exit, axis=1)
    exit_cdf[:, -1] = 1.0
    state = int(generator.choice(count, p=stationary))
    position = 0
    while position < length:
        probability = exit_probability[state]
        if probability <= 0:
            states[position:] = state
            break
        end = min(length, position + int(generator.geometric(probability)))
        states[position:end] = state
        position = end
        if position < length:
            state = int(np.searchsorted(exit_cdf[state], generator.random(), side="right"))
    return states


@dataclass
class TemporalSimulation:
    trace: pd.DataFrame
    full_events: np.ndarray
    measurement_start: int
    measurement_end: int
    workflow_metrics: dict[str, Any]


def shifted_geometric_delays(
    count: int,
    mean_delay_minutes: float,
    maximum_delay_minutes: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Draw non-negative whole-minute delays with the requested mean."""

    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if mean_delay_minutes <= 0.0:
        return np.zeros(count, dtype=np.int64)
    probability = 1.0 / (mean_delay_minutes + 1.0)
    delays = generator.geometric(probability, size=count).astype(np.int64) - 1
    return np.minimum(delays, maximum_delay_minutes)


def schedule_linked_children(
    parent_counts: np.ndarray,
    expected_children_per_parent: float,
    mean_delay_minutes: float,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Schedule child arrivals and retain their parent indexes for auditing."""

    parents = np.asarray(parent_counts, dtype=np.int64)
    parent_indexes = np.repeat(np.arange(len(parents), dtype=np.int64), parents)
    integer_part = int(math.floor(expected_children_per_parent))
    fractional_part = expected_children_per_parent - integer_part
    child_parent_indexes = np.repeat(parent_indexes, integer_part)
    if fractional_part > 0.0 and len(parent_indexes):
        fractional_mask = generator.random(len(parent_indexes)) < fractional_part
        child_parent_indexes = np.concatenate(
            (child_parent_indexes, parent_indexes[fractional_mask])
        )
    delays = shifted_geometric_delays(
        len(child_parent_indexes),
        mean_delay_minutes,
        WORKFLOW_ASSUMPTIONS.maximum_scheduled_delay_minutes,
        generator,
    )
    arrival_indexes = child_parent_indexes + delays
    within_horizon = arrival_indexes < len(parents)
    scheduled = np.bincount(
        arrival_indexes[within_horizon], minlength=len(parents)
    ).astype(np.int64)
    return scheduled, child_parent_indexes, delays, arrival_indexes


def simulate_temporal_day(
    rates: Mapping[str, float],
    scenario: ScenarioProfile,
    generator: np.random.Generator,
    model: str = "causal_workflow",
    explicit_burst: bool = False,
) -> TemporalSimulation:
    """Simulate a warm-up, measurement, and cooldown horizon.

    Both temporal models share the same synthetic MMPP/diurnal stress shape.
    The null model samples all event types independently.  The causal model
    samples five root processes and schedules tombstone/verification children
    from their erasure parents.  Only the middle day is reported, while the
    full horizon initializes queue backlog and supplies delayed carry-in.
    """

    if model not in ("independent_poisson_null", "causal_workflow"):
        raise ValueError(f"Unsupported temporal model: {model}")
    total_days = (
        WORKFLOW_ASSUMPTIONS.temporal_warmup_days
        + WORKFLOW_ASSUMPTIONS.temporal_measurement_days
        + WORKFLOW_ASSUMPTIONS.temporal_cooldown_days
    )
    total_minutes = total_days * MINUTES_PER_DAY
    minutes = np.arange(total_minutes)
    minute_of_day = minutes % MINUTES_PER_DAY
    states = sample_mmpp_states(total_minutes, generator)
    state_multiplier = MMPP_MULTIPLIERS[states]
    # The 30% diurnal amplitude is a transparent stress-shape assumption, not
    # an empirical national governance trace.  MMPP state modulation provides
    # correlated periods that dominate the negligible large-population Poisson
    # counting variance.
    diurnal = 1.0 + 0.30 * np.sin(
        2.0 * np.pi * (minute_of_day - 480) / MINUTES_PER_DAY
    )
    common_factor = state_multiplier * diurnal

    expected_components: dict[str, np.ndarray] = {}
    sampled_components: dict[str, np.ndarray] = {}
    sampled_names = COMPONENT_COLUMNS if model == "independent_poisson_null" else COMPONENT_COLUMNS[:5]
    for component in sampled_names:
        expected = float(rates[component]) / MINUTES_PER_DAY * common_factor
        if explicit_burst and component in ("Ra_Authorization", "Rm_DER"):
            expected = expected.copy()
            expected[(minute_of_day >= 720) & (minute_of_day < 780)] *= 2.0
        expected_components[component] = expected
        sampled_components[component] = generator.poisson(expected).astype(np.int64)

    workflow_metrics: dict[str, Any] = {
        "model": model,
        "erasure_parents_measurement_day": np.nan,
        "tombstone_children_from_measurement_parents": np.nan,
        "verification_children_from_measurement_parents": np.nan,
        "tombstone_delay_mean_minutes": np.nan,
        "tombstone_delay_p95_minutes": np.nan,
        "tombstone_delay_p99_minutes": np.nan,
        "tombstone_within_sla_fraction": np.nan,
        "verification_delay_mean_minutes": np.nan,
        "verification_delay_p95_minutes": np.nan,
        "verification_delay_p99_minutes": np.nan,
        "verification_within_sla_fraction": np.nan,
        "tombstone_carry_in_events": np.nan,
        "verification_carry_in_events": np.nan,
        "tombstone_spillover_events": np.nan,
        "verification_spillover_events": np.nan,
        "orphan_children_generated": np.nan,
        "causal_order_violations": np.nan,
    }
    measurement_start = WORKFLOW_ASSUMPTIONS.temporal_warmup_days * MINUTES_PER_DAY
    measurement_end = measurement_start + (
        WORKFLOW_ASSUMPTIONS.temporal_measurement_days * MINUTES_PER_DAY
    )

    if model == "causal_workflow":
        erasures = sampled_components["Re_Erasure"]
        tombstones, tombstone_parents, tombstone_delays, tombstone_arrivals = schedule_linked_children(
            erasures,
            RATE_ASSUMPTIONS.tombstones_per_erasure,
            WORKFLOW_ASSUMPTIONS.tombstone_mean_delay_minutes,
            generator,
        )
        verifications, verification_parents, verification_delays, verification_arrivals = schedule_linked_children(
            erasures,
            RATE_ASSUMPTIONS.audits_per_erasure * scenario.audit_multiplier,
            WORKFLOW_ASSUMPTIONS.verification_mean_delay_minutes,
            generator,
        )
        sampled_components["Rt_Tombstone"] = tombstones
        sampled_components["Rv_AuditVerification"] = verifications

        parent_mask_t = (tombstone_parents >= measurement_start) & (tombstone_parents < measurement_end)
        parent_mask_v = (verification_parents >= measurement_start) & (verification_parents < measurement_end)
        delays_t = tombstone_delays[parent_mask_t]
        delays_v = verification_delays[parent_mask_v]
        parents_in_day = int(erasures[measurement_start:measurement_end].sum())

        def delay_stat(values: np.ndarray, quantile: float | None = None) -> float:
            if not len(values):
                return 0.0
            return float(values.mean() if quantile is None else np.quantile(values, quantile))

        workflow_metrics.update(
            {
                "erasure_parents_measurement_day": parents_in_day,
                "tombstone_children_from_measurement_parents": int(parent_mask_t.sum()),
                "verification_children_from_measurement_parents": int(parent_mask_v.sum()),
                "tombstone_delay_mean_minutes": delay_stat(delays_t),
                "tombstone_delay_p95_minutes": delay_stat(delays_t, 0.95),
                "tombstone_delay_p99_minutes": delay_stat(delays_t, 0.99),
                "tombstone_within_sla_fraction": (
                    float(np.mean(delays_t <= WORKFLOW_ASSUMPTIONS.tombstone_sla_minutes))
                    if len(delays_t)
                    else 1.0
                ),
                "verification_delay_mean_minutes": delay_stat(delays_v),
                "verification_delay_p95_minutes": delay_stat(delays_v, 0.95),
                "verification_delay_p99_minutes": delay_stat(delays_v, 0.99),
                "verification_within_sla_fraction": (
                    float(np.mean(delays_v <= WORKFLOW_ASSUMPTIONS.verification_sla_minutes))
                    if len(delays_v)
                    else 1.0
                ),
                "tombstone_carry_in_events": int(
                    np.sum(
                        (tombstone_parents < measurement_start)
                        & (tombstone_arrivals >= measurement_start)
                        & (tombstone_arrivals < measurement_end)
                    )
                ),
                "verification_carry_in_events": int(
                    np.sum(
                        (verification_parents < measurement_start)
                        & (verification_arrivals >= measurement_start)
                        & (verification_arrivals < measurement_end)
                    )
                ),
                "tombstone_spillover_events": int(
                    np.sum(parent_mask_t & (tombstone_arrivals >= measurement_end))
                ),
                "verification_spillover_events": int(
                    np.sum(parent_mask_v & (verification_arrivals >= measurement_end))
                ),
                "orphan_children_generated": int(
                    np.count_nonzero(erasures[tombstone_parents] <= 0)
                    + np.count_nonzero(erasures[verification_parents] <= 0)
                ),
                "causal_order_violations": int(
                    np.sum(tombstone_arrivals < tombstone_parents)
                    + np.sum(verification_arrivals < verification_parents)
                ),
            }
        )

    full_events = sum(sampled_components[name] for name in COMPONENT_COLUMNS)
    selection = slice(measurement_start, measurement_end)
    frame = pd.DataFrame(
        {
            "minute": np.arange(measurement_end - measurement_start),
            "hour": np.arange(measurement_end - measurement_start) / 60.0,
            "mmpp_state": MMPP_STATE_NAMES[states[selection]],
            "state_multiplier": state_multiplier[selection],
            "diurnal_multiplier": diurnal[selection],
            "model": model,
        }
    )
    for component in COMPONENT_COLUMNS:
        frame[component] = sampled_components[component][selection]
    frame["Events"] = full_events[selection]
    frame["events_per_minute"] = frame["Events"]
    frame["TPS"] = frame["Events"] / SECONDS_PER_MINUTE
    nominal_expected = sum(
        float(rates[component]) / MINUTES_PER_DAY * common_factor[selection]
        for component in COMPONENT_COLUMNS
    )
    frame["nominal_expected_tps"] = nominal_expected / SECONDS_PER_MINUTE
    frame["reference_anchor_transactions"] = np.where(
        frame["Events"] > 0,
        np.ceil(frame["Events"] / WORKFLOW_ASSUMPTIONS.reference_batch_size),
        0,
    ).astype(np.int64)
    frame["reference_anchor_tps"] = frame["reference_anchor_transactions"] / SECONDS_PER_MINUTE
    return TemporalSimulation(frame, full_events, measurement_start, measurement_end, workflow_metrics)


def queue_metrics(
    arrivals_per_minute: Sequence[float],
    capacity_tps: float,
    measurement_start: int = 0,
    measurement_end: int | None = None,
) -> dict[str, float | bool]:
    if not math.isfinite(capacity_tps) or capacity_tps <= 0:
        raise ValueError("positive finite capacity required")
    service_per_minute = capacity_tps * SECONDS_PER_MINUTE
    arrivals_array = np.asarray(arrivals_per_minute, dtype=float)
    if arrivals_array.ndim != 1 or np.any(~np.isfinite(arrivals_array)) or np.any(arrivals_array < 0):
        raise ValueError("arrivals must be a one-dimensional nonnegative finite sequence")
    end = len(arrivals_array) if measurement_end is None else measurement_end
    if not 0 <= measurement_start < end <= len(arrivals_array):
        raise ValueError("Invalid queue measurement window")
    # Reflection at the running minimum is algebraically equivalent to the
    # Lindley recursion and keeps the full warm-up history.
    cumulative = np.cumsum(arrivals_array - service_per_minute)
    backlog_series = cumulative - np.minimum.accumulate(np.minimum(cumulative, 0.0))
    selected_arrivals = arrivals_array[measurement_start:end]
    selected_backlog = backlog_series[measurement_start:end]
    delay_seconds = selected_backlog / capacity_tps
    return {
        "warmup_minutes": measurement_start,
        "measurement_minutes": end - measurement_start,
        "initial_measurement_backlog_events": (
            float(backlog_series[measurement_start - 1]) if measurement_start else 0.0
        ),
        "mean_arrival_tps": float(selected_arrivals.mean() / SECONDS_PER_MINUTE),
        "peak_arrival_tps": float(selected_arrivals.max() / SECONDS_PER_MINUTE),
        "mean_utilization_pct": float(100.0 * selected_arrivals.mean() / service_per_minute),
        "minutes_with_backlog": int(np.count_nonzero(selected_backlog > 0)),
        "backlog_minute_fraction": float(np.mean(selected_backlog > 0)),
        "max_backlog_events": float(selected_backlog.max()),
        "ending_backlog_events": float(selected_backlog[-1]),
        "p95_queue_delay_seconds": float(np.quantile(delay_seconds, 0.95)),
        "p99_queue_delay_seconds": float(np.quantile(delay_seconds, 0.99)),
        "max_queue_delay_seconds": float(delay_seconds.max()),
        "mean_rate_below_capacity": bool(selected_arrivals.mean() < service_per_minute),
        "drained_by_end_of_day": bool(selected_backlog[-1] == 0.0),
    }


def bootstrap_quantile_interval(
    values: Sequence[float], quantile: float, *labels: object
) -> tuple[float, float]:
    """Percentile bootstrap over independent replication-level maxima."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("bootstrap requires at least two finite independent observations")
    rng = experiment_rng("replication_peak_bootstrap", quantile, *labels)
    estimates = []
    for start in range(0, PEAK_BOOTSTRAP_REPLICATIONS, 64):
        size = min(64, PEAK_BOOTSTRAP_REPLICATIONS - start)
        indices = rng.integers(0, len(data), size=(size, len(data)))
        estimates.extend(np.quantile(data[indices], quantile, axis=1))
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def add_peak_intervals(
    summary: pd.DataFrame, replicates: pd.DataFrame, keys: list[str], column: str
) -> pd.DataFrame:
    rows = []
    for labels, group in replicates.groupby(keys, sort=False):
        if not isinstance(labels, tuple):
            labels = (labels,)
        row = dict(zip(keys, labels))
        for q in (0.95, 0.99):
            low, high = bootstrap_quantile_interval(group[column], q, column, *labels)
            row[f"p{int(q * 100)}_daily_peak_ci95_lower"] = low
            row[f"p{int(q * 100)}_daily_peak_ci95_upper"] = high
        row["bootstrap_replications"] = PEAK_BOOTSTRAP_REPLICATIONS
        row["peak_estimand"] = (
            "quantile of replication-level maximum minute rate; percentile-bootstrap interval"
        )
        rows.append(row)
    return summary.merge(pd.DataFrame(rows), on=keys, validate="one_to_one")


def temporal_and_capacity_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    temporal_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    workflow_rows: list[dict[str, Any]] = []
    temporal_batching_rows: list[dict[str, Any]] = []
    baseline_trace: pd.DataFrame | None = None
    burst_trace: pd.DataFrame | None = None

    for population in REPRESENTED_POPULATIONS:
        for scenario in SCENARIOS:
            rates = analytical_component_rates(population, scenario)
            for model in ("independent_poisson_null", "causal_workflow"):
                print(
                    f"      Temporal {population // 1_000_000}M / {scenario.name} / {model}: {TEMPORAL_REPLICATIONS} days",
                    flush=True,
                )
                for replication in range(1, TEMPORAL_REPLICATIONS + 1):
                    simulation = simulate_temporal_day(
                        rates,
                        scenario,
                        experiment_rng("temporal", population, scenario.name, model, replication),
                        model=model,
                        explicit_burst=False,
                    )
                    trace = simulation.trace
                    events = trace["Events"].to_numpy(dtype=float)
                    daily_events = int(events.sum())
                    minute_variance = float(events.var(ddof=1))
                    minute_mean = float(events.mean())
                    temporal_rows.append(
                        {
                            "population": population,
                            "population_millions": population / 1e6,
                            "scenario": scenario.name,
                            "model": model,
                            "replication": replication,
                            "daily_events": daily_events,
                            "daily_mean_tps": daily_events / SECONDS_PER_DAY,
                            "minute_tps_p50": trace["TPS"].quantile(0.50),
                            "minute_tps_p95": trace["TPS"].quantile(0.95),
                            "minute_tps_p99": trace["TPS"].quantile(0.99),
                            "minute_tps_max": trace["TPS"].max(),
                            "peak_to_average_ratio": trace["TPS"].max() / trace["TPS"].mean(),
                            "minute_fano_factor": minute_variance / minute_mean,
                            "lag1_autocorrelation": lag_autocorrelation(events, 1),
                            "lag5_autocorrelation": lag_autocorrelation(events, 5),
                            "lag60_autocorrelation": lag_autocorrelation(events, 60),
                        }
                    )

                    if model == "causal_workflow":
                        workflow_rows.append(
                            {
                                "population": population,
                                "population_millions": population / 1e6,
                                "scenario": scenario.name,
                                "replication": replication,
                                **simulation.workflow_metrics,
                            }
                        )

                    for batch_size in BATCH_SIZES:
                        anchors = np.where(events > 0, np.ceil(events / batch_size), 0.0)
                        temporal_batching_rows.append(
                            {
                                "population": population,
                                "population_millions": population / 1e6,
                                "scenario": scenario.name,
                                "model": model,
                                "replication": replication,
                                "batch_size": batch_size,
                                "batch_window_seconds": 60,
                                "daily_logical_events": daily_events,
                                "daily_anchor_transactions": int(anchors.sum()),
                                "mean_anchor_tps": float(anchors.mean() / SECONDS_PER_MINUTE),
                                "p99_minute_anchor_tps": float(np.quantile(anchors / SECONDS_PER_MINUTE, 0.99)),
                                "max_minute_anchor_tps": float(np.max(anchors / SECONDS_PER_MINUTE)),
                                "mean_batch_fill_ratio": float(
                                    daily_events / (anchors.sum() * batch_size)
                                ) if anchors.sum() else 0.0,
                                "consensus_reduction_factor": float(
                                    daily_events / anchors.sum()
                                ) if anchors.sum() else np.nan,
                            }
                        )

                    full_anchor_events = np.where(
                        simulation.full_events > 0,
                        np.ceil(
                            simulation.full_events
                            / WORKFLOW_ASSUMPTIONS.reference_batch_size
                        ),
                        0.0,
                    )
                    service_planes = (
                        (
                            "unbatched_consensus_counterfactual",
                            simulation.full_events,
                            CAPACITY_ENVELOPES_TPS,
                        ),
                        (
                            "batched_consensus_reference",
                            full_anchor_events,
                            ANCHOR_CAPACITY_ENVELOPES_TPS,
                        ),
                    )
                    for service_plane, arrivals, capacities in service_planes:
                        for capacity in capacities:
                            queue_rows.append(
                                {
                                    "population": population,
                                    "population_millions": population / 1e6,
                                    "scenario": scenario.name,
                                    "model": model,
                                    "service_plane": service_plane,
                                    "reference_batch_size": (
                                        WORKFLOW_ASSUMPTIONS.reference_batch_size
                                        if service_plane == "batched_consensus_reference"
                                        else 1
                                    ),
                                    "replication": replication,
                                    "capacity_tps": capacity,
                                    "capacity_classification": (
                                        "published_external_configuration_reference"
                                        if capacity == LITERATURE_REFERENCE_CAPACITY_TPS
                                        else "analytical_capacity_envelope"
                                    ),
                                    **queue_metrics(
                                        arrivals,
                                        capacity,
                                        simulation.measurement_start,
                                        simulation.measurement_end,
                                    ),
                                }
                            )

                    if (
                        model == "causal_workflow"
                        and population == REPRESENTED_POPULATIONS[0]
                        and scenario.name == "baseline"
                        and replication == 1
                    ):
                        baseline_trace = trace.copy()
                        burst_trace = simulate_temporal_day(
                            rates,
                            scenario,
                            experiment_rng("temporal", "explicit_burst", population, scenario.name),
                            model="causal_workflow",
                            explicit_burst=True,
                        ).trace

    if baseline_trace is None or burst_trace is None:
        raise RuntimeError("Baseline temporal traces were not generated.")

    # Backward-compatible traces; the v7 plot pipeline also consumes them.
    write_csv(baseline_trace, DATA / "timeseries.csv")
    write_csv(burst_trace, DATA / "burst.csv")
    write_csv(
        pd.DataFrame(
            [
                {
                    "mean_tps": baseline_trace["TPS"].mean(),
                    "max_tps": baseline_trace["TPS"].max(),
                    "peak_to_average_ratio": baseline_trace["TPS"].max() / baseline_trace["TPS"].mean(),
                }
            ]
        ),
        DATA / "timeseries_summary.csv",
    )
    write_csv(
        pd.DataFrame(
            [
                {
                    "mean_tps": burst_trace["TPS"].mean(),
                    "max_tps": burst_trace["TPS"].max(),
                    "peak_to_average_ratio": burst_trace["TPS"].max() / burst_trace["TPS"].mean(),
                }
            ]
        ),
        DATA / "burst_summary.csv",
    )

    temporal_frame = pd.DataFrame(temporal_rows)
    queue_frame = pd.DataFrame(queue_rows)
    workflow_frame = pd.DataFrame(workflow_rows)
    temporal_batching_frame = pd.DataFrame(temporal_batching_rows)
    write_csv(temporal_frame, DATA / "temporal_replication_summary.csv")
    write_csv(queue_frame, DATA / "capacity_queue_metrics.csv")
    write_csv(workflow_frame, DATA / "workflow_temporal_metrics.csv")
    write_csv(temporal_batching_frame, DATA / "temporal_batching_metrics.csv")

    temporal_grouped = (
        temporal_frame.groupby(
            ["population", "population_millions", "scenario", "model"], as_index=False
        )
        .agg(
            replication_count=("replication", "count"),
            mean_daily_tps=("daily_mean_tps", "mean"),
            p95_of_daily_peak_tps=("minute_tps_max", lambda x: x.quantile(0.95)),
            p99_of_daily_peak_tps=("minute_tps_max", lambda x: x.quantile(0.99)),
            maximum_observed_tps=("minute_tps_max", "max"),
            mean_peak_to_average_ratio=("peak_to_average_ratio", "mean"),
            mean_minute_fano_factor=("minute_fano_factor", "mean"),
            mean_lag1_autocorrelation=("lag1_autocorrelation", "mean"),
            mean_lag60_autocorrelation=("lag60_autocorrelation", "mean"),
        )
    )
    temporal_grouped = add_peak_intervals(
        temporal_grouped,
        temporal_frame,
        ["population", "population_millions", "scenario", "model"],
        "minute_tps_max",
    )
    write_csv(temporal_grouped, DATA / "temporal_scenario_summary.csv")

    queue_grouped = (
        queue_frame.groupby(
            [
                "population",
                "population_millions",
                "scenario",
                "model",
                "service_plane",
                "reference_batch_size",
                "capacity_tps",
                "capacity_classification",
            ],
            as_index=False,
        )
        .agg(
            replications=("replication", "count"),
            mean_utilization_pct=("mean_utilization_pct", "mean"),
            any_backlog_replications=("minutes_with_backlog", lambda x: int(np.sum(np.asarray(x) > 0))),
            ending_backlog_replications=("ending_backlog_events", lambda x: int(np.sum(np.asarray(x) > 0))),
            p95_max_backlog_events=("max_backlog_events", lambda x: x.quantile(0.95)),
            p99_max_queue_delay_seconds=("max_queue_delay_seconds", lambda x: x.quantile(0.99)),
            worst_max_queue_delay_seconds=("max_queue_delay_seconds", "max"),
        )
    )
    queue_grouped["probability_any_backlog"] = (
        queue_grouped["any_backlog_replications"] / queue_grouped["replications"]
    )
    queue_grouped["probability_ending_backlog"] = (
        queue_grouped["ending_backlog_replications"] / queue_grouped["replications"]
    )
    any_intervals = [
        wilson_interval(int(successes), int(trials))
        for successes, trials in zip(
            queue_grouped["any_backlog_replications"], queue_grouped["replications"]
        )
    ]
    ending_intervals = [
        wilson_interval(int(successes), int(trials))
        for successes, trials in zip(
            queue_grouped["ending_backlog_replications"], queue_grouped["replications"]
        )
    ]
    queue_grouped["probability_any_backlog_ci95_lower"] = [item[0] for item in any_intervals]
    queue_grouped["probability_any_backlog_ci95_upper"] = [item[1] for item in any_intervals]
    queue_grouped["probability_ending_backlog_ci95_lower"] = [item[0] for item in ending_intervals]
    queue_grouped["probability_ending_backlog_ci95_upper"] = [item[1] for item in ending_intervals]
    write_csv(queue_grouped, DATA / "capacity_queue_summary.csv")

    batching_grouped = (
        temporal_batching_frame.groupby(
            [
                "population",
                "population_millions",
                "scenario",
                "model",
                "batch_size",
                "batch_window_seconds",
            ],
            as_index=False,
        )
        .agg(
            replications=("replication", "count"),
            mean_anchor_tps=("mean_anchor_tps", "mean"),
            p99_of_daily_max_anchor_tps=("max_minute_anchor_tps", lambda x: x.quantile(0.99)),
            mean_batch_fill_ratio=("mean_batch_fill_ratio", "mean"),
            mean_consensus_reduction_factor=("consensus_reduction_factor", "mean"),
        )
    )
    batching_grouped = add_peak_intervals(
        batching_grouped,
        temporal_batching_frame,
        ["population", "population_millions", "scenario", "model", "batch_size", "batch_window_seconds"],
        "max_minute_anchor_tps",
    )
    batching_grouped["estimator"] = "mean of temporal daily anchor rates; periodic one-minute windows"
    write_csv(batching_grouped, DATA / "temporal_batching_summary.csv")

    workflow_grouped = (
        workflow_frame.groupby(
            ["population", "population_millions", "scenario"], as_index=False
        )
        .agg(
            replications=("replication", "count"),
            mean_erasure_parents=("erasure_parents_measurement_day", "mean"),
            mean_tombstone_delay_minutes=("tombstone_delay_mean_minutes", "mean"),
            p99_tombstone_delay_minutes=("tombstone_delay_p99_minutes", lambda x: x.quantile(0.99)),
            mean_tombstone_within_sla_fraction=("tombstone_within_sla_fraction", "mean"),
            mean_verification_delay_minutes=("verification_delay_mean_minutes", "mean"),
            p99_verification_delay_minutes=("verification_delay_p99_minutes", lambda x: x.quantile(0.99)),
            mean_verification_within_sla_fraction=("verification_within_sla_fraction", "mean"),
            total_orphan_children=("orphan_children_generated", "sum"),
            total_causal_order_violations=("causal_order_violations", "sum"),
        )
    )
    write_csv(workflow_grouped, DATA / "workflow_temporal_summary.csv")

    mmpp_frame = pd.DataFrame(
        {
            "state": MMPP_STATE_NAMES,
            "raw_multiplier": MMPP_RAW_MULTIPLIERS,
            "stationary_probability": MMPP_STATIONARY,
            "normalized_multiplier": MMPP_MULTIPLIERS,
        }
    )
    write_csv(mmpp_frame, DATA / "mmpp_state_model.csv")
    return temporal_grouped, queue_grouped


# ---------------------------------------------------------------------------
# Uncertain-parameter experiment and partial rank correlation (PRCC)
# ---------------------------------------------------------------------------


UNCERTAINTY_RANGES = (
    ("consent_events_per_consumer_year", 0.5, 4.0, "uniform", "research uncertainty range"),
    ("authorization_events_per_consumer_day", 0.1, 1.0, "uniform", "research uncertainty range"),
    ("der_adoption", 0.05, 0.40, "uniform", "scenario uncertainty range"),
    ("der_coordination_events_per_active_der_day", 0.5, 4.0, "uniform", "research uncertainty range"),
    ("integrity_events_per_institution_day", 1.0, 24.0, "uniform", "research uncertainty range"),
    ("institutional_actors", 200.0, 2_000.0, "uniform", "research uncertainty range"),
    ("erasure_requests_per_consumer_year", 0.001, 0.05, "log_uniform", "research uncertainty range"),
    ("audits_per_erasure", 0.1, 2.0, "uniform", "research uncertainty range"),
)


def latin_hypercube(samples: int, dimensions: int, generator: np.random.Generator) -> np.ndarray:
    points = np.empty((samples, dimensions), dtype=float)
    for dimension in range(dimensions):
        points[:, dimension] = (generator.permutation(samples) + generator.random(samples)) / samples
    return points


def transform_unit_interval(values: np.ndarray, low: float, high: float, distribution: str) -> np.ndarray:
    if distribution == "uniform":
        return low + values * (high - low)
    if distribution == "log_uniform":
        return np.exp(np.log(low) + values * (np.log(high) - np.log(low)))
    raise ValueError(f"Unsupported distribution: {distribution}")


def residualize(target: np.ndarray, controls: np.ndarray) -> np.ndarray:
    if controls.size == 0:
        return target - target.mean()
    design = np.column_stack((np.ones(len(target)), controls))
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return target - design @ coefficients


def partial_rank_correlations(frame: pd.DataFrame, inputs: Sequence[str], outcome: str) -> pd.DataFrame:
    ranked = frame[list(inputs) + [outcome]].rank(method="average").to_numpy(dtype=float)
    x = ranked[:, : len(inputs)]
    y = ranked[:, -1]
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(inputs):
        other_indexes = [j for j in range(len(inputs)) if j != index]
        controls = x[:, other_indexes]
        x_residual = residualize(x[:, index], controls)
        y_residual = residualize(y, controls)
        coefficient = float(np.corrcoef(x_residual, y_residual)[0, 1])
        rows.append(
            {
                "input_parameter": name,
                "outcome": outcome,
                "prcc": coefficient,
                "absolute_prcc": abs(coefficient),
            }
        )
    return pd.DataFrame(rows).sort_values("absolute_prcc", ascending=False)


def uncertainty_and_sensitivity_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    input_names = [entry[0] for entry in UNCERTAINTY_RANGES]
    unit = latin_hypercube(
        UNCERTAINTY_RUNS_PER_POPULATION,
        len(input_names),
        experiment_rng("latin_hypercube_uncertainty"),
    )
    sampled: dict[str, np.ndarray] = {}
    for index, (name, low, high, distribution, _) in enumerate(UNCERTAINTY_RANGES):
        sampled[name] = transform_unit_interval(unit[:, index], low, high, distribution)

    samples_by_population: list[pd.DataFrame] = []
    prcc_frames: list[pd.DataFrame] = []
    uncertainty_summary_rows: list[dict[str, Any]] = []
    for population in REPRESENTED_POPULATIONS:
        rc = population * sampled["consent_events_per_consumer_year"] / 365.0
        ra = population * sampled["authorization_events_per_consumer_day"]
        rm = (
            population
            * sampled["der_adoption"]
            * sampled["der_coordination_events_per_active_der_day"]
        )
        ri = sampled["institutional_actors"] * sampled["integrity_events_per_institution_day"]
        re = population * sampled["erasure_requests_per_consumer_year"] / 365.0
        rt = re
        rv = re * sampled["audits_per_erasure"]
        daily_events = rc + ra + rm + ri + re + rt + rv

        population_frame = pd.DataFrame(sampled)
        population_frame.insert(0, "sample", np.arange(1, len(population_frame) + 1))
        population_frame.insert(1, "population", population)
        population_frame["daily_events"] = daily_events
        population_frame["logical_event_tps"] = daily_events / SECONDS_PER_DAY
        # Reference design: 1,000-item batches emitted at least every 60s.
        intervals = SECONDS_PER_DAY // 60
        anchors_interval = np.maximum(1.0, np.ceil((daily_events / intervals) / 1_000.0))
        population_frame["reference_anchor_tps"] = anchors_interval * intervals / SECONDS_PER_DAY
        sampled_rates = dict(zip(COMPONENT_COLUMNS, (rc, ra, rm, ri, re, rt, rv)))
        legacy = retained_storage_components(sampled_rates)
        population_frame["retained_storage_gb_day_296B"] = legacy["total_bytes"] / 1e9
        population_frame["membership_storage_gb_day_296B"] = legacy["membership_witness_bytes"] / 1e9
        for policy in WitnessRepository.POLICIES:
            result = retained_storage_components(
                sampled_rates, profile="canonical_protocol_v2", witness_policy=policy
            )
            population_frame[f"retained_storage_gb_day_v2_{policy}"] = result["total_bytes"] / 1e9
        samples_by_population.append(population_frame)

        for outcome in (
            "logical_event_tps",
            "reference_anchor_tps",
            "retained_storage_gb_day_296B",
        ):
            prcc = partial_rank_correlations(population_frame, input_names, outcome)
            prcc.insert(0, "population", population)
            prcc_frames.append(prcc)

        uncertainty_summary_rows.append(
            {
                "population": population,
                "population_millions": population / 1e6,
                "samples": len(population_frame),
                **descriptive_statistics(population_frame["logical_event_tps"], "logical_event_tps"),
                **descriptive_statistics(population_frame["reference_anchor_tps"], "reference_anchor_tps"),
                **descriptive_statistics(
                    population_frame["retained_storage_gb_day_296B"],
                    "retained_storage_gb_day_296B",
                ),
            }
        )

    uncertainty_frame = pd.concat(samples_by_population, ignore_index=True)
    prcc_frame = pd.concat(prcc_frames, ignore_index=True)
    write_csv(uncertainty_frame, DATA / "uncertainty_samples.csv")
    write_csv(pd.DataFrame(uncertainty_summary_rows), DATA / "uncertainty_summary.csv")
    write_csv(prcc_frame, DATA / "prcc_sensitivity.csv")
    write_csv(
        pd.DataFrame(
            [
                {
                    "parameter": name,
                    "lower": low,
                    "upper": high,
                    "distribution": distribution,
                    "classification": classification,
                }
                for name, low, high, distribution, classification in UNCERTAINTY_RANGES
            ]
        ),
        DATA / "uncertainty_parameter_ranges.csv",
    )

    # Backward-compatible deterministic population/DER sensitivity grid.
    grid_rows: list[dict[str, Any]] = []
    for consumers in np.linspace(10e6, 450e6, 12):
        for der_adoption in np.linspace(0.05, 0.40, 8):
            scenario = ScenarioProfile(
                "sensitivity",
                "Sensitivity",
                1.0,
                1.0,
                float(der_adoption),
                1.0,
                1.0,
                "Two-dimensional deterministic sensitivity grid.",
            )
            rates = analytical_component_rates(int(consumers), scenario)
            grid_rows.append(
                {
                    "consumers": int(consumers),
                    "consumer_millions": consumers / 1e6,
                    "der_adoption": der_adoption,
                    "TPS": rates["TPS"],
                }
            )
    grid = pd.DataFrame(grid_rows)
    write_csv(grid, DATA / "sensitivity.csv")
    write_csv(grid, DATA / "der_adoption_sensitivity.csv")
    return uncertainty_frame, prcc_frame


def export_workload_rationale() -> None:
    rows = [
        (
            "consent_events_per_consumer_year",
            2.0,
            "Renewal and change events, not individual meter reads",
            "0.5–4/year covers infrequent to quarterly evidence creation",
        ),
        (
            "authorization_events_per_consumer_day",
            0.5,
            "One retained authorization event per two participant-days on average",
            "0.1–1/day tests sparsely logged to daily authorization",
        ),
        (
            "der_adoption",
            0.10,
            "Ten percent of the represented population is active in this reference scenario",
            "0.05–0.40 tests low to substantial adoption; 0.25 is the high-DER stress setting",
        ),
        (
            "der_coordination_events_per_active_der_day",
            1.0,
            "One governance coordination record per active participant-day",
            "0.5–4/day tests variation in the granularity of recorded decisions",
        ),
        (
            "integrity_events_per_institution_day",
            24.0,
            "Hourly institutional check scenario",
            "1–24/day spans daily to hourly checks",
        ),
        (
            "institutional_actors",
            500.0,
            "Fixed institutional population separates organization-level from participant-level demand",
            "200–2,000 tests different federation sizes",
        ),
        (
            "erasure_requests_per_consumer_year",
            0.01,
            "Rare request scenario of one request per hundred participant-years",
            "0.001–0.05/year spans orders of magnitude on a log scale",
        ),
        (
            "audits_per_erasure",
            1.0,
            "One optional verification receipt per request in the baseline",
            "0.1–2 models fractional participation and multiple receipts",
        ),
        (
            "tombstones_per_erasure",
            1.0,
            "Reference workload charges one declaration per request",
            "Separate protocol experiment tests multiple declared custodians; workload sensitivity preserves the reference multiplicity",
        ),
        (
            "tombstone_mean_delay_minutes",
            5.0,
            "Short logical declaration delay in the synthetic workflow",
            "Scenario input; application-store and consensus latency are measured separately only in the local reference experiment",
        ),
        (
            "verification_mean_delay_minutes",
            60.0,
            "Longer optional verification-receipt delay",
            "Scenario input; no national service-level trace supplied",
        ),
    ]
    result = pd.DataFrame(
        rows, columns=["parameter", "baseline", "scenario_interpretation", "range_rationale"]
    )
    result["evidence_class"] = "declared design hypothesis"
    write_csv(result, DATA / "workload_assumption_rationale.csv")
    write_csv(
        pd.DataFrame(
            [
                {
                    "state": state,
                    "self_transition_probability": float(MMPP_TRANSITION[i, i]),
                    "mean_geometric_dwell_minutes": 1.0 / (1.0 - MMPP_TRANSITION[i, i]),
                    "relative_intensity": MMPP_RAW_MULTIPLIERS[i],
                    "interpretation": "declared persistent minute-level stress regime",
                }
                for i, state in enumerate(MMPP_STATE_NAMES)
            ]
        ),
        DATA / "burst_assumption_rationale.csv",
    )


# ---------------------------------------------------------------------------
# Versioned evidence protocol, authenticated anchors, and persistent audits
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = 2
DOMAIN_COMMITMENT = b"HTGL-ERASABLE-COMPONENT-v2\x00"
DOMAIN_LEAF = b"HTGL-STRUCTURED-EVIDENCE-v2\x00"
DOMAIN_MERKLE = b"HTGL-MERKLE-v2\x00"
DOMAIN_REQUEST = b"HTGL-ERASURE-REQUEST-v2\x00"
DOMAIN_ORIGIN_SIGNATURE = b"HTGL-ORIGIN-SIGNATURE-v2\x00"
DOMAIN_REQUEST_SIGNATURE = b"HTGL-REQUEST-SIGNATURE-v2\x00"
DOMAIN_TOMBSTONE_SIGNATURE = b"HTGL-TOMBSTONE-SIGNATURE-v2\x00"
DOMAIN_ANCHOR_SIGNATURE = b"HTGL-ANCHOR-SIGNATURE-v2\x00"
DOMAIN_PREPARE = b"HTGL-ANCHOR-PREPARE-v2\x00"
DOMAIN_COMMIT = b"HTGL-ANCHOR-COMMIT-v2\x00"
ZERO_HASH = b"\x00" * 32
OPENING_BYTES = 32
MAX_ACTORS = 64
MAX_JSON_PACKAGE_BYTES = 4 * 1024 * 1024


class EvidenceFormatError(ValueError):
    """A noncanonical, malformed, or unsupported evidence record."""


def checked_bytes(value: Any, size: int, field: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != size:
        raise EvidenceFormatError(f"{field} must be exactly {size} bytes")
    return value


def checked_uint(value: Any, bits: int, field: str) -> int:
    if type(value) is not int or not 0 <= value < (1 << bits):
        raise EvidenceFormatError(f"{field} must be an unsigned {bits}-bit integer")
    return value


def reference_id(label: str) -> bytes:
    """Public reproducible fixture references; these are not anonymous IDs."""
    return sha256_bytes(b"HTGL-FIXTURE-REFERENCE-v2\x00" + label.encode("utf-8"))[:16]


def frame_record(kind: int, body: bytes) -> bytes:
    return (
        b"HTGL" + struct.pack(">HBI", PROTOCOL_VERSION, checked_uint(kind, 8, "kind"), len(body)) + body
    )


class BinaryReader:
    def __init__(self, data: bytes):
        if not isinstance(data, bytes):
            raise EvidenceFormatError("record must be bytes")
        self.data, self.offset = data, 0

    def take(self, count: int) -> bytes:
        if count < 0 or self.offset + count > len(self.data):
            raise EvidenceFormatError("truncated canonical record")
        result = self.data[self.offset : self.offset + count]
        self.offset += count
        return result

    def uint(self, bits: int) -> int:
        return int.from_bytes(self.take(bits // 8), "big")

    def finish(self) -> None:
        if self.offset != len(self.data):
            raise EvidenceFormatError("trailing bytes in canonical record")


def unframe_record(data: bytes, expected_kind: int) -> BinaryReader:
    reader = BinaryReader(data)
    if reader.take(4) != b"HTGL" or reader.uint(16) != PROTOCOL_VERSION:
        raise EvidenceFormatError("unsupported protocol magic/version")
    if reader.uint(8) != expected_kind:
        raise EvidenceFormatError("wrong record type")
    body_length = reader.uint(32)
    body = reader.take(body_length)
    reader.finish()
    return BinaryReader(body)


@dataclass(frozen=True)
class EvidenceMetadata:
    event_id: bytes
    tenant_id: bytes
    event_type: int
    actors: tuple[bytes, ...]
    resource_id: bytes
    policy_id: bytes
    timestamp_ms: int
    sequence: int

    def encode(self) -> bytes:
        actors = tuple(self.actors)
        if not 1 <= len(actors) <= MAX_ACTORS or len(set(actors)) != len(actors):
            raise EvidenceFormatError("actor list must contain 1..64 distinct ordered references")
        body = checked_bytes(self.event_id, 16, "event_id") + checked_bytes(
            self.tenant_id, 16, "tenant_id"
        )
        body += struct.pack(">HH", checked_uint(self.event_type, 16, "event_type"), len(actors))
        body += b"".join(checked_bytes(a, 16, "actor") for a in actors)
        body += checked_bytes(self.resource_id, 16, "resource_id") + checked_bytes(
            self.policy_id, 16, "policy_id"
        )
        body += struct.pack(
            ">QQ",
            checked_uint(self.timestamp_ms, 64, "timestamp_ms"),
            checked_uint(self.sequence, 64, "sequence"),
        )
        return frame_record(1, body)

    @classmethod
    def decode(cls, data: bytes) -> EvidenceMetadata:
        r = unframe_record(data, 1)
        event, tenant, event_type, count = r.take(16), r.take(16), r.uint(16), r.uint(16)
        if not 1 <= count <= MAX_ACTORS:
            raise EvidenceFormatError("invalid actor count")
        value = cls(
            event,
            tenant,
            event_type,
            tuple(r.take(16) for _ in range(count)),
            r.take(16),
            r.take(16),
            r.uint(64),
            r.uint(64),
        )
        r.finish()
        if value.encode() != data:
            raise EvidenceFormatError("noncanonical metadata")
        return value


def generate_opening() -> bytes:
    """256 bits from the operating-system CSPRNG, independent for each item."""
    return secrets.token_bytes(OPENING_BYTES)


def payload_commitment(payload: bytes, opening: bytes) -> bytes:
    checked_bytes(opening, OPENING_BYTES, "opening")
    if not isinstance(payload, bytes):
        raise EvidenceFormatError("payload must be bytes")
    return sha256_bytes(DOMAIN_COMMITMENT + opening + len(payload).to_bytes(8, "big") + payload)


def structured_leaf(
    retained_evidence: EvidenceMetadata | Mapping[str, Any], opaque_erased_commitment: bytes
) -> bytes:
    if isinstance(retained_evidence, Mapping):
        retained_evidence = EvidenceMetadata(**retained_evidence)
    return sha256_bytes(
        DOMAIN_LEAF
        + retained_evidence.encode()
        + checked_bytes(opaque_erased_commitment, 32, "commitment")
    )


def merkle_parent(left: bytes, right: bytes) -> bytes:
    return sha256_bytes(
        DOMAIN_MERKLE + checked_bytes(left, 32, "left") + checked_bytes(right, 32, "right")
    )


def build_merkle_levels(leaves: Sequence[bytes]) -> list[list[bytes]]:
    """Sequence-ordered binary tree, duplicating the last node at odd levels."""
    if not leaves:
        raise EvidenceFormatError("empty batches cannot be anchored")
    levels = [[checked_bytes(leaf, 32, "leaf") for leaf in leaves]]
    while len(levels[-1]) > 1:
        current = levels[-1]
        levels.append(
            [
                merkle_parent(current[i], current[min(i + 1, len(current) - 1)])
                for i in range(0, len(current), 2)
            ]
        )
    return levels


def merkle_path(levels: Sequence[Sequence[bytes]], leaf_index: int) -> list[tuple[bytes, bool]]:
    if not levels or not 0 <= leaf_index < len(levels[0]):
        raise EvidenceFormatError("leaf index outside batch")
    index, path = leaf_index, []
    for level in levels[:-1]:
        sibling_index = min(index ^ 1, len(level) - 1)
        path.append((level[sibling_index], bool(index & 1)))
        index //= 2
    return path


def verify_merkle_path(leaf: bytes, path: Sequence[tuple[bytes, bool]], expected_root: bytes) -> bool:
    try:
        current = checked_bytes(leaf, 32, "leaf")
        for sibling, sibling_is_left in path:
            if type(sibling_is_left) is not bool:
                return False
            current = (
                merkle_parent(sibling, current) if sibling_is_left else merkle_parent(current, sibling)
            )
        return current == checked_bytes(expected_root, 32, "root")
    except (EvidenceFormatError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class MembershipProof:
    batch_id: bytes
    leaf_count: int
    leaf_index: int
    siblings: tuple[bytes, ...]

    def encode(self) -> bytes:
        count, index = checked_uint(self.leaf_count, 32, "leaf_count"), checked_uint(
            self.leaf_index, 32, "leaf_index"
        )
        if count < 1 or index >= count or len(self.siblings) != (count - 1).bit_length():
            raise EvidenceFormatError("noncanonical proof dimensions")
        return frame_record(
            2,
            checked_bytes(self.batch_id, 16, "batch_id")
            + struct.pack(">IIH", count, index, len(self.siblings))
            + b"".join(checked_bytes(s, 32, "sibling") for s in self.siblings),
        )

    @classmethod
    def decode(cls, data: bytes) -> MembershipProof:
        r = unframe_record(data, 2)
        batch, count, index, depth = r.take(16), r.uint(32), r.uint(32), r.uint(16)
        if depth > 32:
            raise EvidenceFormatError("proof depth exceeds uint32 tree")
        value = cls(batch, count, index, tuple(r.take(32) for _ in range(depth)))
        r.finish()
        if value.encode() != data:
            raise EvidenceFormatError("noncanonical proof")
        return value

    def verify(self, leaf: bytes, root: bytes) -> bool:
        try:
            self.encode()
            current, index, width = checked_bytes(leaf, 32, "leaf"), self.leaf_index, self.leaf_count
            for sibling in self.siblings:
                if not (index & 1) and index + 1 >= width and sibling != current:
                    return False
                current = (
                    merkle_parent(sibling, current) if index & 1 else merkle_parent(current, sibling)
                )
                index, width = index // 2, (width + 1) // 2
            return current == root
        except (EvidenceFormatError, TypeError, ValueError):
            return False


def make_membership_proof(
    levels: Sequence[Sequence[bytes]], index: int, batch_id: bytes
) -> MembershipProof:
    return MembershipProof(
        batch_id, len(levels[0]), index, tuple(s for s, _ in merkle_path(levels, index))
    )


@dataclass(frozen=True)
class ErasureRequest:
    request_id: bytes
    tenant_id: bytes
    event_id: bytes
    origin_leaf: bytes
    resource_id: bytes
    policy_id: bytes
    authority_id: bytes
    custodians: tuple[bytes, ...]
    scope: int
    requested_at_ms: int
    expires_at_ms: int
    version: int = 1

    def encode(self) -> bytes:
        if (
            not 1 <= len(self.custodians) <= MAX_ACTORS
            or tuple(sorted(set(self.custodians))) != self.custodians
        ):
            raise EvidenceFormatError("custodian roster must be distinct and lexicographically sorted")
        if self.expires_at_ms < self.requested_at_ms:
            raise EvidenceFormatError("request expires before authorization")
        body = b"".join(
            checked_bytes(getattr(self, n), 16, n) for n in ("request_id", "tenant_id", "event_id")
        )
        body += checked_bytes(self.origin_leaf, 32, "origin_leaf")
        body += b"".join(
            checked_bytes(getattr(self, n), 16, n) for n in ("resource_id", "policy_id", "authority_id")
        )
        body += struct.pack(">H", len(self.custodians)) + b"".join(
            checked_bytes(c, 16, "custodian") for c in self.custodians
        )
        body += struct.pack(
            ">IQQI",
            checked_uint(self.scope, 32, "scope"),
            checked_uint(self.requested_at_ms, 64, "requested_at_ms"),
            checked_uint(self.expires_at_ms, 64, "expires_at_ms"),
            checked_uint(self.version, 32, "version"),
        )
        return frame_record(3, body)

    @classmethod
    def decode(cls, data: bytes) -> ErasureRequest:
        r = unframe_record(data, 3)
        prefix = (r.take(16), r.take(16), r.take(16), r.take(32), r.take(16), r.take(16), r.take(16))
        count = r.uint(16)
        if not 1 <= count <= MAX_ACTORS:
            raise EvidenceFormatError("invalid custodian count")
        value = cls(
            *prefix,
            tuple(r.take(16) for _ in range(count)),
            r.uint(32),
            r.uint(64),
            r.uint(64),
            r.uint(32),
        )
        r.finish()
        if value.encode() != data:
            raise EvidenceFormatError("noncanonical request")
        return value


@dataclass(frozen=True)
class Tombstone:
    tenant_id: bytes
    event_id: bytes
    origin_leaf: bytes
    request_id: bytes
    custodian_id: bytes
    resource_id: bytes
    scope: int
    erased_at_ms: int
    version: int = 1

    def encode(self) -> bytes:
        body = checked_bytes(self.tenant_id, 16, "tenant_id") + checked_bytes(
            self.event_id, 16, "event_id"
        )
        body += checked_bytes(self.origin_leaf, 32, "origin_leaf")
        body += b"".join(
            checked_bytes(getattr(self, n), 16, n) for n in ("request_id", "custodian_id", "resource_id")
        )
        body += struct.pack(
            ">IQI",
            checked_uint(self.scope, 32, "scope"),
            checked_uint(self.erased_at_ms, 64, "erased_at_ms"),
            checked_uint(self.version, 32, "version"),
        )
        return frame_record(4, body)

    @classmethod
    def decode(cls, data: bytes) -> Tombstone:
        r = unframe_record(data, 4)
        value = cls(
            r.take(16),
            r.take(16),
            r.take(32),
            r.take(16),
            r.take(16),
            r.take(16),
            r.uint(32),
            r.uint(64),
            r.uint(32),
        )
        r.finish()
        if value.encode() != data:
            raise EvidenceFormatError("noncanonical tombstone")
        return value

    def identity(self) -> bytes:
        # Signature, timestamp and scope are deliberately excluded: modifying
        # those under the same declaration identity is a conflict, not fresh.
        return sha256_bytes(
            b"HTGL-DECLARATION-IDENTITY-v2\x00"
            + self.tenant_id
            + self.event_id
            + self.request_id
            + self.custodian_id
            + self.version.to_bytes(4, "big")
        )


@dataclass(frozen=True)
class AnchorRecord:
    ledger_id: bytes
    batch_id: bytes
    root: bytes
    leaf_count: int
    created_at_ms: int
    height: int
    previous_digest: bytes
    signer_id: bytes

    def encode(self) -> bytes:
        body = (
            checked_bytes(self.ledger_id, 16, "ledger_id")
            + checked_bytes(self.batch_id, 16, "batch_id")
            + checked_bytes(self.root, 32, "root")
        )
        if self.leaf_count < 1 or self.height < 1:
            raise EvidenceFormatError("anchor count and height must be positive")
        body += struct.pack(
            ">IQQ",
            checked_uint(self.leaf_count, 32, "leaf_count"),
            checked_uint(self.created_at_ms, 64, "created_at_ms"),
            checked_uint(self.height, 64, "height"),
        )
        body += checked_bytes(self.previous_digest, 32, "previous_digest") + checked_bytes(
            self.signer_id, 16, "signer_id"
        )
        return frame_record(5, body)

    @classmethod
    def decode(cls, data: bytes) -> AnchorRecord:
        r = unframe_record(data, 5)
        value = cls(
            r.take(16),
            r.take(16),
            r.take(32),
            r.uint(32),
            r.uint(64),
            r.uint(64),
            r.take(32),
            r.take(16),
        )
        r.finish()
        if value.encode() != data:
            raise EvidenceFormatError("noncanonical anchor")
        return value


@dataclass(frozen=True)
class SigningIdentity:
    identity: bytes
    role: str
    private_key: Ed25519PrivateKey

    @classmethod
    def generate(cls, label: str, role: str) -> SigningIdentity:
        return cls(reference_id(label), role, Ed25519PrivateKey.generate())

    @property
    def public_key(self) -> bytes:
        return self.private_key.public_key().public_bytes_raw()

    def sign(self, domain: bytes, payload: bytes) -> bytes:
        return self.private_key.sign(domain + payload)


@lru_cache(maxsize=131072)
def valid_ed25519(public_key: bytes, signature: bytes, message: bytes) -> bool:
    """Cache only immutable cryptographic statements, never trust/replay state."""
    try:
        checked_bytes(public_key, 32, "public_key")
        checked_bytes(signature, 64, "signature")
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
        return True
    except (InvalidSignature, EvidenceFormatError, ValueError, TypeError):
        return False


class TrustRegistry:
    def __init__(self, identities: Sequence[SigningIdentity], ledger_id: bytes, max_faults: int = 1):
        self._configure([(i.identity, i.role, i.public_key) for i in identities], ledger_id, max_faults)

    def _configure(
        self, entries: Sequence[tuple[bytes, str, bytes]], ledger_id: bytes, max_faults: int
    ) -> None:
        self.keys = {}
        for identity, role, key in entries:
            if role not in {"origin", "authority", "custodian", "anchor", "validator", "auditor"}:
                raise EvidenceFormatError("unknown role in trust registry")
            self.keys[checked_bytes(identity, 16, "trust identity")] = (
                role,
                checked_bytes(key, 32, "trust public key"),
            )
        if len(self.keys) != len(entries):
            raise EvidenceFormatError("duplicate trust identity")
        self.ledger_id = checked_bytes(ledger_id, 16, "ledger_id")
        self.validators = frozenset(k for k, (role, _) in self.keys.items() if role == "validator")
        self.max_faults = checked_uint(max_faults, 16, "max_faults")
        if len(self.validators) != 3 * max_faults + 1:
            raise EvidenceFormatError("reference quorum requires n=3f+1 validators")
        self.quorum = 2 * max_faults + 1
        self.revoked: set[bytes] = set()

    def verifies(
        self, identity: bytes, role: str, domain: bytes, payload: bytes, signature: bytes
    ) -> bool:
        entry = self.keys.get(identity)
        return bool(
            entry
            and entry[0] == role
            and identity not in self.revoked
            and valid_ed25519(entry[1], signature, domain + payload)
        )

    def public_manifest(self) -> dict[str, Any]:
        return {
            "ledger_id": self.ledger_id.hex(),
            "max_faults": self.max_faults,
            "quorum": self.quorum,
            "revoked": sorted(i.hex() for i in self.revoked),
            "identities": [
                {"id": i.hex(), "role": role, "public_key": key.hex()}
                for i, (role, key) in sorted(self.keys.items())
            ],
        }

    @classmethod
    def from_manifest(cls, data: Mapping[str, Any]) -> TrustRegistry:
        if set(data) != {"ledger_id", "max_faults", "quorum", "revoked", "identities"}:
            raise EvidenceFormatError("invalid trust manifest fields")
        value = cls.__new__(cls)
        entries = [
            (bytes.fromhex(i["id"]), i["role"], bytes.fromhex(i["public_key"]))
            for i in data["identities"]
        ]
        value._configure(entries, bytes.fromhex(data["ledger_id"]), data["max_faults"])
        if value.quorum != data["quorum"]:
            raise EvidenceFormatError("trust manifest quorum disagrees with validator set")
        value.revoked = {bytes.fromhex(i) for i in data["revoked"]}
        if not value.revoked <= value.keys.keys():
            raise EvidenceFormatError("revocation of unknown identity")
        return value


@dataclass(frozen=True)
class FinalizedAnchor:
    record: AnchorRecord
    signature: bytes
    votes: tuple[tuple[bytes, bytes], ...]

    @property
    def digest(self) -> bytes:
        return sha256_bytes(self.record.encode() + self.signature)

    def encode(self) -> bytes:
        return frame_record(
            6,
            self.record.encode()
            + checked_bytes(self.signature, 64, "anchor signature")
            + struct.pack(">H", len(self.votes))
            + b"".join(
                checked_bytes(i, 16, "validator") + checked_bytes(s, 64, "vote") for i, s in self.votes
            ),
        )

    @classmethod
    def decode(cls, data: bytes) -> FinalizedAnchor:
        r = unframe_record(data, 6)
        header = r.take(11)
        if header[:4] != b"HTGL":
            raise EvidenceFormatError("invalid embedded anchor")
        length = int.from_bytes(header[7:11], "big")
        record = AnchorRecord.decode(header + r.take(length))
        signature, count = r.take(64), r.uint(16)
        if count > MAX_ACTORS:
            raise EvidenceFormatError("oversized finality certificate")
        value = cls(record, signature, tuple((r.take(16), r.take(64)) for _ in range(count)))
        r.finish()
        if value.encode() != data:
            raise EvidenceFormatError("noncanonical certificate")
        return value

    def authenticity(self, trust: TrustRegistry) -> bool:
        return self.record.ledger_id == trust.ledger_id and trust.verifies(
            self.record.signer_id,
            "anchor",
            DOMAIN_ANCHOR_SIGNATURE,
            self.record.encode(),
            self.signature,
        )

    def finality(self, trust: TrustRegistry) -> bool:
        voters = [v for v, _ in self.votes]
        if (
            len(voters) < trust.quorum
            or len(voters) > len(trust.validators)
            or len(set(voters)) != len(voters)
        ):
            return False
        message = self.record.height.to_bytes(8, "big") + self.digest
        return all(
            v in trust.validators and trust.verifies(v, "validator", DOMAIN_COMMIT, message, signature)
            for v, signature in self.votes
        )


def database_connection(path: Path | str) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA secure_delete=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


class QuorumAnchorLedger:
    """Executable local authenticated log with persistent prepare/commit locks.

    Four identities execute a fixed-view two-phase quorum protocol; f=1,
    quorum=3. This is an in-process fault experiment, not a deployed BFT
    network, a Fabric benchmark, or a proof of liveness under view changes.
    A durable validator lock prevents honest equivocation at a log height.
    """

    def __init__(
        self,
        path: Path,
        signer: SigningIdentity,
        validators: Sequence[SigningIdentity],
        trust: TrustRegistry,
    ):
        self.path, self.signer, self.validators, self.trust = path, signer, tuple(validators), trust
        self.connection = database_connection(path)
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS anchors(height INTEGER PRIMARY KEY, batch_id BLOB UNIQUE NOT NULL, digest BLOB UNIQUE NOT NULL, certificate BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS validator_locks(validator BLOB NOT NULL, height INTEGER NOT NULL, digest BLOB NOT NULL, PRIMARY KEY(validator,height)) WITHOUT ROWID;
            CREATE TRIGGER IF NOT EXISTS anchors_no_update BEFORE UPDATE ON anchors BEGIN SELECT RAISE(ABORT,'append-only anchor log'); END;
            CREATE TRIGGER IF NOT EXISTS anchors_no_delete BEFORE DELETE ON anchors BEGIN SELECT RAISE(ABORT,'append-only anchor log'); END;
        """)

    def close(self) -> None:
        self.connection.close()

    def head(self) -> tuple[int, bytes]:
        row = self.connection.execute(
            "SELECT height,digest FROM anchors ORDER BY height DESC LIMIT 1"
        ).fetchone()
        return (int(row[0]), bytes(row[1])) if row else (0, ZERO_HASH)

    def append(
        self,
        batch_id: bytes,
        root: bytes,
        count: int,
        timestamp_ms: int,
        unavailable: frozenset[bytes] = frozenset(),
        equivocate: frozenset[bytes] = frozenset(),
    ) -> FinalizedAnchor:
        if (
            not (unavailable | equivocate) <= self.trust.validators
            or len(equivocate) > self.trust.max_faults
        ):
            raise EvidenceFormatError(
                "fault injection must stay within the declared validator set and equivocation bound"
            )
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self.connection.execute(
                "SELECT certificate FROM anchors WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if existing:
                certificate = self.lookup(batch_id)
                if (root, count, timestamp_ms) != (
                    certificate.record.root,
                    certificate.record.leaf_count,
                    certificate.record.created_at_ms,
                ):
                    raise EvidenceFormatError("conflicting anchor for existing batch identity")
                self.connection.execute("COMMIT")
                return certificate
            height, previous = self.head()
            record = AnchorRecord(
                self.trust.ledger_id,
                batch_id,
                root,
                count,
                timestamp_ms,
                height + 1,
                previous,
                self.signer.identity,
            )
            if height:
                prior = self.connection.execute(
                    "SELECT certificate FROM anchors WHERE height=?", (height,)
                ).fetchone()
                if timestamp_ms < FinalizedAnchor.decode(bytes(prior[0])).record.created_at_ms:
                    raise EvidenceFormatError("anchor timestamp regresses")
            signature = self.signer.sign(DOMAIN_ANCHOR_SIGNATURE, record.encode())
            unsigned = FinalizedAnchor(record, signature, ())
            if not unsigned.authenticity(self.trust):
                raise EvidenceFormatError("unauthenticated anchor proposal")
            message = record.height.to_bytes(8, "big") + unsigned.digest
            prepared: list[tuple[bytes, bytes]] = []
            for validator in self.validators:
                if validator.identity in unavailable:
                    continue
                lock = self.connection.execute(
                    "SELECT digest FROM validator_locks WHERE validator=? AND height=?",
                    (validator.identity, record.height),
                ).fetchone()
                if lock and bytes(lock[0]) != unsigned.digest and validator.identity not in equivocate:
                    continue
                if not lock:
                    self.connection.execute(
                        "INSERT INTO validator_locks VALUES(?,?,?)",
                        (validator.identity, record.height, unsigned.digest),
                    )
                prepared.append((validator.identity, validator.sign(DOMAIN_PREPARE, message)))
            if len(prepared) < self.trust.quorum:
                self.connection.execute("COMMIT")
                raise EvidenceFormatError("insufficient prepare quorum; durable locks retained")
            prepared_ids = {i for i, _ in prepared}
            if len(prepared_ids) != len(prepared) or not all(
                self.trust.verifies(i, "validator", DOMAIN_PREPARE, message, s) for i, s in prepared
            ):
                raise EvidenceFormatError("invalid prepare certificate")
            votes = tuple(
                (v.identity, v.sign(DOMAIN_COMMIT, message))
                for v in self.validators
                if v.identity in prepared_ids
            )
            certificate = FinalizedAnchor(record, signature, votes[: self.trust.quorum])
            if not certificate.finality(self.trust):
                raise EvidenceFormatError("invalid commit certificate")
            self.connection.execute(
                "INSERT INTO anchors VALUES(?,?,?,?)",
                (record.height, batch_id, certificate.digest, certificate.encode()),
            )
            self.connection.execute("COMMIT")
        except BaseException:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise
        return certificate

    def lookup(self, batch_id: bytes) -> FinalizedAnchor:
        row = self.connection.execute(
            "SELECT certificate FROM anchors WHERE batch_id=?", (batch_id,)
        ).fetchone()
        if not row:
            raise EvidenceFormatError("anchor not found")
        value = FinalizedAnchor.decode(bytes(row[0]))
        if not value.authenticity(self.trust) or not value.finality(self.trust):
            raise EvidenceFormatError("stored anchor certificate failed verification")
        return value

    def verify_history(self) -> bool:
        previous, height, timestamp = ZERO_HASH, 0, 0
        for row in self.connection.execute(
            "SELECT height,digest,certificate FROM anchors ORDER BY height"
        ):
            anchor = FinalizedAnchor.decode(bytes(row[2]))
            if (
                row[0] != height + 1
                or anchor.record.height != row[0]
                or anchor.record.previous_digest != previous
                or anchor.record.created_at_ms < timestamp
            ):
                return False
            if (
                anchor.digest != bytes(row[1])
                or not anchor.authenticity(self.trust)
                or not anchor.finality(self.trust)
            ):
                return False
            previous, height, timestamp = (
                anchor.digest,
                anchor.record.height,
                anchor.record.created_at_ms,
            )
        return True


class ReplayStore:
    """Atomic verifier-local admission; inspection never consumes freshness."""

    def __init__(self, path: Path | str, verifier_id: bytes):
        self.verifier_id = checked_bytes(verifier_id, 16, "verifier_id")
        self.connection = database_connection(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS declarations(verifier BLOB NOT NULL, identity BLOB NOT NULL, digest BLOB NOT NULL, accepted_at_ns INTEGER NOT NULL, PRIMARY KEY(verifier,identity)) WITHOUT ROWID"
        )

    def close(self) -> None:
        self.connection.close()

    def register(self, tombstone: Tombstone) -> tuple[bool, str]:
        identity, digest = tombstone.identity(), sha256_bytes(tombstone.encode())
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT digest FROM declarations WHERE verifier=? AND identity=?",
                (self.verifier_id, identity),
            ).fetchone()
            if row:
                result = (
                    False,
                    "DUPLICATE_DECLARATION" if bytes(row[0]) == digest else "CONFLICTING_DECLARATION",
                )
            else:
                self.connection.execute(
                    "INSERT INTO declarations VALUES(?,?,?,?)",
                    (self.verifier_id, identity, digest, time.time_ns()),
                )
                result = (True, "REGISTERED_NEW_DECLARATION")
            self.connection.execute("COMMIT")
            return result
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise


@dataclass(frozen=True)
class AuditTarget:
    tenant_id: bytes
    event_id: bytes
    resource_id: bytes
    policy_id: bytes
    request_id: bytes
    custodian_id: bytes
    scope: int
    origin_batch_id: bytes
    request_batch_id: bytes
    origin_leaf: bytes
    request_leaf: bytes

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            if field.name == "scope":
                checked_uint(self.scope, 32, "scope")
            else:
                size = 32 if field.name.endswith("_leaf") else 16
                checked_bytes(getattr(self, field.name), size, field.name)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AuditTarget:
        expected = {f.name for f in dataclasses.fields(cls)}
        if set(data) != expected:
            raise EvidenceFormatError("unexpected or missing independent target fields")
        return cls(**{k: v if k == "scope" else bytes.fromhex(v) for k, v in data.items()})


@dataclass(frozen=True)
class EvidencePackage:
    metadata: EvidenceMetadata
    commitment: bytes
    leaf: bytes
    origin_signature: bytes
    origin_proof: MembershipProof
    origin_anchor: FinalizedAnchor
    request: ErasureRequest
    request_signature: bytes
    request_proof: MembershipProof
    request_anchor: FinalizedAnchor
    tombstone: Tombstone
    tombstone_signature: bytes

    def to_dict(self) -> dict[str, str | int]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "metadata": self.metadata.encode().hex(),
            "commitment": self.commitment.hex(),
            "leaf": self.leaf.hex(),
            "origin_signature": self.origin_signature.hex(),
            "origin_proof": self.origin_proof.encode().hex(),
            "origin_anchor": self.origin_anchor.encode().hex(),
            "request": self.request.encode().hex(),
            "request_signature": self.request_signature.hex(),
            "request_proof": self.request_proof.encode().hex(),
            "request_anchor": self.request_anchor.encode().hex(),
            "tombstone": self.tombstone.encode().hex(),
            "tombstone_signature": self.tombstone_signature.hex(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidencePackage:
        required = {f.name for f in dataclasses.fields(cls)} | {"protocol_version"}
        if (
            set(data) != required
            or type(data["protocol_version"]) is not int
            or data["protocol_version"] != PROTOCOL_VERSION
        ):
            raise EvidenceFormatError("missing/unknown fields or unsupported protocol")
        try:
            blobs = {}
            for key in required - {"protocol_version"}:
                text = data[key]
                if (
                    not isinstance(text, str)
                    or text != text.lower()
                    or len(text) % 2
                    or any(c not in "0123456789abcdef" for c in text)
                ):
                    raise EvidenceFormatError("transport fields must be lowercase hexadecimal")
                blobs[key] = bytes.fromhex(text)
            return cls(
                EvidenceMetadata.decode(blobs["metadata"]),
                checked_bytes(blobs["commitment"], 32, "commitment"),
                checked_bytes(blobs["leaf"], 32, "leaf"),
                checked_bytes(blobs["origin_signature"], 64, "origin_signature"),
                MembershipProof.decode(blobs["origin_proof"]),
                FinalizedAnchor.decode(blobs["origin_anchor"]),
                ErasureRequest.decode(blobs["request"]),
                checked_bytes(blobs["request_signature"], 64, "request_signature"),
                MembershipProof.decode(blobs["request_proof"]),
                FinalizedAnchor.decode(blobs["request_anchor"]),
                Tombstone.decode(blobs["tombstone"]),
                checked_bytes(blobs["tombstone_signature"], 64, "tombstone_signature"),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise EvidenceFormatError(str(exc)) from exc

    def encode_json(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def decode_json(cls, data: bytes) -> EvidencePackage:
        if len(data) > MAX_JSON_PACKAGE_BYTES:
            raise EvidenceFormatError("oversized package")

        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in items:
                if key in value:
                    raise EvidenceFormatError("duplicate JSON key")
                value[key] = item
            return value

        try:
            decoded = json.loads(data, object_pairs_hook=pairs)
            if not isinstance(decoded, dict):
                raise EvidenceFormatError("package must be an object")
            return cls.from_dict(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise EvidenceFormatError(str(exc)) from exc


def request_leaf(request: ErasureRequest, signature: bytes) -> bytes:
    return sha256_bytes(
        DOMAIN_REQUEST + request.encode() + checked_bytes(signature, 64, "request_signature")
    )


class EvidenceVerifier:
    def __init__(self, trust: TrustRegistry, replay: ReplayStore | None = None):
        self.trust, self.replay = trust, replay

    def verify(
        self,
        package: EvidencePackage | Mapping[str, Any] | bytes,
        target: AuditTarget,
        mode: str = "inspect",
    ) -> tuple[bool, str]:
        if mode not in ("inspect", "register"):
            raise ValueError("mode must be inspect or register")
        try:
            if isinstance(package, bytes):
                p = EvidencePackage.decode_json(package)
            elif isinstance(package, Mapping):
                p = EvidencePackage.from_dict(package)
            elif isinstance(package, EvidencePackage):
                p = package
            else:
                raise EvidenceFormatError("unsupported package type")
            m, request, t = p.metadata, p.request, p.tombstone
            # Encode every record before any trust decision; dataclass types
            # alone do not guarantee canonical field lengths/ranges.
            m.encode()
            request.encode()
            t.encode()
            p.origin_proof.encode()
            p.request_proof.encode()
            if (m.tenant_id, m.event_id, m.resource_id, m.policy_id) != (
                target.tenant_id,
                target.event_id,
                target.resource_id,
                target.policy_id,
            ):
                return False, "AUDIT_TARGET_MISMATCH"
            if (request.request_id, t.custodian_id, request.scope) != (
                target.request_id,
                target.custodian_id,
                target.scope,
            ):
                return False, "AUDIT_TARGET_MISMATCH"
            if (p.origin_proof.batch_id, p.request_proof.batch_id) != (
                target.origin_batch_id,
                target.request_batch_id,
            ):
                return False, "AUDIT_ANCHOR_CONTEXT_MISMATCH"
            if p.leaf != target.origin_leaf:
                return False, "AUDIT_ORIGIN_LEAF_MISMATCH"
            if not self.trust.verifies(
                m.actors[0],
                "origin",
                DOMAIN_ORIGIN_SIGNATURE,
                m.encode() + p.commitment + p.leaf,
                p.origin_signature,
            ):
                return False, "INVALID_ORIGIN_SIGNATURE"
            if structured_leaf(m, p.commitment) != p.leaf:
                return False, "STRUCTURED_COMMITMENT_MISMATCH"
            request_hash = request_leaf(request, p.request_signature)
            for proof, anchor, leaf in (
                (p.origin_proof, p.origin_anchor, p.leaf),
                (p.request_proof, p.request_anchor, request_hash),
            ):
                if not anchor.authenticity(self.trust):
                    return False, "INVALID_ANCHOR_AUTHENTICITY"
                if not anchor.finality(self.trust):
                    return False, "UNRESOLVED_ANCHOR_FINALITY"
                if (proof.batch_id, proof.leaf_count) != (
                    anchor.record.batch_id,
                    anchor.record.leaf_count,
                ):
                    return False, "ANCHOR_PROOF_CONTEXT_MISMATCH"
                if not proof.verify(leaf, anchor.record.root):
                    return False, "MERKLE_MEMBERSHIP_FAILURE"
            if request_hash != target.request_leaf:
                return False, "AUDIT_REQUEST_LEAF_MISMATCH"
            if not self.trust.verifies(
                request.authority_id,
                "authority",
                DOMAIN_REQUEST_SIGNATURE,
                request.encode(),
                p.request_signature,
            ):
                return False, "INVALID_REQUEST_AUTHORIZATION"
            if (
                request.tenant_id,
                request.event_id,
                request.origin_leaf,
                request.resource_id,
                request.policy_id,
            ) != (m.tenant_id, m.event_id, p.leaf, m.resource_id, m.policy_id):
                return False, "REQUEST_ORIGIN_MISMATCH"
            if (
                t.custodian_id not in request.custodians
                or t.scope != request.scope
                or t.version != request.version
            ):
                return False, "UNAUTHORIZED_SCOPE_OR_CUSTODIAN"
            if not self.trust.verifies(
                t.custodian_id,
                "custodian",
                DOMAIN_TOMBSTONE_SIGNATURE,
                t.encode(),
                p.tombstone_signature,
            ):
                return False, "INVALID_TOMBSTONE_SIGNATURE"
            if (t.tenant_id, t.event_id, t.origin_leaf, t.request_id, t.resource_id) != (
                m.tenant_id,
                m.event_id,
                p.leaf,
                request.request_id,
                m.resource_id,
            ):
                return False, "TOMBSTONE_LINKAGE_MISMATCH"
            if not (
                m.timestamp_ms
                <= p.origin_anchor.record.created_at_ms
                <= request.requested_at_ms
                <= p.request_anchor.record.created_at_ms
                <= t.erased_at_ms
                <= request.expires_at_ms
            ):
                return False, "TEMPORAL_ORDER_FAILURE"
            if mode == "register":
                if self.replay is None:
                    raise ValueError("registration requires a persistent ReplayStore")
                return self.replay.register(t)
            return True, "VERIFIED_POST_ERASURE"
        except (EvidenceFormatError, KeyError, TypeError, AttributeError, OverflowError) as exc:
            return False, "MALFORMED_EVIDENCE:" + type(exc).__name__

    def verify_case(
        self, packages: Sequence[EvidencePackage], targets: Sequence[AuditTarget]
    ) -> dict[str, Any]:
        if len(packages) != len(targets) or not packages:
            return {"accepted": False, "reason": "EMPTY_OR_UNPAIRED_CASE", "complete": False}
        expected = set(packages[0].request.custodians)
        covered: set[bytes] = set()
        request_digest = sha256_bytes(packages[0].request.encode())
        for package, target in zip(packages, targets):
            accepted, reason = self.verify(package, target)
            if not accepted or sha256_bytes(package.request.encode()) != request_digest:
                return {
                    "accepted": False,
                    "reason": reason if not accepted else "MIXED_REQUEST_CASE",
                    "complete": False,
                }
            if package.tombstone.custodian_id in covered:
                return {"accepted": False, "reason": "DUPLICATE_CUSTODIAN_RESPONSE", "complete": False}
            covered.add(package.tombstone.custodian_id)
        return {
            "accepted": True,
            "reason": "DECLARED_CUSTODIAN_COVERAGE",
            "complete": covered == expected,
            "expected_custodians": len(expected),
            "covered_custodians": len(covered),
            "missing_custodians": sorted(i.hex() for i in expected - covered),
            "physical_deletion_established": False,
            "undisclosed_custodians_covered": False,
        }


def verify_abstract_package(
    package: EvidencePackage | Mapping[str, Any] | bytes,
    *,
    target: AuditTarget,
    trust: TrustRegistry,
    replay: ReplayStore | None = None,
    mode: str = "inspect",
) -> tuple[bool, str]:
    """Compatibility entry point; all signatures/certificates are now executed."""
    return EvidenceVerifier(trust, replay).verify(package, target, mode)


def flip_first_byte(value: bytes) -> bytes:
    return bytes((value[0] ^ 1,)) + value[1:]


class ErasableApplicationStore:
    """Logical application-store removal only; no physical-media guarantee."""

    def __init__(self, path: Path):
        self.connection = database_connection(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS payloads(event BLOB PRIMARY KEY, payload BLOB NOT NULL, opening BLOB NOT NULL) WITHOUT ROWID"
        )

    def put(self, event: bytes, payload: bytes) -> bytes:
        opening = generate_opening()
        self.connection.execute("INSERT INTO payloads VALUES(?,?,?)", (event, payload, opening))
        return payload_commitment(payload, opening)

    def erase(self, event: bytes) -> None:
        self.connection.execute("DELETE FROM payloads WHERE event=?", (event,))

    def contains(self, event: bytes) -> bool:
        return (
            self.connection.execute("SELECT 1 FROM payloads WHERE event=?", (event,)).fetchone()
            is not None
        )

    def close(self) -> None:
        self.connection.close()


@dataclass(frozen=True)
class ClosedBatch:
    batch_id: bytes
    leaves: tuple[bytes, ...]
    sequences: tuple[int, ...]
    arrivals: tuple[float, ...]
    closed_at: float
    reason: str


class CapacityTimeoutBatcher:
    """Event-driven capacity/first-arrival timeout; no fixed-window proxy."""

    def __init__(self, capacity: int, timeout_seconds: float):
        if (
            type(capacity) is not int
            or capacity < 1
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("positive capacity and timeout required")
        self.capacity, self.timeout = capacity, float(timeout_seconds)
        self.pending: list[tuple[int, bytes, float]] = []
        self.last_arrival, self.ordinal = -math.inf, 0
        self.sequences: set[int] = set()

    def _close(self, closed_at: float, reason: str) -> ClosedBatch:
        items = sorted(self.pending)
        self.pending = []
        batch_id = reference_id(f"batch:{self.capacity}:{self.timeout}:{self.ordinal}")
        self.ordinal += 1
        return ClosedBatch(
            batch_id,
            tuple(i[1] for i in items),
            tuple(i[0] for i in items),
            tuple(i[2] for i in items),
            closed_at,
            reason,
        )

    def push(self, arrival_seconds: float, sequence: int, leaf: bytes) -> list[ClosedBatch]:
        if not math.isfinite(arrival_seconds) or arrival_seconds < self.last_arrival:
            raise EvidenceFormatError("arrival timestamps must be finite and nondecreasing")
        if type(sequence) is not int or sequence < 0 or sequence in self.sequences:
            raise EvidenceFormatError("event sequence must be nonnegative and unique")
        self.sequences.add(sequence)
        self.last_arrival = arrival_seconds
        closed = []
        if self.pending and arrival_seconds >= self.pending[0][2] + self.timeout:
            closed.append(self._close(self.pending[0][2] + self.timeout, "timeout"))
        self.pending.append((sequence, checked_bytes(leaf, 32, "leaf"), arrival_seconds))
        if len(self.pending) == self.capacity:
            closed.append(self._close(arrival_seconds, "capacity"))
        return closed

    def finish(self, observation_end: float) -> list[ClosedBatch]:
        if not math.isfinite(observation_end) or observation_end < self.last_arrival:
            raise EvidenceFormatError("observation end precedes arrivals")
        if not self.pending:
            return []
        deadline = self.pending[0][2] + self.timeout
        return [
            self._close(
                min(observation_end, deadline),
                "timeout" if deadline <= observation_end else "observation_end",
            )
        ]


def make_protocol_fixture(count: int, destination: Path, roster_size: int = 2) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    destination.mkdir(parents=True, exist_ok=True)
    origin = SigningIdentity.generate("origin", "origin")
    authority = SigningIdentity.generate("authority", "authority")
    auditor = SigningIdentity.generate("auditor-A", "auditor")
    anchor_signer = SigningIdentity.generate("anchor", "anchor")
    custodians = tuple(
        SigningIdentity.generate(f"custodian-{i}", "custodian") for i in range(roster_size)
    )
    validators = tuple(SigningIdentity.generate(f"validator-{i}", "validator") for i in range(4))
    identities = (origin, authority, auditor, anchor_signer, *custodians, *validators)
    trust = TrustRegistry(identities, reference_id("local-reference-ledger"))
    ledger = QuorumAnchorLedger(destination / "anchors.sqlite", anchor_signer, validators, trust)
    store = ErasableApplicationStore(destination / "application.sqlite")
    base = 1_800_000_000_000
    metadata, commitments, leaves, origin_signatures = [], [], [], []
    for index in range(count):
        item = EvidenceMetadata(
            reference_id(f"event-{index}"),
            reference_id("tenant"),
            index % 5,
            (origin.identity, reference_id("requester")),
            reference_id(f"resource-{index}"),
            reference_id("policy"),
            base + index,
            index,
        )
        commitment = store.put(item.event_id, f"synthetic application payload {index}".encode())
        leaf = structured_leaf(item, commitment)
        metadata.append(item)
        commitments.append(commitment)
        leaves.append(leaf)
        origin_signatures.append(origin.sign(DOMAIN_ORIGIN_SIGNATURE, item.encode() + commitment + leaf))
    recorded_ns = time.perf_counter_ns()

    def anchor_series(
        label: str, series: Sequence[bytes], timestamps_ms: Sequence[int]
    ) -> tuple[dict[int, MembershipProof], dict[int, FinalizedAnchor], int]:
        batcher = CapacityTimeoutBatcher(
            WORKFLOW_ASSUMPTIONS.reference_batch_size,
            WORKFLOW_ASSUMPTIONS.reference_batch_window_seconds,
        )
        closed = []
        for index, (leaf, timestamp) in enumerate(zip(series, timestamps_ms)):
            closed.extend(batcher.push((timestamp - base) / 1000, index, leaf))
        closed.extend(batcher.finish((timestamps_ms[-1] - base) / 1000 + batcher.timeout))
        proofs, anchors = {}, {}
        for ordinal, batch in enumerate(closed):
            levels = build_merkle_levels(batch.leaves)
            batch_id = reference_id(f"{label}-batch-{ordinal}")
            anchored = ledger.append(
                batch_id, levels[-1][0], len(batch.leaves), base + round(batch.closed_at * 1000)
            )
            for local, index in enumerate(batch.sequences):
                proofs[index] = make_membership_proof(levels, local, batch_id)
                anchors[index] = anchored
        return proofs, anchors, max(anchor.record.created_at_ms for anchor in anchors.values())

    origin_proofs, origin_anchors, last_origin_anchor = anchor_series(
        "origin", leaves, [m.timestamp_ms for m in metadata]
    )
    origin_anchored_ns = time.perf_counter_ns()
    requests, request_signatures, request_leaves = [], [], []
    for index, item in enumerate(metadata):
        request = ErasureRequest(
            reference_id(f"request-{index}"),
            item.tenant_id,
            item.event_id,
            leaves[index],
            item.resource_id,
            item.policy_id,
            authority.identity,
            tuple(sorted(c.identity for c in custodians)),
            7,
            last_origin_anchor + 1 + index,
            last_origin_anchor + count * 4 + 200_000,
        )
        signature = authority.sign(DOMAIN_REQUEST_SIGNATURE, request.encode())
        requests.append(request)
        request_signatures.append(signature)
        request_leaves.append(request_leaf(request, signature))
    request_proofs, request_anchors, last_request_anchor = anchor_series(
        "request", request_leaves, [r.requested_at_ms for r in requests]
    )
    request_anchored_ns = time.perf_counter_ns()
    packages, targets, truth = [], [], []
    selected_custodian = custodians[0]
    for index, item in enumerate(metadata):
        # The normal workflow erases before signing. The explicit false
        # declaration case skips removal while preserving valid evidence.
        false_declaration = index % 17 == 0
        if not false_declaration:
            store.erase(item.event_id)
        payload_present = store.contains(item.event_id)
        if payload_present != false_declaration:
            raise RuntimeError("application store did not reach the intended fixture state")
        tombstone = Tombstone(
            item.tenant_id,
            item.event_id,
            leaves[index],
            requests[index].request_id,
            selected_custodian.identity,
            item.resource_id,
            7,
            last_request_anchor + 1 + index,
        )
        package = EvidencePackage(
            item,
            commitments[index],
            leaves[index],
            origin_signatures[index],
            origin_proofs[index],
            origin_anchors[index],
            requests[index],
            request_signatures[index],
            request_proofs[index],
            request_anchors[index],
            tombstone,
            selected_custodian.sign(DOMAIN_TOMBSTONE_SIGNATURE, tombstone.encode()),
        )
        packages.append(package)
        targets.append(
            AuditTarget(
                item.tenant_id,
                item.event_id,
                item.resource_id,
                item.policy_id,
                requests[index].request_id,
                selected_custodian.identity,
                7,
                origin_proofs[index].batch_id,
                request_proofs[index].batch_id,
                leaves[index],
                request_leaves[index],
            )
        )
        truth.append(
            {
                "event_id": item.event_id.hex(),
                "declared_erased": True,
                "application_payload_present": payload_present,
                "opening_in_witness": False,
                "physical_storage_state_observed": False,
            }
        )
    if not ledger.verify_history():
        raise RuntimeError("reference ledger history failed verification")
    stage_ms = {
        "record_commitment_and_origin_signature": (recorded_ns - started_ns) / 1e6,
        "origin_batch_and_quorum_anchoring": (origin_anchored_ns - recorded_ns) / 1e6,
        "request_signature_batch_and_quorum_anchoring": (request_anchored_ns - origin_anchored_ns) / 1e6,
        "logical_store_removal_and_tombstones": (time.perf_counter_ns() - request_anchored_ns) / 1e6,
    }
    return {
        "packages": packages,
        "targets": targets,
        "truth": truth,
        "trust": trust,
        "ledger": ledger,
        "store": store,
        "origin": origin,
        "authority": authority,
        "auditor": auditor,
        "custodians": custodians,
        "anchor_signer": anchor_signer,
        "validators": validators,
        "stage_ms": stage_ms,
    }


def export_reference_measurements(fixture: Mapping[str, Any]) -> None:
    packages, targets = fixture["packages"], fixture["targets"]
    public_trust = TrustRegistry.from_manifest(fixture["trust"].public_manifest())
    verifier = EvidenceVerifier(public_trust)
    cold_us, warm_us, transport_bytes, receipts = [], [], [], []
    sample_indexes = np.unique(np.linspace(0, len(packages) - 1, min(128, len(packages)), dtype=int))
    for index in sample_indexes:
        p, target = packages[int(index)], targets[int(index)]
        transport = p.encode_json()
        transport_bytes.append(len(transport))
        valid_ed25519.cache_clear()
        start = time.perf_counter_ns()
        accepted, _ = verifier.verify(transport, target)
        cold_us.append((time.perf_counter_ns() - start) / 1000)
        start = time.perf_counter_ns()
        repeated, _ = verifier.verify(transport, target)
        warm_us.append((time.perf_counter_ns() - start) / 1000)
        if not accepted or not repeated:
            raise RuntimeError("portable witness failed independently loaded trust registry")
        receipt = AuditReceipt(
            fixture["auditor"].identity,
            p.metadata.event_id,
            p.request.request_id,
            p.tombstone.identity(),
            sha256_bytes(transport),
            p.tombstone.erased_at_ms + 1,
            0,
        )
        signature = fixture["auditor"].sign(b"HTGL-AUDIT-RECEIPT-v2\x00", receipt.encode())
        round_trip = AuditReceipt.decode(receipt.encode())
        if not round_trip.verify(signature, public_trust) or round_trip.verify(
            flip_first_byte(signature), public_trust
        ):
            raise RuntimeError("audit-receipt authenticity round trip failed")
        receipts.append({"receipt_hex": receipt.encode().hex(), "signature_hex": signature.hex()})
    rows = [
        {
            "stage": key,
            "packages": len(packages),
            "wall_clock_ms": value,
            "scope": "local executable application store, signer and quorum log",
        }
        for key, value in fixture["stage_ms"].items()
    ]
    write_csv(pd.DataFrame(rows), DATA / "local_reference_pipeline_timings.csv")
    write_csv(
        pd.DataFrame(
            [
                {
                    "packages_sampled": len(sample_indexes),
                    "local_reference_packages_created": len(packages),
                    "anchor_count": fixture["ledger"].head()[0],
                    "median_portable_witness_json_bytes": float(np.median(transport_bytes)),
                    "median_cold_verify_us": float(np.median(cold_us)),
                    "p99_cold_verify_us": float(np.quantile(cold_us, 0.99)),
                    "median_warm_verify_us": float(np.median(warm_us)),
                    "p99_warm_verify_us": float(np.quantile(warm_us, 0.99)),
                    "cold_definition": "clear immutable signature-result cache; decode and check complete witness",
                    "anchor_backend": "local SQLite append-only log with in-process 3-of-4 prepare/commit certificates",
                    "scope": "measured local reference execution; consensus network throughput unmeasured",
                }
            ]
        ),
        DATA / "local_reference_verification_timings.csv",
    )
    (DATA / "signed_audit_receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    anchors = [
        bytes(r[0]).hex()
        for r in fixture["ledger"].connection.execute("SELECT certificate FROM anchors ORDER BY height")
    ]
    (DATA / "worked_anchor_history.json").write_text(
        json.dumps({"certificates_hex": anchors}, indent=2) + "\n"
    )


def formal_model_experiment() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concrete signed evidence, independently bound targets, and real replay."""
    execution = tempfile.TemporaryDirectory(prefix="protocol-execution-", dir=DATA)
    folder = Path(execution.name)
    fixture = make_protocol_fixture(FORMAL_MODEL_EVENTS, folder)
    trust = fixture["trust"]
    replay_path = folder / "verifiers.sqlite"
    replay = ReplayStore(replay_path, reference_id("auditor-A"))
    other = ReplayStore(replay_path, reference_id("auditor-B"))
    verifier, other_verifier = EvidenceVerifier(trust, replay), EvidenceVerifier(trust, other)
    results: dict[str, dict[str, Any]] = {}
    trace_rows: list[dict[str, Any]] = []

    def exercise(
        name: str,
        package: Any,
        target: AuditTarget,
        expected: bool,
        expected_reason: str,
        mode: str = "inspect",
        selected: EvidenceVerifier = verifier,
    ) -> None:
        start = time.perf_counter_ns()
        accepted, reason = selected.verify(package, target, mode)
        elapsed = time.perf_counter_ns() - start
        bucket = results.setdefault(
            name,
            {
                "cases_tested": 0,
                "accepted": 0,
                "rejected": 0,
                "expected_accept": expected,
                "correct_reason": 0,
                "reason_counts": Counter(),
                "latency_ns": [],
            },
        )
        bucket["cases_tested"] += 1
        bucket["accepted" if accepted else "rejected"] += 1
        bucket["reason_counts"][reason] += 1
        bucket["correct_reason"] += int(reason == expected_reason)
        bucket["latency_ns"].append(elapsed)
        if accepted != expected or reason != expected_reason:
            trace_rows.append(
                {
                    "case": name,
                    "event_id": target.event_id.hex(),
                    "accepted": accepted,
                    "reason": reason,
                    "expected_accept": expected,
                    "expected_reason": expected_reason,
                }
            )

    def sign_tombstone(p: EvidencePackage, t: Tombstone) -> EvidencePackage:
        return dataclasses.replace(
            p,
            tombstone=t,
            tombstone_signature=fixture["custodians"][0].sign(DOMAIN_TOMBSTONE_SIGNATURE, t.encode()),
        )

    for index, (p, target) in enumerate(zip(fixture["packages"], fixture["targets"])):
        exercise("honest_post_erasure_package", p, target, True, "VERIFIED_POST_ERASURE")
        exercise(
            "first_declaration_registration", p, target, True, "REGISTERED_NEW_DECLARATION", "register"
        )
        exercise("exact_replay_registration", p, target, False, "DUPLICATE_DECLARATION", "register")
        exercise("repeat_inspection_after_registration", p, target, True, "VERIFIED_POST_ERASURE")
        exercise(
            "different_verifier_local_registration",
            p,
            target,
            True,
            "REGISTERED_NEW_DECLARATION",
            "register",
            other_verifier,
        )
        exercise(
            "modified_retained_evidence",
            dataclasses.replace(
                p, metadata=dataclasses.replace(p.metadata, sequence=p.metadata.sequence + 1)
            ),
            target,
            False,
            "INVALID_ORIGIN_SIGNATURE",
        )
        exercise(
            "modified_erased_commitment",
            dataclasses.replace(p, commitment=flip_first_byte(p.commitment)),
            target,
            False,
            "INVALID_ORIGIN_SIGNATURE",
        )
        exercise(
            "invalid_origin_signature",
            dataclasses.replace(p, origin_signature=flip_first_byte(p.origin_signature)),
            target,
            False,
            "INVALID_ORIGIN_SIGNATURE",
        )
        exercise(
            "invalid_tombstone_signature",
            dataclasses.replace(p, tombstone_signature=flip_first_byte(p.tombstone_signature)),
            target,
            False,
            "INVALID_TOMBSTONE_SIGNATURE",
        )
        proof = p.origin_proof
        if proof.siblings:
            mutated = dataclasses.replace(
                proof, siblings=(flip_first_byte(proof.siblings[0]), *proof.siblings[1:])
            )
            exercise(
                "modified_merkle_path",
                dataclasses.replace(p, origin_proof=mutated),
                target,
                False,
                "MERKLE_MEMBERSHIP_FAILURE",
            )
            truncated = dataclasses.replace(proof, siblings=proof.siblings[:-1])
            exercise(
                "truncated_merkle_path",
                dataclasses.replace(p, origin_proof=truncated),
                target,
                False,
                "MALFORMED_EVIDENCE:EvidenceFormatError",
            )
            direction = dataclasses.replace(proof, leaf_index=proof.leaf_index ^ 1)
            # The final unpaired leaf can have no right sibling.
            if direction.leaf_index < proof.leaf_count:
                exercise(
                    "swapped_merkle_direction",
                    dataclasses.replace(p, origin_proof=direction),
                    target,
                    False,
                    "MERKLE_MEMBERSHIP_FAILURE",
                )
        bad_record = dataclasses.replace(
            p.origin_anchor.record, root=flip_first_byte(p.origin_anchor.record.root)
        )
        exercise(
            "substituted_anchor_root",
            dataclasses.replace(
                p, origin_anchor=dataclasses.replace(p.origin_anchor, record=bad_record)
            ),
            target,
            False,
            "INVALID_ANCHOR_AUTHENTICITY",
        )
        exercise(
            "invalid_anchor_signature",
            dataclasses.replace(
                p,
                origin_anchor=dataclasses.replace(
                    p.origin_anchor, signature=flip_first_byte(p.origin_anchor.signature)
                ),
            ),
            target,
            False,
            "INVALID_ANCHOR_AUTHENTICITY",
        )
        exercise(
            "insufficient_finality_quorum",
            dataclasses.replace(
                p,
                origin_anchor=dataclasses.replace(
                    p.origin_anchor, votes=p.origin_anchor.votes[: trust.quorum - 1]
                ),
            ),
            target,
            False,
            "UNRESOLVED_ANCHOR_FINALITY",
        )
        duplicate_votes = (p.origin_anchor.votes[0],) * trust.quorum
        exercise(
            "duplicate_finality_signer",
            dataclasses.replace(
                p, origin_anchor=dataclasses.replace(p.origin_anchor, votes=duplicate_votes)
            ),
            target,
            False,
            "UNRESOLVED_ANCHOR_FINALITY",
        )
        invalid_votes = (
            (p.origin_anchor.votes[0][0], flip_first_byte(p.origin_anchor.votes[0][1])),
            *p.origin_anchor.votes[1:],
        )
        exercise(
            "invalid_finality_signature",
            dataclasses.replace(
                p, origin_anchor=dataclasses.replace(p.origin_anchor, votes=invalid_votes)
            ),
            target,
            False,
            "UNRESOLVED_ANCHOR_FINALITY",
        )
        t = dataclasses.replace(p.tombstone, origin_leaf=flip_first_byte(p.leaf))
        exercise(
            "authenticated_wrong_tombstone_linkage",
            sign_tombstone(p, t),
            target,
            False,
            "TOMBSTONE_LINKAGE_MISMATCH",
        )
        t = dataclasses.replace(p.tombstone, erased_at_ms=p.metadata.timestamp_ms - 1)
        exercise(
            "authenticated_tombstone_before_origin",
            sign_tombstone(p, t),
            target,
            False,
            "TEMPORAL_ORDER_FAILURE",
        )
        t = dataclasses.replace(p.tombstone, erased_at_ms=p.request.requested_at_ms - 1)
        exercise(
            "authenticated_tombstone_before_request",
            sign_tombstone(p, t),
            target,
            False,
            "TEMPORAL_ORDER_FAILURE",
        )
        t = dataclasses.replace(p.tombstone, erased_at_ms=p.request.expires_at_ms + 1)
        exercise(
            "authenticated_expired_request",
            sign_tombstone(p, t),
            target,
            False,
            "TEMPORAL_ORDER_FAILURE",
        )
        t = dataclasses.replace(p.tombstone, scope=p.tombstone.scope ^ 1)
        exercise(
            "authenticated_scope_mismatch",
            sign_tombstone(p, t),
            target,
            False,
            "UNAUTHORIZED_SCOPE_OR_CUSTODIAN",
        )
        t = dataclasses.replace(p.tombstone, erased_at_ms=p.tombstone.erased_at_ms + 1)
        exercise(
            "conflicting_declaration_identity",
            sign_tombstone(p, t),
            target,
            False,
            "CONFLICTING_DECLARATION",
            "register",
        )
        next_p = fixture["packages"][(index + 1) % FORMAL_MODEL_EVENTS]
        if FORMAL_MODEL_EVENTS > 1:
            exercise(
                "coherent_whole_witness_substitution", next_p, target, False, "AUDIT_TARGET_MISMATCH"
            )
            exercise(
                "cross_event_tombstone_substitution",
                dataclasses.replace(
                    p, tombstone=next_p.tombstone, tombstone_signature=next_p.tombstone_signature
                ),
                target,
                False,
                "TOMBSTONE_LINKAGE_MISMATCH",
            )
        exercise(
            "invalid_request_authority_signature",
            dataclasses.replace(p, request_signature=flip_first_byte(p.request_signature)),
            target,
            False,
            "MERKLE_MEMBERSHIP_FAILURE",
        )
        transport = p.to_dict()
        transport.pop("tombstone")
        exercise(
            "missing_required_artifact",
            transport,
            target,
            False,
            "MALFORMED_EVIDENCE:EvidenceFormatError",
        )
        exercise("canonical_transport_roundtrip", p.encode_json(), target, True, "VERIFIED_POST_ERASURE")
        if fixture["truth"][index]["application_payload_present"]:
            exercise("authenticated_false_erasure_declaration", p, target, True, "VERIFIED_POST_ERASURE")
        else:
            exercise("logical_application_erasure_observed", p, target, True, "VERIFIED_POST_ERASURE")

    replay.close()
    restart = ReplayStore(replay_path, reference_id("auditor-A"))
    restarted = EvidenceVerifier(trust, restart)
    for p, target in zip(fixture["packages"], fixture["targets"]):
        exercise(
            "replay_after_verifier_restart",
            p,
            target,
            False,
            "DUPLICATE_DECLARATION",
            "register",
            restarted,
        )
        exercise(
            "inspection_after_verifier_restart",
            p,
            target,
            True,
            "VERIFIED_POST_ERASURE",
            "inspect",
            restarted,
        )
    restart.close()
    other.close()
    rows = []
    for name, bucket in results.items():
        correct = bucket["accepted"] if bucket["expected_accept"] else bucket["rejected"]
        rows.append(
            {
                "case": name,
                "cases_tested": bucket["cases_tested"],
                "expected_accept": bucket["expected_accept"],
                "accepted": bucket["accepted"],
                "rejected": bucket["rejected"],
                "model_correct_classification_rate": correct / bucket["cases_tested"],
                "correct_reason_rate": bucket["correct_reason"] / bucket["cases_tested"],
                "reason_counts_json": json.dumps(bucket["reason_counts"], sort_keys=True),
                "median_latency_us": float(np.median(bucket["latency_ns"])) / 1000,
                "p99_latency_us": float(np.quantile(bucket["latency_ns"], 0.99)) / 1000,
                "latency_cache_policy": "bounded immutable signature-result cache; trust and replay never cached",
            }
        )
    frame = pd.DataFrame(rows)
    write_csv(frame, DATA / "formal_model_verification.csv")
    write_csv(
        pd.DataFrame(
            trace_rows,
            columns=["case", "event_id", "accepted", "reason", "expected_accept", "expected_reason"],
        ),
        DATA / "protocol_test_failures.csv",
    )
    write_csv(pd.DataFrame(fixture["truth"]), DATA / "false_declaration_ground_truth.csv")
    sample = fixture["packages"][0]
    (DATA / "worked_witness.json").write_bytes(sample.encode_json() + b"\n")
    (DATA / "worked_audit_target.json").write_text(
        json.dumps(
            {
                k: v.hex() if isinstance(v, bytes) else v
                for k, v in asdict(fixture["targets"][0]).items()
            },
            indent=2,
        )
        + "\n"
    )
    (DATA / "public_trust_registry.json").write_text(
        json.dumps(trust.public_manifest(), indent=2) + "\n"
    )
    export_reference_measurements(fixture)
    invalid = frame[~frame.expected_accept]
    metrics = pd.DataFrame(
        [
            {
                "formal_model_events": FORMAL_MODEL_EVENTS,
                "merkle_tree_height": max(len(p.origin_proof.siblings) for p in fixture["packages"]),
                "merkle_path_bytes": len(sample.origin_proof.encode()),
                "honest_post_erasure_acceptance_rate": float(
                    frame.loc[
                        frame["case"] == "honest_post_erasure_package",
                        "model_correct_classification_rate",
                    ].iloc[0]
                ),
                "modeled_invalid_package_acceptance_rate": float(
                    invalid.accepted.sum() / invalid.cases_tested.sum()
                ),
                "modeled_tamper_or_missing_artifact_detection_rate": float(
                    1 - invalid.accepted.sum() / invalid.cases_tested.sum()
                ),
                "erased_payload_bytes_required_by_verifier": 0,
                "cryptographic_runtime_measured": True,
                "signatures_actually_verified": True,
                "persistent_replay_exercised": True,
                "physical_deletion_proven": False,
                "legal_compliance_proven": False,
            }
        ]
    )
    write_csv(metrics, DATA / "formal_model_metrics.csv")
    write_csv(
        pd.DataFrame(
            [
                {
                    "assumption": "SHA-256 binding and computational hiding",
                    "role": "256-bit independent CSPRNG opening remains secret and unavailable; hash collision/preimage assumptions are distinct",
                    "evaluated_by": "actual hashing and independent byte vectors; no cryptanalytic proof",
                },
                {
                    "assumption": "Ed25519 and trusted role registry",
                    "role": "source authentication under uncompromised role-bound keys",
                    "evaluated_by": "real signing and verification, revocation and fault tests",
                },
                {
                    "assumption": "local quorum anchor protocol",
                    "role": "n=4,f=1,q=3; persistent fixed-view validator locks and certificates",
                    "evaluated_by": "executed in-process protocol and fault cases; no distributed benchmark",
                },
                {
                    "assumption": "retained metadata remains linkable",
                    "role": "opaque 128-bit references are not an anonymity guarantee",
                    "evaluated_by": "field catalogue and bounded identifier-linkage experiment",
                },
                {
                    "assumption": "physical deletion and custodian truthfulness",
                    "role": "outside structural verifier observation",
                    "evaluated_by": "explicit false-declaration fixture with separate application-store truth",
                },
            ]
        ),
        DATA / "formal_model_assumptions.csv",
    )
    protocol_conformance_experiments(fixture)
    fixture["store"].close()
    fixture["ledger"].close()
    execution.cleanup()
    if trace_rows:
        raise RuntimeError(
            f"{len(trace_rows)} protocol outcomes disagreed with their exact expected result; see protocol_test_failures.csv"
        )
    return frame, metrics


def protocol_conformance_experiments(fixture: Mapping[str, Any]) -> None:
    """Independent byte vectors, partial trees, persistence and quorum faults."""
    rows: list[dict[str, Any]] = []

    def record(name: str, passed: bool, observed: Any) -> None:
        rows.append({"check": name, "passed": bool(passed), "observed": str(observed)})

    arrivals = [120, 0, 250, 55, 60, 90, 200, 1]
    backlog, expected = 0, []
    for count in arrivals:
        backlog = max(0, backlog + count - 60)
        expected.append(backlog)
    result = queue_metrics(arrivals, 1, measurement_start=2, measurement_end=7)
    exact_queue = (
        result["initial_measurement_backlog_events"] == expected[1]
        and result["ending_backlog_events"] == expected[6]
        and result["max_backlog_events"] == max(expected[2:7])
    )
    record("vectorized_queue_matches_lindley_recursion", exact_queue, result["ending_backlog_events"])
    states = sample_mmpp_states(200_000, experiment_rng("markov_dwell_conformance"))
    occupancy = np.bincount(states, minlength=len(MMPP_STATIONARY)) / len(states)
    record(
        "geometric_dwell_occupancy_matches_stationary_model",
        bool(np.max(np.abs(occupancy - MMPP_STATIONARY)) < 0.035),
        occupancy.tolist(),
    )

    m = fixture["packages"][0].metadata
    independent = (
        b"HTGL"
        + b"\x00\x02\x01"
        + (84 + 16 * len(m.actors)).to_bytes(4, "big")
        + m.event_id
        + m.tenant_id
        + m.event_type.to_bytes(2, "big")
        + len(m.actors).to_bytes(2, "big")
        + b"".join(m.actors)
        + m.resource_id
        + m.policy_id
        + m.timestamp_ms.to_bytes(8, "big")
        + m.sequence.to_bytes(8, "big")
    )
    record("independent_metadata_binary_vector", independent == m.encode(), len(independent))
    record("metadata_decode_roundtrip", EvidenceMetadata.decode(m.encode()) == m, len(m.encode()))
    for name, bad in (
        ("trailing_bytes", m.encode() + b"\x00"),
        ("truncated_bytes", m.encode()[:-1]),
        ("unsupported_version", m.encode()[:4] + b"\x00\x01" + m.encode()[6:]),
    ):
        try:
            EvidenceMetadata.decode(bad)
            rejected = False
        except EvidenceFormatError:
            rejected = True
        record("reject_" + name, rejected, "malformed byte record")
    try:
        dataclasses.replace(m, sequence=True).encode()
        rejected = False
    except EvidenceFormatError:
        rejected = True
    record("reject_boolean_as_integer", rejected, "strict unsigned integer encoding")
    tree_rows = []
    for size in (1, 2, 3, 5, 17, 999, 1000, 1001, 5000, 10000):
        leaves = [sha256_bytes(b"independent-tree-fixture" + i.to_bytes(4, "big")) for i in range(size)]
        levels, bid = build_merkle_levels(leaves), reference_id(f"conformance-{size}")
        correct = 0
        for i, leaf in enumerate(leaves):
            proof = make_membership_proof(levels, i, bid)
            correct += int(
                proof.verify(leaf, levels[-1][0]) and MembershipProof.decode(proof.encode()) == proof
            )
        record(f"merkle_all_leaves_{size}", correct == size, f"{correct}/{size}")
        tree_rows.append(
            {
                "leaf_count": size,
                "verified_leaves": correct,
                "levels": len(levels),
                "node_count": sum(map(len, levels)),
                "root_hex": levels[-1][0].hex(),
            }
        )

    verifier = EvidenceVerifier(fixture["trust"])
    p, target = fixture["packages"][0], fixture["targets"][0]
    accepted, reason = verifier.verify(
        p, dataclasses.replace(target, origin_leaf=flip_first_byte(target.origin_leaf))
    )
    record(
        "independent_origin_leaf_binding",
        not accepted and reason == "AUDIT_ORIGIN_LEAF_MISMATCH",
        reason,
    )
    accepted, reason = verifier.verify(
        p, dataclasses.replace(target, request_leaf=flip_first_byte(target.request_leaf))
    )
    record(
        "independent_request_leaf_binding",
        not accepted and reason == "AUDIT_REQUEST_LEAF_MISMATCH",
        reason,
    )
    custody_packages, custody_targets = [], []
    for custodian in fixture["custodians"]:
        tombstone = dataclasses.replace(p.tombstone, custodian_id=custodian.identity)
        custody_packages.append(
            dataclasses.replace(
                p,
                tombstone=tombstone,
                tombstone_signature=custodian.sign(DOMAIN_TOMBSTONE_SIGNATURE, tombstone.encode()),
            )
        )
        custody_targets.append(dataclasses.replace(target, custodian_id=custodian.identity))
    complete = verifier.verify_case(custody_packages, custody_targets)
    partial = verifier.verify_case(custody_packages[:1], custody_targets[:1])
    duplicate = verifier.verify_case(
        [custody_packages[0], custody_packages[0]], [custody_targets[0], custody_targets[0]]
    )
    record("declared_roster_complete", complete["accepted"] and complete["complete"], complete)
    record("declared_roster_missing_custodian", partial["accepted"] and not partial["complete"], partial)
    record("duplicate_custodian_cannot_complete_case", not duplicate["accepted"], duplicate)

    revoked = fixture["origin"].identity
    fixture["trust"].revoked.add(revoked)
    accepted, reason = verifier.verify(p, target)
    fixture["trust"].revoked.remove(revoked)
    record(
        "trust_revocation_not_bypassed_by_crypto_cache",
        not accepted and reason == "INVALID_ORIGIN_SIGNATURE",
        reason,
    )

    # Exercise leaf reconstruction independently of authenticity by having a
    # genuinely authorized origin sign a deliberately inconsistent leaf.
    inconsistent = dataclasses.replace(p, commitment=flip_first_byte(p.commitment))
    inconsistent = dataclasses.replace(
        inconsistent,
        origin_signature=fixture["origin"].sign(
            DOMAIN_ORIGIN_SIGNATURE,
            inconsistent.metadata.encode() + inconsistent.commitment + inconsistent.leaf,
        ),
    )
    accepted, reason = verifier.verify(inconsistent, target)
    record(
        "signed_inconsistent_leaf_rejected",
        not accepted and reason == "STRUCTURED_COMMITMENT_MISMATCH",
        reason,
    )

    # Request authorization is exercised independently of membership: the
    # local log anchors a byte-consistent request with a wrong authority sig.
    malformed_request_signature = flip_first_byte(p.request_signature)
    new_leaf = request_leaf(p.request, malformed_request_signature)
    bid = reference_id("request-unauthorized-independent")
    at = p.tombstone.erased_at_ms
    anchor = fixture["ledger"].append(bid, new_leaf, 1, at)
    invalid_request = dataclasses.replace(
        p,
        request_signature=malformed_request_signature,
        request_proof=MembershipProof(bid, 1, 0, ()),
        request_anchor=anchor,
    )
    accepted, reason = verifier.verify(
        invalid_request, dataclasses.replace(target, request_batch_id=bid, request_leaf=new_leaf)
    )
    record(
        "anchored_unauthorized_request_rejected",
        not accepted and reason == "INVALID_REQUEST_AUTHORIZATION",
        reason,
    )

    root = fixture["ledger"].path.parent / "protocol_conformance"
    root.mkdir(exist_ok=True)
    registry = fixture["trust"]
    fault_ledger = QuorumAnchorLedger(
        root / "quorum.sqlite", fixture["anchor_signer"], fixture["validators"], registry
    )
    validator_ids = [v.identity for v in fixture["validators"]]
    one_offline = fault_ledger.append(
        reference_id("one-offline"),
        sha256_bytes(b"one"),
        1,
        at,
        unavailable=frozenset(validator_ids[:1]),
    )
    record("quorum_tolerates_one_unavailable", one_offline.finality(registry), len(one_offline.votes))
    previous_height = fault_ledger.head()[0]
    try:
        fault_ledger.append(
            reference_id("two-offline"),
            sha256_bytes(b"two"),
            1,
            at + 1,
            unavailable=frozenset(validator_ids[:2]),
        )
        failed = False
    except EvidenceFormatError:
        failed = True
    record(
        "no_finality_without_quorum",
        failed and fault_ledger.head()[0] == previous_height,
        fault_ledger.head()[0],
    )
    # The two validators locked to the failed proposal are honest. Even one
    # equivocator cannot obtain a conflicting quorum at that height.
    try:
        fault_ledger.append(
            reference_id("conflicting-proposal"),
            sha256_bytes(b"conflict"),
            1,
            at + 2,
            equivocate=frozenset(validator_ids[:1]),
        )
        failed = False
    except EvidenceFormatError:
        failed = True
    record("honest_prepare_locks_block_conflicting_quorum", failed, "fixed-view persisted locks")
    try:
        fault_ledger.connection.execute("DELETE FROM anchors")
        protected = False
    except sqlite3.IntegrityError:
        protected = True
    record(
        "anchor_log_append_only_trigger", protected, "UPDATE/DELETE denied through reference interface"
    )
    record("local_anchor_history_verified", fault_ledger.verify_history(), fault_ledger.head()[0])
    retry = fault_ledger.append(one_offline.record.batch_id, one_offline.record.root, 1, at)
    record(
        "anchor_idempotent_retry",
        retry.digest == one_offline.digest and fault_ledger.head()[0] == previous_height,
        retry.record.height,
    )
    try:
        fault_ledger.append(one_offline.record.batch_id, sha256_bytes(b"conflicting same ID"), 1, at)
        rejected_conflict = False
    except EvidenceFormatError:
        rejected_conflict = True
    record(
        "conflicting_anchor_batch_identity_rejected",
        rejected_conflict,
        "existing batch ID cannot name another root",
    )
    fault_ledger.close()
    reopened = QuorumAnchorLedger(
        root / "quorum.sqlite", fixture["anchor_signer"], fixture["validators"], registry
    )
    record(
        "anchor_history_survives_restart",
        reopened.verify_history()
        and reopened.lookup(one_offline.record.batch_id).digest == one_offline.digest,
        reopened.head()[0],
    )
    reopened.close()

    # Two independent SQLite connections race to admit the same declaration.
    concurrent_path = root / "concurrent_replay.sqlite"

    def admit() -> tuple[bool, str]:
        local = ReplayStore(concurrent_path, reference_id("concurrent-auditor"))
        try:
            return local.register(p.tombstone)
        finally:
            local.close()

    setup = ReplayStore(concurrent_path, reference_id("concurrent-auditor"))
    setup.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        admissions = list(pool.map(lambda _: admit(), range(2)))
    record(
        "concurrent_registration_exactly_once",
        sum(a for a, _ in admissions) == 1
        and sorted(r for _, r in admissions) == ["DUPLICATE_DECLARATION", "REGISTERED_NEW_DECLARATION"],
        admissions,
    )

    # Deterministic, public vectors are conformance inputs only. Actual
    # fixtures above use independent CSPRNG openings and generated keys.
    vector_metadata = EvidenceMetadata(
        bytes(range(16)),
        bytes(range(16, 32)),
        2,
        (bytes(range(32, 48)),),
        bytes(range(48, 64)),
        bytes(range(64, 80)),
        123456789,
        42,
    )
    vector_opening, vector_payload = bytes(range(32)), b"HTGL public test vector; never a secret"
    vector_commitment = hashlib.sha256(
        DOMAIN_COMMITMENT + vector_opening + len(vector_payload).to_bytes(8, "big") + vector_payload
    ).digest()
    vector_leaf = hashlib.sha256(DOMAIN_LEAF + vector_metadata.encode() + vector_commitment).digest()
    record(
        "independent_commitment_vector",
        payload_commitment(vector_payload, vector_opening) == vector_commitment,
        vector_commitment.hex(),
    )
    record(
        "independent_leaf_vector",
        structured_leaf(vector_metadata, vector_commitment) == vector_leaf,
        vector_leaf.hex(),
    )
    vectors = {
        "classification": "public conformance vectors, not confidential fixture material",
        "metadata_hex": vector_metadata.encode().hex(),
        "opening_hex": vector_opening.hex(),
        "payload_hex": vector_payload.hex(),
        "commitment_hex": vector_commitment.hex(),
        "leaf_hex": vector_leaf.hex(),
    }
    (DATA / "canonical_test_vectors.json").write_text(json.dumps(vectors, indent=2) + "\n")
    write_csv(pd.DataFrame(rows), DATA / "protocol_conformance_checks.csv")
    write_csv(pd.DataFrame(tree_rows), DATA / "merkle_construction_cases.csv")
    (DATA / "custodian_coverage_cases.json").write_text(
        json.dumps({"complete": complete, "partial": partial, "duplicate": duplicate}, indent=2) + "\n"
    )
    if not all(r["passed"] for r in rows):
        raise RuntimeError(
            "Protocol conformance failure: " + ", ".join(r["check"] for r in rows if not r["passed"])
        )


class WitnessRepository:
    """Four actual serialized retention representations of the same tree."""

    POLICIES = (
        "independent_encoded_paths",
        "shared_tree_nodes",
        "leaves_on_demand",
        "compressed_path_bundle",
    )

    def __init__(self, leaves: Sequence[bytes], batch_id: bytes, policy: str):
        if policy not in self.POLICIES:
            raise ValueError("unknown witness retention policy")
        self.policy, self.batch_id, self.count = policy, batch_id, len(leaves)
        levels = build_merkle_levels(leaves)
        self.root = levels[-1][0]
        if policy in ("independent_encoded_paths", "compressed_path_bundle"):
            paths = [make_membership_proof(levels, i, batch_id).encode() for i in range(len(leaves))]
            body = struct.pack(">II", len(paths), len(paths[0])) + b"".join(paths)
            self.blob = frame_record(
                7, body if policy == "independent_encoded_paths" else zlib.compress(body, 6)
            )
        elif policy == "shared_tree_nodes":
            self.blob = frame_record(
                8,
                batch_id
                + struct.pack(">IH", len(leaves), len(levels))
                + b"".join(struct.pack(">I", len(level)) + b"".join(level) for level in levels),
            )
        else:
            self.blob = frame_record(9, batch_id + struct.pack(">I", len(leaves)) + b"".join(leaves))

    def proof(self, index: int) -> MembershipProof:
        if not 0 <= index < self.count:
            raise EvidenceFormatError("proof index outside retained batch")
        if self.policy in ("independent_encoded_paths", "compressed_path_bundle"):
            body = unframe_record(self.blob, 7).data
            if self.policy == "compressed_path_bundle":
                body = zlib.decompress(body)
            count, length = struct.unpack(">II", body[:8])
            if count != self.count or len(body) != 8 + count * length:
                raise EvidenceFormatError("corrupt path repository")
            return MembershipProof.decode(body[8 + index * length : 8 + (index + 1) * length])
        if self.policy == "shared_tree_nodes":
            r = unframe_record(self.blob, 8)
            bid, count, depth = r.take(16), r.uint(32), r.uint(16)
            levels = []
            for _ in range(depth):
                width = r.uint(32)
                levels.append([r.take(32) for _ in range(width)])
            r.finish()
            if bid != self.batch_id or count != self.count or len(levels[0]) != count:
                raise EvidenceFormatError("corrupt shared-node repository")
        else:
            r = unframe_record(self.blob, 9)
            bid, count = r.take(16), r.uint(32)
            if bid != self.batch_id or count != self.count:
                raise EvidenceFormatError("corrupt leaf repository")
            levels = build_merkle_levels([r.take(32) for _ in range(count)])
            r.finish()
        return make_membership_proof(levels, index, self.batch_id)


def retention_policy_experiments() -> pd.DataFrame:
    rows = []
    for capacity in sorted(set(BATCH_SIZES) | {3, 17, 999}):
        leaves = [sha256_bytes(b"retention-fixture" + i.to_bytes(4, "big")) for i in range(capacity)]
        bid = reference_id(f"retention-{capacity}")
        for policy in WitnessRepository.POLICIES:
            start = time.perf_counter_ns()
            repository = WitnessRepository(leaves, bid, policy)
            build_ns = time.perf_counter_ns() - start
            samples = np.unique(np.linspace(0, capacity - 1, min(capacity, 32), dtype=int))
            latencies, sizes = [], []
            for i in samples:
                start = time.perf_counter_ns()
                proof = repository.proof(int(i))
                latencies.append((time.perf_counter_ns() - start) / 1000)
                if not proof.verify(leaves[int(i)], repository.root):
                    raise RuntimeError("Retained witness representation produced an invalid proof")
                sizes.append(len(proof.encode()))
            rows.append(
                {
                    "batch_size": capacity,
                    "policy": policy,
                    "retained_bytes_per_batch": len(repository.blob),
                    "retained_bytes_per_event": len(repository.blob) / capacity,
                    "root_hex": repository.root.hex(),
                    "proofs_verified": len(samples),
                    "build_ms": build_ns / 1e6,
                    "median_proof_generation_us": float(np.median(latencies)),
                    "p99_proof_generation_us": float(np.quantile(latencies, 0.99)),
                    "mean_disclosed_proof_bytes": float(np.mean(sizes)),
                    "requires_retained_repository_for_new_proof": policy != "independent_encoded_paths",
                    "availability_boundary": "new proofs unavailable if repository is lost; already exported independent proofs remain inspectable",
                    "scaling_method": "measured serialized full-batch bytes extrapolated to analytical batch count",
                }
            )
    result = pd.DataFrame(rows)
    write_csv(result, DATA / "retention_policy_measurements.csv")
    return result


def batching_implementation_experiments() -> pd.DataFrame:
    rows = []
    count = min(FORMAL_MODEL_EVENTS, 4096)
    leaves = [sha256_bytes(b"batcher-fixture" + i.to_bytes(8, "big")) for i in range(count)]
    for intensity in (0.2, 10.0, 1000.0):
        generator = experiment_rng("event_driven_batcher", intensity)
        arrivals = np.cumsum(generator.exponential(1.0 / intensity, size=count))
        for capacity in BATCH_SIZES:
            for window in BATCH_WINDOWS_SECONDS:
                batcher, closed = CapacityTimeoutBatcher(capacity, window), []
                for i, (arrival, leaf) in enumerate(zip(arrivals, leaves)):
                    closed.extend(batcher.push(float(arrival), i, leaf))
                closed.extend(batcher.finish(float(arrivals[-1]) + window))
                waits = [batch.closed_at - arrival for batch in closed for arrival in batch.arrivals]
                verified = 0
                for batch in closed:
                    levels = build_merkle_levels(batch.leaves)
                    for i in {0, len(batch.leaves) - 1}:
                        verified += int(
                            make_membership_proof(levels, i, batch.batch_id).verify(
                                batch.leaves[i], levels[-1][0]
                            )
                        )
                if (
                    sum(len(b.leaves) for b in closed) != count
                    or min(waits) < -1e-9
                    or max(waits) > window + 1e-9
                ):
                    raise RuntimeError("Event-driven batcher violated conservation or timeout bound")
                rows.append(
                    {
                        "source": "synthetic event-level Poisson fixture",
                        "events": count,
                        "offered_events_per_second": intensity,
                        "batch_size": capacity,
                        "timeout_seconds": window,
                        "batch_count": len(closed),
                        "capacity_closures": sum(b.reason == "capacity" for b in closed),
                        "timeout_closures": sum(b.reason == "timeout" for b in closed),
                        "mean_batch_wait_seconds": float(np.mean(waits)),
                        "p99_batch_wait_seconds": float(np.quantile(waits, 0.99)),
                        "maximum_batch_wait_seconds": max(waits),
                        "boundary_proofs_verified": verified,
                        "consensus_and_finality_latency_included": False,
                        "population_scale_performance_claim": False,
                    }
                )
    result = pd.DataFrame(rows)
    write_csv(result, DATA / "event_driven_batching_metrics.csv")
    return result


def metadata_linkage_experiments() -> None:
    generator = experiment_rng("metadata_linkage")
    secret = secrets.token_bytes(32)
    count = 10000
    left = generator.integers(0, 1000, count)
    same = generator.random(count) < 0.5
    right = np.where(same, left, (left + generator.integers(1, 1000, count)) % 1000)
    left_domain, right_domain = generator.integers(0, 4, count), generator.integers(0, 4, count)
    left_epoch, right_epoch = generator.integers(0, 7, count), generator.integers(0, 7, count)

    def identifier(principal: int, domain: int, epoch: int, policy: str) -> bytes:
        context = b"global" if policy == "stable_global" else domain.to_bytes(4, "big")
        if policy == "custodian_epoch_scoped":
            context += epoch.to_bytes(4, "big")
        return hmac.new(
            secret, b"HTGL-PSEUDONYM-v2\x00" + context + principal.to_bytes(8, "big"), hashlib.sha256
        ).digest()[:16]

    rows = []
    for policy in ("stable_global", "custodian_scoped", "custodian_epoch_scoped"):
        linked = np.array(
            [
                identifier(int(a), int(d), int(e), policy) == identifier(int(b), int(f), int(g), policy)
                for a, b, d, f, e, g in zip(
                    left, right, left_domain, right_domain, left_epoch, right_epoch
                )
            ]
        )
        rows.append(
            {
                "identifier_policy": policy,
                "pairs": count,
                "true_same_principal_pairs": int(same.sum()),
                "equality_linkage_true_positive_rate": float(linked[same].mean()),
                "equality_linkage_false_positive_rate": float(linked[~same].mean()),
                "adversary": "identifier equality only; no key or mapping access; timing/frequency/auxiliary inference excluded",
                "anonymity_or_legal_compliance_established": False,
            }
        )
    write_csv(pd.DataFrame(rows), DATA / "metadata_linkage_metrics.csv")
    fields = [
        (
            "event_id",
            "128-bit opaque reference",
            "stable within declared event",
            "may identify a person via mapping",
            "authorized evidence reader",
            "audit horizon",
        ),
        (
            "tenant_id",
            "128-bit opaque reference",
            "stable organization/tenant",
            "may be personal for sole traders",
            "authorized evidence reader",
            "audit horizon",
        ),
        (
            "actors",
            "ordered 128-bit references",
            "stable within selected identifier domain",
            "may be stable personal/pseudonymous identifiers",
            "role-limited resolver",
            "audit horizon",
        ),
        (
            "resource_id",
            "128-bit opaque reference",
            "stable target resource",
            "meter/resource can link to household",
            "role-limited resolver",
            "audit horizon",
        ),
        (
            "policy_id",
            "128-bit opaque reference",
            "stable policy version",
            "auxiliary join possible",
            "authorized evidence reader",
            "policy/audit horizon",
        ),
        (
            "timestamp_ms",
            "uint64 UTC milliseconds",
            "time/frequency correlation remains",
            "behavioral inference possible",
            "authorized evidence reader",
            "audit horizon",
        ),
        (
            "sequence",
            "uint64",
            "batch/event ordering",
            "ordering correlation remains",
            "authorized evidence reader",
            "audit horizon",
        ),
        (
            "commitment",
            "SHA-256 digest with secret 256-bit opening",
            "binding to origin; secrecy assumption separate",
            "not automatically anonymous",
            "authorized evidence reader",
            "audit horizon",
        ),
        (
            "scope",
            "uint32 application-defined bitmask",
            "matches signed authorized request",
            "scope/context may be revealing",
            "authorized evidence reader",
            "request/declaration horizon",
        ),
    ]
    write_csv(
        pd.DataFrame(
            fields,
            columns=[
                "field",
                "representation",
                "linkability",
                "personal_data_possibility",
                "access_model",
                "retention_policy",
            ],
        ),
        DATA / "retained_metadata_catalogue.csv",
    )


def export_protocol_specification() -> None:
    domains = {
        name: globals()[name].hex()
        for name in (
            "DOMAIN_COMMITMENT",
            "DOMAIN_LEAF",
            "DOMAIN_MERKLE",
            "DOMAIN_REQUEST",
            "DOMAIN_ORIGIN_SIGNATURE",
            "DOMAIN_REQUEST_SIGNATURE",
            "DOMAIN_TOMBSTONE_SIGNATURE",
            "DOMAIN_ANCHOR_SIGNATURE",
            "DOMAIN_PREPARE",
            "DOMAIN_COMMIT",
        )
    }
    specification = {
        "protocol_version": PROTOCOL_VERSION,
        "frame": "ASCII HTGL (4), version uint16 BE, kind uint8, body_length uint32 BE, body",
        "integer_encoding": "unsigned big-endian; Python bool is rejected as an integer",
        "identifier_bytes": 16,
        "timestamp": "uint64 UTC milliseconds since Unix epoch",
        "domains_hex": domains,
        "metadata_fields": [
            "event_id:16",
            "tenant_id:16",
            "event_type:u16",
            "actor_count:u16",
            "ordered_actors:16 each",
            "resource_id:16",
            "policy_id:16",
            "timestamp_ms:u64",
            "sequence:u64",
        ],
        "request_fields": [
            "request_id:16",
            "tenant_id:16",
            "event_id:16",
            "origin_leaf:32",
            "resource_id:16",
            "policy_id:16",
            "authority_id:16",
            "custodian_count:u16",
            "sorted_custodians:16 each",
            "scope:u32",
            "requested_at_ms:u64",
            "expires_at_ms:u64",
            "version:u32",
        ],
        "tombstone_fields": [
            "tenant_id:16",
            "event_id:16",
            "origin_leaf:32",
            "request_id:16",
            "custodian_id:16",
            "resource_id:16",
            "scope:u32",
            "erased_at_ms:u64",
            "version:u32",
        ],
        "anchor_fields": [
            "ledger_id:16",
            "batch_id:16",
            "root:32",
            "leaf_count:u32",
            "created_at_ms:u64",
            "height:u64",
            "previous_digest:32",
            "signer_id:16",
        ],
        "proof_fields": [
            "batch_id:16",
            "leaf_count:u32",
            "leaf_index:u32",
            "depth:u16",
            "bottom_up_siblings:32 each",
        ],
        "proof_direction": "derived from leaf_index bits; odd final node duplicated and checked against running hash",
        "opening": "32 independent OS-CSPRNG bytes per payload, confidential application-plane state, never in evidence",
        "commitment_preimage": "domain || opening || payload_length:u64 || payload",
        "confidentiality_assumptions": "unpredictable unavailable opening, no leaked copy, appropriate computational hash preimage/hiding assumptions; collision resistance alone is insufficient",
        "replay_identity": "SHA256(identity-domain || tenant || event || request || custodian || version:u32)",
        "replay_digest": "SHA256(canonical unsigned tombstone); same identity/digest duplicate, different digest conflict",
        "inspection": "repeatable stateless validation, never consumes freshness",
        "independent_audit_target": "tenant, event, resource, policy, request, custodian, scope, both batch IDs and exact origin/request leaves",
        "registration": "persistent atomic SQLite admission after all validation; verifier namespace is local; restart preserves state; another verifier may independently register",
        "finality": "executed local n=4 f=1 q=3 fixed-view prepare/commit signatures and durable height locks; not a deployed consensus performance claim",
        "transport": "strict JSON object with exact keys; canonical binary records carried as lowercase hex; duplicates/unknown fields rejected",
        "new_guarantees": "signed request membership, declared-scope authorization and declared-roster coverage; no coverage of undisclosed custodians or physical deletion",
    }
    (DATA / "evidence_protocol_specification.json").write_text(
        json.dumps(specification, indent=2) + "\n"
    )


def export_reviewer_traceability(trace_supplied: bool) -> None:
    checks = [
        (
            "R1.1",
            "Canonical witness, domain bytes and opening generation",
            "Versioned binary codec, exact domain bytes, OS CSPRNG and explicit confidentiality assumptions",
            "evidence_protocol_specification.json; canonical_test_vectors.json; protocol_conformance_checks.csv",
            "executed",
        ),
        (
            "R1.2",
            "Record-size derivation and redundant leaf",
            "Field-level legacy sizing assumptions plus independently summed measured codec sizes, with leaf counted explicitly",
            "record_size_field_breakdown.csv; record_size_totals.csv",
            "executed",
        ),
        (
            "R1.3",
            "Component and coherent whole-witness substitution",
            "Independently supplied target binds tenant, event, resource, policy, request, custodian, scope and batch contexts",
            "formal_model_verification.csv; worked_audit_target.json",
            "executed",
        ),
        (
            "R1.4",
            "Replay identity, restart and matrix",
            "Canonical declaration identity with atomic per-verifier admission and exact reason checks",
            "formal_model_verification.csv; protocol_conformance_checks.csv",
            "executed",
        ),
        (
            "R1.5",
            "Retained metadata and linkability",
            "Versioned field catalogue and identifier equality linkage under explicit visibility assumptions",
            "retained_metadata_catalogue.csv; metadata_linkage_metrics.csv",
            "executed for the stated linkage model",
        ),
        (
            "R1.6",
            "Workload ranges and justification",
            "Event-granularity rationale, declared MMPP dwell assumptions, LHS and uncertainty outputs",
            "workload_assumption_rationale.csv; burst_assumption_rationale.csv; uncertainty_summary.csv",
            "executed design study",
        ),
        (
            "R1.7",
            "3,500 TPS reference provenance",
            "Configuration-specific external point labelled and cited in figures and source register",
            "assumptions_and_sources.csv; paper_artifacts/figure_captions.md",
            "documented literature comparison",
        ),
        (
            "R1.8",
            "Comparison with Basu and Yang",
            "Literature comparison belongs in the manuscript discussion",
            "",
            "outside simulation scope",
        ),
        (
            "R2.1",
            "Repeat audit versus replay",
            "Repeat inspection, duplicate/conflicting registration, restart and independent verifier tests",
            "formal_model_verification.csv",
            "executed",
        ),
        (
            "R2.2",
            "Request authority, scope and custodian coverage",
            "Anchored signed requests, exact scope matching, declared-roster coverage and separate false-declaration ground truth",
            "custodian_coverage_cases.json; protocol_conformance_checks.csv; false_declaration_ground_truth.csv",
            "executed for declared custody",
        ),
        (
            "R2.3",
            "Checker reproducibility and external predicates",
            "Portable worked witness and public trust registry; real Ed25519 checks, quorum certificate checks, exact fault counts",
            "worked_witness.json; public_trust_registry.json; formal_model_verification.csv",
            "executed local reference backend",
        ),
        (
            "R2.4",
            "Operational trace or end-to-end implementation",
            "Complete local reference pipeline, trace ingestion, observed event-time replay and training/holdout comparison",
            "local_reference_pipeline_timings.csv; trace_conformance/; trace_replay/ if supplied",
            (
                "trace supplied"
                if trace_supplied
                else "local implementation and synthetic trace fixture executed"
            ),
        ),
        (
            "R2.4 storage",
            "Compact witness comparison",
            "Measured encoded independent paths, shared nodes, leaf-only regeneration and compressed paths; exact retrieved proof verification",
            "retention_policy_measurements.csv; storage_policy_comparison.csv; event_class_retention_comparison.csv",
            "executed",
        ),
        (
            "R2 minor",
            "3.17 versus 3.16 anchors/s",
            "Deterministic and temporal estimators exported side by side without rounding ambiguity",
            "paper_batching_reconciliation.csv",
            "executed",
        ),
        (
            "R3 workload",
            "Synthetic versus real workloads and scale",
            "Separate analytical design envelopes, stochastic stress estimates and optional trace replay",
            "temporal_scenario_summary.csv; trace_conformance/; reproducibility_manifest.csv",
            "conditional model plus trace path",
        ),
        (
            "R3 storage",
            "Retention and replica cost",
            "Path-inclusive accounting, actual compact proof representations, class horizons and 1/3/5 copies",
            "storage_policy_comparison.csv; event_class_retention_comparison.csv",
            "executed",
        ),
        (
            "R3 authenticity",
            "Practical authenticity and finality validation",
            "Local durable authenticated log, separate role-bound signatures, quorum faults and measured local verification",
            "protocol_conformance_checks.csv; local_reference_verification_timings.csv",
            "local backend executed; network performance remains unmeasured",
        ),
    ]
    write_csv(
        pd.DataFrame(
            checks, columns=["review_comment", "concern", "implemented_response", "evidence", "status"]
        ),
        DATA / "reviewer_comment_traceability.csv",
    )


def utc_timestamp(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp with timezone") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


TRACE_COLUMNS = ("timestamp", "event_id", "event_class", "parent_event_id")


def load_governance_trace(path: Path, start: datetime, end: datetime) -> pd.DataFrame:
    if (end - start).total_seconds() < 7200:
        raise ValueError("trace exposure window must span at least two hours")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError("trace CSV cannot be parsed") from exc
    if (
        set(frame.columns) != set(TRACE_COLUMNS)
        or len(frame.columns) != len(TRACE_COLUMNS)
        or frame.empty
    ):
        raise ValueError(
            "trace requires timestamp,event_id,event_class,parent_event_id and at least one row"
        )
    if (frame.event_id.str.len() == 0).any() or frame.event_id.duplicated().any():
        raise ValueError("trace event IDs must be present and unique")
    if not frame.event_class.isin(COMPONENT_COLUMNS).all():
        raise ValueError("trace contains an unknown event class")
    times = [utc_timestamp(value, "trace timestamp") for value in frame.timestamp]
    if any(not start <= timestamp < end for timestamp in times):
        raise ValueError("trace timestamp falls outside the declared half-open exposure window")
    frame["seconds_from_start"] = [(timestamp - start).total_seconds() for timestamp in times]
    is_child = frame.event_class.isin(COMPONENT_COLUMNS[5:])
    if (frame.loc[~is_child, "parent_event_id"] != "").any():
        raise ValueError("root events cannot carry a parent_event_id")
    parent_times = frame[frame.event_class == "Re_Erasure"].set_index("event_id").seconds_from_start
    linked = frame.loc[is_child, "parent_event_id"].map(parent_times)
    if linked.isna().any() or (linked > frame.loc[is_child, "seconds_from_start"]).any():
        raise ValueError(
            "tombstones/verification receipts require a present erasure parent no later than the child"
        )
    frame["parent_delay_seconds"] = np.nan
    frame.loc[is_child, "parent_delay_seconds"] = frame.loc[is_child, "seconds_from_start"] - linked
    return frame.sort_values(["seconds_from_start", "event_id"], kind="stable").reset_index(drop=True)


def replay_governance_trace(
    path: Path,
    start: datetime,
    end: datetime,
    participants: int,
    institutions: int,
    source_description: str,
    output: Path,
    classification: str = "user-supplied trace",
) -> dict[str, Any]:
    if participants <= 0 or institutions <= 0 or not source_description.strip():
        raise ValueError(
            "trace requires positive participant/institution exposure counts and source description"
        )
    frame = load_governance_trace(path, start, end)
    output.mkdir(parents=True, exist_ok=True)
    duration = (end - start).total_seconds()
    minutes = math.ceil(duration / 60)
    seconds = frame.seconds_from_start.to_numpy()
    minute_index = (seconds // 60).astype(np.int64)
    observed_counts = np.bincount(minute_index, minlength=minutes)
    minute_exposure = np.minimum(60, duration - 60 * np.arange(minutes))
    counts = frame.event_class.value_counts()
    exposure_days = duration / 86400
    rate_rows = []
    for name in COMPONENT_COLUMNS:
        quantity = int(counts.get(name, 0))
        population = institutions if name == "Ri_Integrity" else participants
        delays = frame.loc[frame.event_class == name, "parent_delay_seconds"].dropna()
        rate_rows.append(
            {
                "event_class": name,
                "observed_count": quantity,
                "exposure_days": exposure_days,
                "exposure_entities": population,
                "events_per_day": quantity / exposure_days,
                "events_per_entity_day": quantity / (exposure_days * population),
                "mean_parent_delay_s": float(delays.mean()) if len(delays) else None,
                "p95_parent_delay_s": float(delays.quantile(0.95)) if len(delays) else None,
                "source_classification": classification,
            }
        )
    write_csv(pd.DataFrame(rate_rows), output / "observed_event_rates.csv")
    arrivals = [
        sha256_bytes(b"HTGL-TRACE-LEAF-v2\x00" + i.to_bytes(8, "big")) for i in range(len(frame))
    ]
    trace_rows, queue_rows = [], []
    for capacity in BATCH_SIZES:
        for timeout in BATCH_WINDOWS_SECONDS:
            batcher, batches = CapacityTimeoutBatcher(capacity, timeout), []
            for index, (arrival, leaf) in enumerate(zip(seconds, arrivals)):
                batches.extend(batcher.push(float(arrival), index, leaf))
            batches.extend(batcher.finish(duration))
            emitted = [
                batch
                for batch in batches
                if batch.reason != "observation_end" and batch.closed_at < duration
            ]
            censored = sum(len(b.leaves) for b in batches) - sum(len(b.leaves) for b in emitted)
            waits = [batch.closed_at - arrival for batch in emitted for arrival in batch.arrivals]
            if sum(len(batch.leaves) for batch in batches) != len(frame):
                raise RuntimeError("trace replay did not conserve events")
            anchor_minute = np.bincount([int(b.closed_at // 60) for b in emitted], minlength=minutes)
            trace_rows.append(
                {
                    "batch_size": capacity,
                    "timeout_seconds": timeout,
                    "events": len(frame),
                    "anchors_within_exposure": len(emitted),
                    "right_censored_unanchored_events": censored,
                    "observed_anchor_tps": len(emitted) / duration,
                    "mean_wait_s": float(np.mean(waits)) if waits else None,
                    "p99_wait_s": float(np.quantile(waits, 0.99)) if waits else None,
                    "source_classification": classification,
                }
            )
            # The queue is an analytical service model applied to observed
            # arrivals. Exclude a partial final minute from the queue calculation.
            full_minutes = int(duration // 60)
            for service in ANCHOR_CAPACITY_ENVELOPES_TPS:
                queue_rows.append(
                    {
                        "batch_size": capacity,
                        "timeout_seconds": timeout,
                        "capacity_tps": service,
                        **queue_metrics(anchor_minute[:full_minutes], service),
                        "service_model": "assumed constant service; observed event-time arrivals",
                    }
                )
    write_csv(pd.DataFrame(trace_rows), output / "event_time_batch_replay.csv")
    write_csv(pd.DataFrame(queue_rows), output / "queue_envelope.csv")
    write_csv(
        pd.DataFrame(
            {
                "minute_from_start": np.arange(minutes),
                "exposure_seconds": minute_exposure,
                "observed_events": observed_counts,
                "events_per_second": observed_counts / minute_exposure,
            }
        ),
        output / "observed_minutes.csv",
    )

    # Keep training and later observations separate. Sample contiguous blocks
    # of aggregate counts to retain short-range bursts in a data-driven
    # comparator. This is distinct from the synthetic causal workflow model.
    full = observed_counts[: int(duration // 60)]
    training_size = int(len(full) * 0.8)
    training, holdout = full[:training_size], full[training_size:]
    block_length = min(60, max(1, len(training) // 4))
    generator = experiment_rng("trace_holdout_block_bootstrap", file_checksum(path))
    peak, mean = [], []
    for _ in range(PEAK_BOOTSTRAP_REPLICATIONS):
        starts = generator.integers(
            0, len(training) - block_length + 1, size=math.ceil(len(holdout) / block_length)
        )
        sample = np.concatenate([training[i : i + block_length] for i in starts])[: len(holdout)]
        peak.append(sample.max() / 60)
        mean.append(sample.mean() / 60)
    comparison = {
        "training_minutes": len(training),
        "holdout_minutes": len(holdout),
        "moving_block_minutes": block_length,
        "bootstrap_replications": PEAK_BOOTSTRAP_REPLICATIONS,
        "holdout_mean_events_per_second": float(holdout.mean() / 60),
        "holdout_max_events_per_second": float(holdout.max() / 60),
        "training_bootstrap_mean_p05": float(np.quantile(mean, 0.05)),
        "training_bootstrap_mean_p95": float(np.quantile(mean, 0.95)),
        "training_bootstrap_peak_p05": float(np.quantile(peak, 0.05)),
        "training_bootstrap_peak_p95": float(np.quantile(peak, 0.95)),
        "estimand": "later trace window compared with moving blocks of earlier aggregate counts; same observed exposure",
    }
    write_csv(pd.DataFrame([comparison]), output / "holdout_comparison.csv")
    manifest = {
        "source_description": source_description,
        "classification": classification,
        "sha256": file_checksum(path),
        "exposure_start_utc": start.isoformat(),
        "exposure_end_utc": end.isoformat(),
        "participants": participants,
        "institutions": institutions,
        "rows": len(frame),
        "orphan_or_early_child_rows": 0,
        "retained_input_identifiers": False,
        "calibration": "observed exposure rates and aggregate training-block comparator; default synthetic scenarios remain distinct",
    }
    (output / "trace_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def trace_conformance_experiment() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=4)
    generator = experiment_rng("trace_replay_conformance")
    rows = []
    for minute in range(240):
        count = int(generator.poisson(1.5 + 3.0 * (minute % 60 < 8)))
        for index in range(count):
            timestamp = start + timedelta(seconds=minute * 60 + index)
            rows.append((timestamp.isoformat(), f"root-{minute}-{index}", "Ra_Authorization", ""))
        if minute % 30 == 0:
            timestamp = start + timedelta(minutes=minute)
            root = f"request-{minute}"
            rows.append((timestamp.isoformat(), root, "Re_Erasure", ""))
            rows.append(
                ((timestamp + timedelta(seconds=15)).isoformat(), f"tomb-{minute}", "Rt_Tombstone", root)
            )
            rows.append(
                (
                    (timestamp + timedelta(seconds=45)).isoformat(),
                    f"receipt-{minute}",
                    "Rv_AuditVerification",
                    root,
                )
            )
    with tempfile.TemporaryDirectory(prefix="trace-test-", dir=DATA) as folder:
        fixture = Path(folder) / "fixture.csv"
        data = pd.DataFrame(rows, columns=TRACE_COLUMNS)
        write_csv(data, fixture)
        result = replay_governance_trace(
            fixture,
            start,
            end,
            100,
            5,
            "deterministic synthetic ingestion fixture",
            DATA / "trace_conformance",
            classification="synthetic test fixture",
        )
        bad = data.copy()
        bad.loc[bad.event_class == "Rt_Tombstone", "parent_event_id"] = "missing"
        write_csv(bad, fixture)
        try:
            load_governance_trace(fixture, start, end)
        except ValueError:
            rejected = True
        else:
            rejected = False
    if not rejected or result["rows"] != len(rows):
        raise RuntimeError("trace fixture conformance failed")
    write_csv(
        pd.DataFrame(
            [
                {"case": "trace_ingestion_batch_replay_holdout", "passed": True, "events": len(rows)},
                {"case": "reject_missing_erasure_parent", "passed": rejected, "events": len(rows)},
            ]
        ),
        DATA / "trace_conformance_checks.csv",
    )


# ---------------------------------------------------------------------------
# Compatibility utilization table, paper artifacts, and reproducibility
# ---------------------------------------------------------------------------


def validator_utilization_compatibility(
    baseline_mc: pd.DataFrame,
    scenario_results: pd.DataFrame,
    population_results: pd.DataFrame,
    temporal_summary: pd.DataFrame,
) -> pd.DataFrame:
    baseline_rates = analytical_component_rates(REPRESENTED_POPULATIONS[0], SCENARIOS[0])
    causal_temporal = temporal_summary[temporal_summary["model"] == "causal_workflow"]
    temporal_batching = pd.read_csv(DATA / "temporal_batching_summary.csv")
    reference_anchor = temporal_batching[
        (temporal_batching["model"] == "causal_workflow")
        & (temporal_batching["batch_size"] == WORKFLOW_ASSUMPTIONS.reference_batch_size)
    ]
    workloads = (
        (
            "baseline_analytical",
            "unbatched_consensus_counterfactual",
            baseline_rates["TPS"],
            CAPACITY_ENVELOPES_TPS,
        ),
        (
            "monte_carlo_p99",
            "unbatched_consensus_counterfactual",
            float(baseline_mc["TPS"].quantile(0.99)),
            CAPACITY_ENVELOPES_TPS,
        ),
        (
            "stress_scenario_p99_max_50M",
            "unbatched_consensus_counterfactual",
            float(scenario_results["TPS_p99"].max()),
            CAPACITY_ENVELOPES_TPS,
        ),
        (
            "population_scenario_p99_max_450M",
            "unbatched_consensus_counterfactual",
            float(
                population_results.loc[
                    population_results["population"] == 450_000_000, "TPS_p99"
                ].max()
            ),
            CAPACITY_ENVELOPES_TPS,
        ),
        (
            "causal_temporal_p99_daily_peak_max",
            "unbatched_consensus_counterfactual",
            float(causal_temporal["p99_of_daily_peak_tps"].max()),
            CAPACITY_ENVELOPES_TPS,
        ),
        (
            "causal_temporal_p99_reference_anchor_peak_max",
            "batched_consensus_reference",
            float(reference_anchor["p99_of_daily_max_anchor_tps"].max()),
            ANCHOR_CAPACITY_ENVELOPES_TPS,
        ),
    )
    rows = []
    for workload, service_plane, tps, capacities in workloads:
        for capacity in capacities:
            rows.append(
                {
                    "workload": workload,
                    "service_plane": service_plane,
                    "tps": tps,
                    "capacity_tps": capacity,
                    "capacity_classification": (
                        "published_external_configuration_reference"
                        if capacity == LITERATURE_REFERENCE_CAPACITY_TPS
                        else "analytical_capacity_envelope"
                    ),
                    "utilization_pct": 100.0 * tps / capacity,
                    "headroom_tps": capacity - tps,
                    "within_capacity": tps <= capacity,
                    "interpretation": (
                        "reference Merkle-batched consensus workload"
                        if service_plane == "batched_consensus_reference"
                        else "conservative event-per-transaction counterfactual"
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    write_csv(frame, DATA / "validator_utilization.csv")
    return frame


def write_paper_and_reproducibility_artifacts(
    baseline_mc: pd.DataFrame,
    population_results: pd.DataFrame,
    batching: pd.DataFrame,
    formal_metrics: pd.DataFrame,
    validator_utilization: pd.DataFrame,
) -> None:
    baseline_rates = analytical_component_rates(REPRESENTED_POPULATIONS[0], SCENARIOS[0])
    population_450_baseline = population_results[
        (population_results["population"] == 450_000_000)
        & (population_results["scenario"] == "baseline")
    ].iloc[0]
    population_450_stress = population_results[
        (population_results["population"] == 450_000_000)
        & (population_results["scenario"] == "combined_stress")
    ].iloc[0]
    reference_batch = batching[
        (batching["population"] == 450_000_000)
        & (batching["scenario"] == "baseline")
        & (batching["batch_size"] == 1_000)
        & (batching["max_batch_wait_seconds"] == 60)
    ].iloc[0]

    # Machine-readable parameter and profile catalogues.
    parameters = {
        "SIMUP_CODE_VERSION": SIMUP_CODE_VERSION,
        "run_profile": RUN_PROFILE,
        "random_seed": ACTIVE_SEED,
        "rng_substream_scheme": "SHA-256(global-seed|experiment-labels), first 64 bits",
        "daily_monte_carlo_runs": DAILY_MONTE_CARLO_RUNS,
        "temporal_replications": TEMPORAL_REPLICATIONS,
        "peak_bootstrap_replications": PEAK_BOOTSTRAP_REPLICATIONS,
        "uncertainty_runs_per_population": UNCERTAINTY_RUNS_PER_POPULATION,
        "formal_model_events": FORMAL_MODEL_EVENTS,
        "represented_populations": json.dumps(REPRESENTED_POPULATIONS),
        "institutional_actors": REFERENCE_INSTITUTIONAL_ACTORS,
        "meter_readings_per_consumer_day": METER_READINGS_PER_CONSUMER_DAY,
        "capacity_envelopes_tps": json.dumps(CAPACITY_ENVELOPES_TPS),
        "anchor_capacity_envelopes_tps": json.dumps(ANCHOR_CAPACITY_ENVELOPES_TPS),
        "batch_sizes": json.dumps(BATCH_SIZES),
        "batch_windows_seconds": json.dumps(BATCH_WINDOWS_SECONDS),
        "reference_reduced_evidence_bytes": EVIDENCE_SIZES.reduced_evidence_bytes,
        **asdict(RATE_ASSUMPTIONS),
        **{f"workflow_{key}": value for key, value in asdict(WORKFLOW_ASSUMPTIONS).items()},
    }
    write_csv(
        pd.DataFrame([{"parameter": key, "value": value} for key, value in parameters.items()]),
        DATA / "parameters.csv",
    )
    write_csv(pd.DataFrame([asdict(scenario) for scenario in SCENARIOS]), DATA / "scenario_profiles.csv")
    write_csv(pd.DataFrame(SOURCE_REGISTER), DATA / "assumptions_and_sources.csv")
    write_csv(pd.DataFrame([CLAIM_BOUNDARY]), DATA / "claim_boundary.csv")

    daily_models = pd.read_csv(DATA / "daily_model_comparison.csv")
    causal_450_baseline = daily_models[
        (daily_models["population"] == 450_000_000)
        & (daily_models["scenario"] == "baseline")
        & (daily_models["model"] == "causal_workflow")
    ].iloc[0]
    temporal_batch = pd.read_csv(DATA / "temporal_batching_summary.csv")
    temporal_batch_450 = temporal_batch[
        (temporal_batch["population"] == 450_000_000)
        & (temporal_batch["scenario"] == "baseline")
        & (temporal_batch["model"] == "causal_workflow")
        & (temporal_batch["batch_size"] == WORKFLOW_ASSUMPTIONS.reference_batch_size)
    ].iloc[0]
    workflow_temporal = pd.read_csv(DATA / "workflow_temporal_summary.csv")
    workflow_450 = workflow_temporal[
        (workflow_temporal["population"] == 450_000_000)
        & (workflow_temporal["scenario"] == "baseline")
    ].iloc[0]
    uncertainty_450 = pd.read_csv(DATA / "uncertainty_summary.csv").query(
        "population == 450000000"
    ).iloc[0]
    dependence = pd.read_csv(DATA / "workflow_dependence_metrics.csv")
    erasure_tombstone_correlation = dependence[
        (dependence["population"] == 450_000_000)
        & (dependence["scenario"] == "baseline")
        & (dependence["model"] == "causal_workflow")
        & (dependence["left_component"] == "Re_Erasure")
        & (dependence["right_component"] == "Rt_Tombstone")
    ]["empirical_correlation"].iloc[0]

    experiment_summary = pd.DataFrame(
        [
            {"metric": "baseline_50M_logical_events_day", "value": baseline_rates["Daily_Events"]},
            {"metric": "baseline_50M_logical_event_tps", "value": baseline_rates["TPS"]},
            {"metric": "baseline_50M_mc_mean_tps", "value": baseline_mc["TPS"].mean()},
            {"metric": "baseline_50M_mc_p99_tps", "value": baseline_mc["TPS"].quantile(0.99)},
            {"metric": "baseline_450M_p99_logical_event_tps", "value": population_450_baseline["TPS_p99"]},
            {"metric": "stress_450M_p99_logical_event_tps", "value": population_450_stress["TPS_p99"]},
            {"metric": "baseline_450M_reference_batch_anchor_tps", "value": reference_batch["anchor_tps"]},
            {"metric": "baseline_450M_causal_p99_logical_event_tps", "value": causal_450_baseline["TPS_p99"]},
            {"metric": "baseline_450M_causal_erasure_tombstone_correlation", "value": erasure_tombstone_correlation},
            {"metric": "baseline_450M_temporal_reference_anchor_mean_tps", "value": temporal_batch_450["mean_anchor_tps"]},
            {"metric": "baseline_450M_temporal_reference_anchor_p99_peak_tps", "value": temporal_batch_450["p99_of_daily_max_anchor_tps"]},
            {"metric": "baseline_450M_mean_tombstone_delay_minutes", "value": workflow_450["mean_tombstone_delay_minutes"]},
            {"metric": "baseline_450M_mean_verification_delay_minutes", "value": workflow_450["mean_verification_delay_minutes"]},
            {"metric": "uncertainty_450M_logical_tps_p99", "value": uncertainty_450["logical_event_tps_p99"]},
            {
                "metric": "formal_model_honest_post_erasure_acceptance_rate",
                "value": formal_metrics.iloc[0]["honest_post_erasure_acceptance_rate"],
            },
            {
                "metric": "formal_model_invalid_package_acceptance_rate",
                "value": formal_metrics.iloc[0]["modeled_invalid_package_acceptance_rate"],
            },
        ]
    )
    write_csv(experiment_summary, DATA / "experiment_summary.csv")

    # Paper-ready tables preserve current file names and add stronger ones.
    write_csv(pd.read_csv(DATA / "parameters.csv"), TABLES / "baseline_parameters_table.csv")
    write_csv(
        pd.read_csv(DATA / "scenario_results.csv")[
            ["scenario", "analytical_tps", "TPS_mean", "TPS_p95", "TPS_p99"]
        ],
        TABLES / "scenario_tps_table.csv",
    )
    write_csv(
        population_results[
            ["population_millions", "scenario", "analytical_tps", "TPS_p99"]
        ],
        TABLES / "population_scenario_tps_table.csv",
    )
    write_csv(validator_utilization, TABLES / "validator_utilization_table.csv")
    write_csv(
        batching[
            (batching["max_batch_wait_seconds"] == 60)
            & (batching["batch_size"].isin((1, 100, 1_000, 10_000)))
        ],
        TABLES / "batching_tradeoff_table.csv",
    )
    write_csv(
        pd.read_csv(DATA / "formal_model_verification.csv"),
        TABLES / "formal_verification_table.csv",
    )

    (ARTIFACTS / "paper_result_sentences.md").write_text(
        "\n".join(
            (
                f"- The 50M baseline represents {baseline_rates['Daily_Events']:,.0f} logical HTGL events/day "
                f"({baseline_rates['TPS']:.2f} event TPS).",
                f"- At 450M, baseline p99 logical-event demand is {population_450_baseline['TPS_p99']:.2f} TPS; "
                f"combined-stress p99 demand is {population_450_stress['TPS_p99']:.2f} TPS.",
                f"- With 1,000-item/60-second batches, the 450M baseline deterministic periodic-window anchor rate is "
                f"{reference_batch['anchor_tps']:.6f} anchor TPS. The temporal mean is {temporal_batch_450['mean_anchor_tps']:.6f} "
                f"anchor TPS, with a {temporal_batch_450['p99_of_daily_max_anchor_tps']:.3f} TPS p99 daily peak over {TEMPORAL_REPLICATIONS:,} replications.",
                f"- Explicit workflow generation produced an erasure–tombstone correlation of "
                f"{erasure_tombstone_correlation:.3f}, with zero orphan descendants or causal-order violations.",
                f"- Under the declared uncertainty distributions, 450M logical demand has mean {uncertainty_450['logical_event_tps_mean']:.2f}, "
                f"median {uncertainty_450['logical_event_tps_p50']:.2f}, central 90% empirical interval "
                f"[{uncertainty_450['logical_event_tps_empirical_pi90_lower']:.2f}, {uncertainty_450['logical_event_tps_empirical_pi90_upper']:.2f}], "
                f"central 95% interval [{uncertainty_450['logical_event_tps_empirical_pi95_lower']:.2f}, {uncertainty_450['logical_event_tps_empirical_pi95_upper']:.2f}], "
                f"and p99 {uncertainty_450['logical_event_tps_p99']:.2f} events/s.",
                "- The signed reference verifier exercises the specified structural predicates, independently supplied targets, "
                "authorization and declared scope, and persistent admission. The case table reports exact counts and reasons.",
                "- The observed authenticated false-declaration cases expose the boundary between evidence and application-store truth.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (ARTIFACTS / "claim_boundary_notes.md").write_text(
        CLAIM_BOUNDARY["boundary"] + "\n",
        encoding="utf-8",
    )
    (ARTIFACTS / "formal_feasibility_claims.md").write_text(
        """# Bounded formal claims

1. An honestly generated retained package verifies after logical removal of its synthetic application payload and opening. The verifier consumes signed origin/request/tombstone records, anchored membership proofs, quorum certificates, and an independently supplied audit target.

2. The executed fault matrix checks each tested mutation and records the specific rejection reason. Exact duplicate registration is rejected across restart. Repeat inspection and independent verifier registration remain possible.

3. The causal workload schedules tombstone and verification descendants from erasure parents with nonnegative delay. The independent Poisson process serves as the null comparator.

4. Evidence validity is conditional on the trust registry and declared roster. Physical deletion, custodian truth, undisclosed copies/custodians, legal compliance, and deployment performance remain outside the validated scope.
""",
        encoding="utf-8",
    )

    environment = {
        "run_date_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "script_sha256": file_checksum(Path(__file__)),
        "version": SIMUP_CODE_VERSION,
        "seed": ACTIVE_SEED,
        "run_profile": RUN_PROFILE,
        "rng_substream_scheme": "SHA-256 labelled deterministic substreams",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": importlib.metadata.version("matplotlib"),
        "cryptography_version": importlib.metadata.version("cryptography"),
        "sqlite_version": sqlite3.sqlite_version,
        "platform": platform.platform(),
        "statistical_results_deterministic_given_code_seed_and_dependencies": True,
        "cryptographic_openings_and_key_material": "fresh OS randomness; protocol artifact hashes vary",
        "timings": "measured wall-clock durations; machine dependent",
        "submitted_paper_sha256": file_checksum(SUBMITTED_PAPER) if SUBMITTED_PAPER else "unspecified",
        "submitted_paper_name": SUBMITTED_PAPER.name if SUBMITTED_PAPER else "unspecified",
        "output_folder": str(DATA.relative_to(BASE)),
    }
    write_csv(pd.DataFrame([environment]), DATA / "reproducibility_manifest.csv")


def write_publication_tables() -> pd.DataFrame:
    """Create concise CSV and LaTeX tables from authoritative result files."""

    def latex_escape(value: Any) -> str:
        if pd.isna(value):
            return "--"
        if isinstance(value, (bool, np.bool_)):
            return "true" if bool(value) else "false"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4g}"
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        text = str(value)
        return "".join(replacements.get(character, character) for character in text)

    def render_latex_table(frame: pd.DataFrame, caption: str, label: str) -> str:
        alignment = "".join(
            "r" if pd.api.types.is_numeric_dtype(frame[column]) else "l"
            for column in frame.columns
        )
        header = " & ".join(latex_escape(column) for column in frame.columns) + r" \\"
        body = [
            " & ".join(latex_escape(value) for value in row) + r" \\"
            for row in frame.itertuples(index=False, name=None)
        ]
        return "\n".join(
            (
                r"\begin{table}[t]",
                r"\centering",
                r"\small",
                f"\\caption{{{latex_escape(caption)}}}",
                f"\\label{{{label}}}",
                f"\\begin{{tabular}}{{{alignment}}}",
                r"\toprule",
                header,
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            )
        )

    table_specs: list[tuple[str, str, str, pd.DataFrame]] = []
    scale = pd.read_csv(DATA / "daily_model_comparison.csv")
    scale = scale[scale["model"] == "causal_workflow"][
        ["population_millions", "scenario", "analytical_tps", "TPS_mean", "TPS_p95", "TPS_p99"]
    ]
    table_specs.append(
        (
            "tab_01_scale_envelope",
            "Population and scenario workload envelope under the causal model.",
            "tab:scale-envelope",
            scale,
        )
    )

    moments = pd.read_csv(DATA / "model_moment_validation.csv")
    moments = moments[
        (moments["population"] == 450_000_000)
        & (moments["scenario"] == "baseline")
        & (moments["quantity"].isin(("Re_Erasure", "Rt_Tombstone", "Rv_AuditVerification", "Daily_Events")))
    ][
        [
            "model",
            "quantity",
            "theoretical_mean",
            "empirical_mean",
            "mean_relative_error_pct",
            "theoretical_fano_factor",
            "empirical_fano_factor",
        ]
    ]
    table_specs.append(
        (
            "tab_02_model_moment_validation",
            "Closed-form and simulated moments at 450 million represented participants.",
            "tab:model-validation",
            moments,
        )
    )

    dependence = pd.read_csv(DATA / "workflow_dependence_metrics.csv")
    dependence = dependence[
        (dependence["population"] == 450_000_000)
        & (dependence["scenario"] == "baseline")
    ][
        [
            "model",
            "left_component",
            "right_component",
            "theoretical_correlation",
            "empirical_correlation",
        ]
    ]
    table_specs.append(
        (
            "tab_03_causal_dependence",
            "Dependence introduced by explicit erasure descendants versus the null model.",
            "tab:causal-dependence",
            dependence,
        )
    )

    batching = pd.read_csv(DATA / "temporal_batching_summary.csv")
    batching = batching[
        (batching["population"] == 450_000_000)
        & (batching["scenario"] == "baseline")
        & (batching["model"] == "causal_workflow")
    ][
        [
            "batch_size",
            "mean_anchor_tps",
            "p99_of_daily_max_anchor_tps",
            "mean_batch_fill_ratio",
            "mean_consensus_reduction_factor",
        ]
    ]
    table_specs.append(
        (
            "tab_04_batching_tradeoff",
            "One-minute temporal batching trade-off at 450 million participants.",
            "tab:batching",
            batching,
        )
    )

    queue = pd.read_csv(DATA / "capacity_queue_summary.csv")
    queue = queue[
        (queue["population"] == 450_000_000)
        & (queue["model"] == "causal_workflow")
        & (queue["scenario"].isin(("baseline", "combined_stress")))
        & (queue["capacity_tps"].isin((10, 3_500, 8_000)))
    ][
        [
            "scenario",
            "service_plane",
            "capacity_tps",
            "mean_utilization_pct",
            "probability_any_backlog",
            "probability_ending_backlog",
            "p99_max_queue_delay_seconds",
        ]
    ]
    table_specs.append(
        (
            "tab_05_queue_envelope",
            "Warm-start discrete-time queue envelope for selected service rates.",
            "tab:queue-envelope",
            queue,
        )
    )

    uncertainty = pd.read_csv(DATA / "uncertainty_summary.csv")[
        [
            "population_millions",
            "logical_event_tps_p50",
            "logical_event_tps_p95",
            "logical_event_tps_p99",
            "reference_anchor_tps_p99",
            "retained_storage_gb_day_296B_p99",
        ]
    ]
    table_specs.append(
        (
            "tab_06_uncertainty_envelope",
            "Latin-hypercube uncertainty envelope over declared research ranges.",
            "tab:uncertainty",
            uncertainty,
        )
    )

    formal = pd.read_csv(DATA / "formal_model_verification.csv")[
        ["case", "cases_tested", "expected_accept", "accepted", "rejected", "model_correct_classification_rate"]
    ]
    table_specs.append(
        (
            "tab_07_abstract_verifier",
            "Signed reference verifier with exact injected-case outcomes.",
            "tab:abstract-verifier",
            formal,
        )
    )
    comparison = pd.read_csv(DATA / "paper_batching_reconciliation.csv")
    comparison = comparison[
        (comparison.population == 450_000_000)
        & (comparison.model == "causal_workflow")
        & comparison.scenario.isin(("baseline", "combined_stress"))
    ]
    table_specs.append(
        (
            "tab_08_estimator_reconciliation",
            "Deterministic and temporal anchor estimates for the 60-second window.",
            "tab:estimators",
            comparison[
                [
                    "scenario",
                    "batch_size",
                    "deterministic_anchor_tps",
                    "mean_anchor_tps",
                    "replications",
                    "p99_of_daily_max_anchor_tps",
                ]
            ],
        )
    )
    policy = pd.read_csv(DATA / "storage_policy_comparison.csv")
    policy = policy[
        (policy.population == 450_000_000)
        & (policy.days == 365)
        & (policy.replication_factor == 3)
        & (policy.size_profile == "canonical_protocol_v2")
        & policy.scenario.isin(("baseline", "combined_stress"))
    ]
    table_specs.append(
        (
            "tab_09_compact_storage",
            "Annual retained evidence at three copies under the canonical protocol profile, in decimal GB.",
            "tab:compact-storage",
            policy[["scenario", "witness_policy", "membership_witness_gb", "total_gb"]],
        )
    )

    index_rows: list[dict[str, Any]] = []
    for stem, caption, label, frame in table_specs:
        csv_path = TABLES / f"{stem}.csv"
        tex_path = TABLES / f"{stem}.tex"
        write_csv(frame, csv_path)
        latex = render_latex_table(frame, caption, label)
        tex_path.write_text(latex, encoding="utf-8")
        index_rows.append(
            {
                "table_id": stem,
                "caption": caption,
                "source_rows": len(frame),
                "csv_file": str(csv_path.relative_to(BASE)),
                "latex_file": str(tex_path.relative_to(BASE)),
            }
        )
    index_frame = pd.DataFrame(index_rows)
    write_csv(index_frame, ARTIFACTS / "publication_table_manifest.csv")
    return index_frame


def export_result_reconciliation() -> None:
    periodic = pd.read_csv(DATA / "batching_analysis.csv")
    periodic = periodic[periodic.max_batch_wait_seconds == 60][
        ["population", "scenario", "batch_size", "anchor_tps"]
    ]
    periodic = periodic.rename(columns={"anchor_tps": "deterministic_anchor_tps"})
    temporal = pd.read_csv(DATA / "temporal_batching_summary.csv")
    result = temporal.merge(
        periodic, on=["population", "scenario", "batch_size"], how="left", validate="many_to_one"
    )
    if result.deterministic_anchor_tps.isna().any():
        raise RuntimeError("a temporal batch configuration lacks its deterministic comparator")
    result["temporal_minus_deterministic_tps"] = result.mean_anchor_tps - result.deterministic_anchor_tps
    result["deterministic_estimator"] = "Equation (13), expected occupancy per fixed closure window"
    result["temporal_estimator"] = (
        "mean over stochastic daily anchor rates, Equation (14) applied to minute counts"
    )
    write_csv(result, DATA / "paper_batching_reconciliation.csv")
    summary = pd.read_csv(DATA / "uncertainty_summary.csv")
    columns = [
        "population",
        "samples",
        "logical_event_tps_mean",
        "logical_event_tps_p50",
        "logical_event_tps_empirical_pi90_lower",
        "logical_event_tps_empirical_pi90_upper",
        "logical_event_tps_empirical_pi95_lower",
        "logical_event_tps_empirical_pi95_upper",
        "logical_event_tps_p99",
    ]
    write_csv(summary[columns], DATA / "paper_uncertainty_statistics.csv")


def generate_publication_figures() -> pd.DataFrame:
    """Generate the final paper figures in vector PDF and 600-dpi PNG."""

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/htgl_matplotlib_cache")
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:  # pragma: no cover - dependency error is explicit
        raise RuntimeError("Matplotlib is required for publication-ready figures.") from exc

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d9dde3",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.85,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    scenario_labels = {scenario.name: scenario.label for scenario in SCENARIOS}
    figure_rows: list[dict[str, Any]] = []

    def save_figure(
        figure: Any,
        figure_id: str,
        title: str,
        caption: str,
        source_files: Sequence[str],
    ) -> None:
        png_path = FIGURES / f"{figure_id}.png"
        pdf_path = FIGURES / f"{figure_id}.pdf"
        figure.savefig(png_path, bbox_inches="tight", facecolor="white")
        figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        figure_rows.append(
            {
                "figure_id": figure_id,
                "title": title,
                "caption": caption,
                "source_files": ";".join(source_files),
                "png_file": str(png_path.relative_to(BASE)),
                "pdf_file": str(pdf_path.relative_to(BASE)),
            }
        )

    scale = pd.read_csv(DATA / "daily_model_comparison.csv")
    scale = scale[scale["model"] == "causal_workflow"]
    fig, ax = plt.subplots(figsize=(7.15, 3.45))
    for color, scenario in zip(colors, SCENARIOS):
        group = scale[scale["scenario"] == scenario.name].sort_values("population_millions")
        ax.plot(
            group["population_millions"],
            group["TPS_p99"],
            marker="o",
            linewidth=1.8,
            color=color,
            label=scenario.label,
        )
    ax.axhline(
        LITERATURE_REFERENCE_CAPACITY_TPS,
        color="#555555",
        linestyle="--",
        linewidth=1.1,
        label="Fabric reference (3,500 TPS)",
    )
    ax.axhline(8_000, color="#888888", linestyle=":", linewidth=1.1, label="8,000 TPS analytical envelope")
    ax.set(xlabel="Represented population (millions)", ylabel="p99 logical-event demand (TPS)")
    ax.set_xticks([50, 150, 450])
    ax.legend(ncol=3, frameon=False, loc="upper left")
    save_figure(
        fig,
        "fig_01_population_scale",
        "Population-scale workload envelope",
        f"Causal-model p99 daily demand. The 3,500 TPS line marks the configuration-specific Fabric literature comparison from Androulaki et al., EuroSys 2018 ({FABRIC_REFERENCE_DOI}). The 8,000 TPS line is an assumed capacity envelope.",
        ("htgl_data/daily_model_comparison.csv",),
    )

    dependence = pd.read_csv(DATA / "workflow_dependence_metrics.csv")
    dependence = dependence[
        (dependence["population"] == 450_000_000)
        & (dependence["scenario"] == "baseline")
    ].copy()
    dependence["pair"] = dependence["left_component"].str.replace("Re_Erasure", "Erasure", regex=False)
    dependence["pair"] += "–" + dependence["right_component"].replace(
        {"Rt_Tombstone": "tombstone", "Rv_AuditVerification": "verification"}
    )
    pair_order = list(dict.fromkeys(dependence["pair"]))
    x = np.arange(len(pair_order))
    fig, ax = plt.subplots(figsize=(7.15, 3.25))
    width = 0.34
    for offset, (model, label, color) in enumerate(
        (
            ("independent_poisson_null", "Independent-Poisson null", colors[0]),
            ("causal_workflow", "Causal workflow", colors[1]),
        )
    ):
        group = dependence[dependence["model"] == model].set_index("pair").reindex(pair_order)
        ax.bar(x + (offset - 0.5) * width, group["empirical_correlation"], width, label=label, color=color)
    ax.set_xticks(x, pair_order)
    ax.set_ylim(-0.1, 1.08)
    ax.set_ylabel("Empirical Pearson correlation")
    ax.legend(frameon=False, ncol=2)
    save_figure(
        fig,
        "fig_02_causal_dependence",
        "Causal dependence versus the Poisson null",
        "Explicit descendant generation restores the required erasure–tombstone–verification dependence that is absent from the independent-Poisson comparator.",
        ("htgl_data/workflow_dependence_metrics.csv",),
    )

    temporal = pd.read_csv(DATA / "temporal_scenario_summary.csv")
    temporal = temporal[
        (temporal["population"] == 450_000_000)
        & (temporal["model"] == "causal_workflow")
    ].set_index("scenario").reindex([scenario.name for scenario in SCENARIOS])
    x = np.arange(len(temporal))
    fig, ax = plt.subplots(figsize=(7.15, 3.35))
    ax.bar(x - 0.18, temporal["mean_daily_tps"], 0.36, color=colors[0], label="Mean daily TPS")
    ax.bar(x + 0.18, temporal["p99_of_daily_peak_tps"], 0.36, color=colors[1], label="p99 daily peak TPS")
    ax.axhline(
        LITERATURE_REFERENCE_CAPACITY_TPS,
        color="#555555",
        linestyle="--",
        linewidth=1.1,
        label="Fabric reference (3,500 TPS)",
    )
    ax.set_xticks(x, [scenario.label for scenario in SCENARIOS], rotation=12, ha="right")
    ax.set_ylabel("Logical-event TPS")
    ax.legend(frameon=False, ncol=3)
    save_figure(
        fig,
        "fig_03_temporal_peaks",
        "Correlated temporal demand at 450 million",
        f"Temporal peaks under the declared MMPP and diurnal stress parameters. The 3,500 TPS reference is the configuration-specific Fabric comparison from Androulaki et al. ({FABRIC_REFERENCE_DOI}).",
        ("htgl_data/temporal_scenario_summary.csv",),
    )

    batching = pd.read_csv(DATA / "temporal_batching_summary.csv")
    batching = batching[
        (batching["population"] == 450_000_000)
        & (batching["scenario"] == "baseline")
        & (batching["model"] == "causal_workflow")
    ].sort_values("batch_size")
    queue = pd.read_csv(DATA / "capacity_queue_summary.csv")
    queue = queue[
        (queue["population"] == 450_000_000)
        & (queue["scenario"] == "combined_stress")
        & (queue["model"] == "causal_workflow")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15))
    axes[0].loglog(batching["batch_size"], batching["mean_anchor_tps"], marker="o", color=colors[0], label="Mean")
    axes[0].loglog(batching["batch_size"], batching["p99_of_daily_max_anchor_tps"], marker="s", color=colors[1], label="p99 daily peak")
    axes[0].set(xlabel="Merkle batch size", ylabel="Anchor transactions per second")
    axes[0].legend(frameon=False)
    for color, (plane, label) in zip(
        (colors[2], colors[3]),
        (
            ("unbatched_consensus_counterfactual", "Event-per-transaction"),
            ("batched_consensus_reference", "1,000-item batches"),
        ),
    ):
        group = queue[queue["service_plane"] == plane].sort_values("capacity_tps")
        axes[1].semilogx(group["capacity_tps"], group["probability_ending_backlog"], marker="o", color=color, label=label)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set(xlabel="Service capacity (TPS)", ylabel="Probability of end-of-day backlog", ylim=(-0.03, 1.03))
    axes[1].legend(frameon=False)
    fig.tight_layout(w_pad=2.0)
    save_figure(
        fig,
        "fig_04_batching_and_queue",
        "Batching and queue feasibility",
        "Reference Merkle batching reduces consensus arrivals by orders of magnitude. The queue panel separates that design from the conservative event-per-transaction counterfactual.",
        ("htgl_data/temporal_batching_summary.csv", "htgl_data/capacity_queue_summary.csv"),
    )

    uncertainty = pd.read_csv(DATA / "uncertainty_summary.csv").sort_values("population_millions")
    prcc = pd.read_csv(DATA / "prcc_sensitivity.csv")
    prcc = prcc[
        (prcc["population"] == 450_000_000) & (prcc["outcome"] == "logical_event_tps")
    ].nlargest(8, "absolute_prcc").sort_values("prcc")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.25))
    axes[0].plot(uncertainty["population_millions"], uncertainty["logical_event_tps_p50"], marker="o", color=colors[0], label="Median")
    axes[0].fill_between(
        uncertainty["population_millions"].to_numpy(),
        uncertainty["logical_event_tps_p95"].to_numpy(),
        uncertainty["logical_event_tps_p99"].to_numpy(),
        color=colors[0],
        alpha=0.22,
        label="p95–p99 envelope",
    )
    axes[0].axhline(3_500, color="#555555", linestyle="--", linewidth=1.0)
    axes[0].set(xlabel="Represented population (millions)", ylabel="Logical-event TPS")
    axes[0].legend(frameon=False)
    prcc_labels = prcc["input_parameter"].str.replace("_", " ")
    axes[1].barh(prcc_labels, prcc["prcc"], color=np.where(prcc["prcc"] >= 0, colors[2], colors[1]))
    axes[1].axvline(0, color="#555555", linewidth=0.8)
    axes[1].set(xlabel="Partial rank correlation", ylabel="")
    fig.tight_layout(w_pad=2.2)
    save_figure(
        fig,
        "fig_05_uncertainty_and_prcc",
        "Uncertainty envelope and parameter influence",
        "Latin-hypercube samples propagate declared assumption ranges; PRCC identifies which inputs dominate 450-million logical demand.",
        ("htgl_data/uncertainty_summary.csv", "htgl_data/prcc_sensitivity.csv"),
    )

    storage = pd.read_csv(DATA / "storage_detailed.csv")
    storage = storage[
        (storage["days"] == 365) & (storage["record_size_bytes"] == 296)
    ]
    fig, ax = plt.subplots(figsize=(7.15, 3.35))
    for color, scenario in zip(colors, SCENARIOS):
        group = storage[storage["scenario"] == scenario.name].sort_values("population_millions")
        ax.plot(
            group["population_millions"],
            group["total_retained_storage_gb"] / 1_000.0,
            marker="o",
            color=color,
            linewidth=1.8,
            label=scenario.label,
        )
    ax.set(xlabel="Represented population (millions)", ylabel="Annual retained evidence (TB, one copy)")
    ax.legend(frameon=False, ncol=2)
    save_figure(
        fig,
        "fig_06_storage_envelope",
        "Retained-evidence storage envelope",
        "Annual single-copy decimal-terabyte storage under the legacy 296-byte envelope, with requests, tombstones, receipts, one digest path per event, and anchor receipts.",
        ("htgl_data/storage_detailed.csv",),
    )

    formal = pd.read_csv(DATA / "formal_model_verification.csv").copy()
    formal["reported_rate"] = np.where(
        formal["expected_accept"],
        formal["accepted"] / formal["cases_tested"],
        formal["rejected"] / formal["cases_tested"],
    )
    formal["label"] = formal["case"].str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(8.0, max(4.25, 0.23 * len(formal))))
    bar_colors = np.where(formal["expected_accept"], colors[2], colors[1])
    ax.barh(formal["label"], formal["reported_rate"], color=bar_colors)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set(xlabel="Correct model outcome", xlim=(0.0, 1.02), ylabel="")
    ax.invert_yaxis()
    save_figure(
        fig,
        "fig_07_abstract_verifier",
        "Signed post-erasure reference verification",
        "Exact outcomes for the executed signed-reference cases, including deliberately accepted boundary cases. The data table includes counts and reasons for every case.",
        ("htgl_data/formal_model_verification.csv",),
    )

    retention = pd.read_csv(DATA / "storage_policy_comparison.csv")
    retention = retention[
        (retention.population == 450_000_000)
        & (retention.days == 365)
        & (retention.replication_factor == 3)
        & (retention.size_profile == "canonical_protocol_v2")
    ]
    policies = WitnessRepository.POLICIES
    x = np.arange(len(policies))
    fig, ax = plt.subplots(figsize=(7.15, 3.35))
    for offset, scenario, color in (
        (-0.18, "baseline", colors[0]),
        (0.18, "combined_stress", colors[1]),
    ):
        selected = (
            retention[retention.scenario == scenario].set_index("witness_policy").reindex(policies)
        )
        ax.bar(x + offset, selected.total_gb / 1000, 0.36, color=color, label=scenario_labels[scenario])
    ax.set_xticks(x, ["Encoded paths", "Shared nodes", "Leaves on demand", "Compressed paths"])
    ax.set_ylabel("Annual retained evidence (TB, three copies)")
    ax.legend(frameon=False)
    save_figure(
        fig,
        "fig_08_compact_retention",
        "Compact witness retention policies",
        "Protocol-v2 storage at 450 million represented participants under uniform 365-day retention. Shared and leaf-only representations derive proof bytes from the actual codecs. Compressed storage extrapolates measured serialized fixture costs.",
        ("htgl_data/storage_policy_comparison.csv", "htgl_data/retention_policy_measurements.csv"),
    )

    manifest = pd.DataFrame(figure_rows)
    write_csv(manifest, ARTIFACTS / "publication_figure_manifest.csv")
    caption_lines = ["# Publication figure captions", ""]
    for row in figure_rows:
        caption_lines.extend(
            (
                f"## {row['figure_id']}: {row['title']}",
                "",
                row["caption"],
                "",
                f"Source data: `{row['source_files']}`.",
                "",
            )
        )
    (ARTIFACTS / "figure_captions.md").write_text(
        "\n".join(caption_lines), encoding="utf-8"
    )
    return manifest


def run_validation_audit() -> pd.DataFrame:
    """Execute blocking numerical, structural, and artifact checks."""

    rows: list[dict[str, Any]] = []

    def record(check: str, passed: bool, observed: Any, criterion: str, blocking: bool = True) -> None:
        rows.append(
            {
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "observed": observed,
                "criterion": criterion,
                "blocking": blocking,
            }
        )

    moments = pd.read_csv(DATA / "model_moment_validation.csv")
    for model in ("independent_poisson_null", "causal_workflow"):
        group = moments[moments["model"] == model]
        maximum_mean_error = float(group["mean_relative_error_pct"].abs().max())
        maximum_variance_error = float(group["variance_relative_error_pct"].abs().max())
        record(
            f"{model}_mean_matches_closed_form",
            maximum_mean_error < 0.25,
            maximum_mean_error,
            "maximum absolute mean relative error < 0.25%",
        )
        record(
            f"{model}_variance_matches_closed_form",
            maximum_variance_error < 10.0,
            maximum_variance_error,
            "maximum absolute variance relative error < 10%",
        )

    dependence = pd.read_csv(DATA / "workflow_dependence_metrics.csv")
    null_dependence = dependence[dependence["model"] == "independent_poisson_null"]
    causal_dependence = dependence[dependence["model"] == "causal_workflow"]
    null_max = float(null_dependence["empirical_correlation"].abs().max())
    causal_error = float(
        (causal_dependence["empirical_correlation"] - causal_dependence["theoretical_correlation"])
        .abs()
        .max()
    )
    record("poisson_null_has_negligible_cross_component_correlation", null_max < 0.05, null_max, "maximum |r| < 0.05")
    record("causal_correlations_match_closed_form", causal_error < 0.05, causal_error, "maximum |empirical - theoretical r| < 0.05")

    daily_invariants = pd.read_csv(DATA / "workflow_invariant_checks.csv")
    daily_causal = daily_invariants[daily_invariants["model"] == "causal_workflow"]
    daily_violations = int(
        daily_causal[
            [
                "tombstone_parent_bound_violations",
                "verification_parent_bound_violations",
                "orphan_children_generated",
                "causal_order_violations",
            ]
        ].sum().sum()
    )
    record("daily_workflow_invariants", daily_violations == 0, daily_violations, "zero violations")

    temporal_workflow = pd.read_csv(DATA / "workflow_temporal_summary.csv")
    temporal_violations = int(
        temporal_workflow[["total_orphan_children", "total_causal_order_violations"]].sum().sum()
    )
    record("temporal_workflow_invariants", temporal_violations == 0, temporal_violations, "zero violations")

    formal = pd.read_csv(DATA / "formal_model_verification.csv")
    honest = formal[formal["expected_accept"] == True]
    invalid = formal[formal["expected_accept"] == False]
    record("formal_honest_completeness", bool((honest["accepted"] == honest["cases_tested"]).all()), int(honest["accepted"].sum()), "all honest packages accepted")
    record("formal_fault_detection", int(invalid["accepted"].sum()) == 0, int(invalid["accepted"].sum()), "zero injected invalid packages accepted")
    record(
        "exact_protocol_outcomes_and_reasons",
        bool((formal.correct_reason_rate == 1).all()),
        int(formal.cases_tested.sum()),
        "every case returns its expected reason",
    )
    conformance = pd.read_csv(DATA / "protocol_conformance_checks.csv")
    record(
        "protocol_conformance",
        bool(conformance.passed.all()),
        len(conformance),
        "all independent vectors, partial trees, quorum and concurrent admission checks pass",
    )
    trace_checks = pd.read_csv(DATA / "trace_conformance_checks.csv")
    record(
        "trace_ingestion_conformance",
        bool(trace_checks.passed.all()),
        len(trace_checks),
        "valid trace replays; missing parent is rejected",
    )
    required_cases = {
        "exact_replay_registration",
        "replay_after_verifier_restart",
        "repeat_inspection_after_registration",
        "coherent_whole_witness_substitution",
        "authenticated_tombstone_before_origin",
        "invalid_anchor_signature",
        "authenticated_false_erasure_declaration",
        "conflicting_declaration_identity",
    }
    record(
        "reviewer_fault_coverage",
        required_cases <= set(formal.case),
        len(required_cases & set(formal.case)),
        "all requested distinct fault classes executed",
    )
    truth = pd.read_csv(DATA / "false_declaration_ground_truth.csv")
    false_count = int(truth.application_payload_present.sum())
    false_row = formal[formal.case == "authenticated_false_erasure_declaration"]
    record(
        "false_declaration_truth_boundary",
        len(false_row) == 1 and false_count > 0 and int(false_row.accepted.iloc[0]) == false_count,
        false_count,
        "retained application payload is confirmed by the separate store; valid signed witness remains acceptable",
    )

    analytical = pd.read_csv(DATA / "batching_analysis.csv")
    desired_grid = {
        (p, s.name, b, w)
        for p in REPRESENTED_POPULATIONS
        for s in SCENARIOS
        for b in BATCH_SIZES
        for w in BATCH_WINDOWS_SECONDS
    }
    observed_grid = set(
        analytical[["population", "scenario", "batch_size", "max_batch_wait_seconds"]].itertuples(
            index=False, name=None
        )
    )
    record(
        "complete_analytical_batch_grid",
        observed_grid == desired_grid and len(analytical) == len(desired_grid),
        len(observed_grid),
        "all requested population/scenario/capacity/window combinations exactly once",
    )
    temporal_counts = pd.read_csv(DATA / "temporal_scenario_summary.csv")
    temporal_count_ok = (
        len(temporal_counts) == len(REPRESENTED_POPULATIONS) * len(SCENARIOS) * 2
        and temporal_counts.replication_count.eq(TEMPORAL_REPLICATIONS).all()
        and temporal_counts.bootstrap_replications.eq(PEAK_BOOTSTRAP_REPLICATIONS).all()
        and (
            temporal_counts.p99_daily_peak_ci95_lower <= temporal_counts.p99_daily_peak_ci95_upper
        ).all()
    )
    record(
        "temporal_configuration_and_intervals",
        bool(temporal_count_ok),
        temporal_counts.replication_count.unique().tolist(),
        "all groups match selected profile and ordered peak confidence intervals",
    )
    storage = pd.read_csv(DATA / "storage_detailed.csv")
    summands = [
        "reduced_evidence_storage_gb",
        "batch_anchor_storage_gb",
        "tombstone_storage_gb",
        "audit_receipt_storage_gb",
        "membership_storage_gb",
    ]
    record(
        "storage_components_conserve_bytes",
        bool(np.allclose(storage[summands].sum(axis=1), storage.total_retained_storage_gb, rtol=1e-12)),
        len(storage),
        "all path-inclusive storage totals equal sum of components",
    )
    ref_rates = analytical_component_rates(450_000_000, SCENARIOS[0])
    d, dt, dv = ref_rates["Daily_Events"], ref_rates["Rt_Tombstone"], ref_rates["Rv_AuditVerification"]
    reference = (
        (d - dt - dv) * 296 + dt * 204 + dv * 170 + d * 320 + 1440 * math.ceil(d / 1440 / 1000) * 208
    ) / 1e9
    row = storage[
        (storage.population == 450_000_000)
        & (storage.scenario == "baseline")
        & (storage.days == 1)
        & (storage.record_size_bytes == 296)
    ]
    record(
        "independent_equation_16_reference",
        len(row) == 1
        and math.isclose(float(row.total_retained_storage_gb.iloc[0]), reference, rel_tol=1e-12),
        reference,
        "independently calculated paper-reference storage includes 320 bytes per logical event",
    )
    measured = pd.read_csv(DATA / "retention_policy_measurements.csv")
    exact = measured[measured.policy != "compressed_path_bundle"]
    matches = [
        witness_cost_lookup(int(item.batch_size), item.policy)[int(item.batch_size)]
        == item.retained_bytes_per_batch
        for item in exact.itertuples()
    ]
    record(
        "retention_model_matches_codec",
        all(matches) and measured.proofs_verified.gt(0).all(),
        len(matches),
        "exact representations match measured serialized lengths and all sampled proofs verify",
    )
    traceability = pd.read_csv(DATA / "reviewer_comment_traceability.csv")
    record(
        "reviewer_comments_mapped",
        len(traceability) == 17 and traceability.review_comment.is_unique,
        len(traceability),
        "one evidence row per mapped reviewer concern",
    )

    mmpp = pd.read_csv(DATA / "mmpp_state_model.csv")
    stationary_sum = float(mmpp["stationary_probability"].sum())
    record("mmpp_stationary_distribution", abs(stationary_sum - 1.0) < 1e-10, stationary_sum, "stationary probabilities sum to one")

    batching = pd.read_csv(DATA / "temporal_batching_metrics.csv")
    invalid_batch_rows = int((batching["daily_anchor_transactions"] > batching["daily_logical_events"]).sum())
    record("batching_never_increases_transaction_count", invalid_batch_rows == 0, invalid_batch_rows, "zero rows with anchors > logical events")

    queue = pd.read_csv(DATA / "capacity_queue_summary.csv")
    batched_reference = queue[
        (queue["model"] == "causal_workflow")
        & (queue["service_plane"] == "batched_consensus_reference")
        & (queue["capacity_tps"] == 3_500)
    ]
    batched_backlog_probability = float(batched_reference["probability_ending_backlog"].max())
    record("reference_batching_drains_at_external_reference", batched_backlog_probability == 0.0, batched_backlog_probability, "zero observed end-of-day backlog probability")

    uncertainty = pd.read_csv(DATA / "uncertainty_summary.csv")
    ordered_quantiles = bool(
        (
            (uncertainty["logical_event_tps_p50"] <= uncertainty["logical_event_tps_p95"])
            & (uncertainty["logical_event_tps_p95"] <= uncertainty["logical_event_tps_p99"])
        ).all()
    )
    record("uncertainty_quantiles_are_ordered", ordered_quantiles, ordered_quantiles, "p50 <= p95 <= p99 for every population")

    first = experiment_rng("reproducibility_self_test").integers(0, 2**31, size=128)
    second = experiment_rng("reproducibility_self_test").integers(0, 2**31, size=128)
    reproducible = bool(np.array_equal(first, second))
    record("labelled_rng_substreams_are_reproducible", reproducible, reproducible, "identical labels reproduce identical samples")

    key_files = (
        "daily_model_comparison.csv",
        "model_moment_validation.csv",
        "temporal_scenario_summary.csv",
        "capacity_queue_summary.csv",
        "uncertainty_summary.csv",
        "formal_model_verification.csv",
    )
    nonfinite_count = 0
    for file_name in key_files:
        frame = pd.read_csv(DATA / file_name)
        numeric = frame.select_dtypes(include=[np.number])
        nonfinite_count += int((~np.isfinite(numeric.to_numpy(dtype=float))).sum())
    record("key_outputs_are_finite", nonfinite_count == 0, nonfinite_count, "zero NaN or infinite values in key result tables")

    figure_manifest = pd.read_csv(ARTIFACTS / "publication_figure_manifest.csv")
    missing_figure_files = 0
    for column in ("png_file", "pdf_file"):
        missing_figure_files += sum(not (BASE / path).is_file() for path in figure_manifest[column])
    record(
        "publication_figures_complete",
        len(figure_manifest) == 8 and missing_figure_files == 0,
        f"{len(figure_manifest)} figures; {missing_figure_files} missing files",
        "all eight figures have PNG and PDF files",
    )

    report = pd.DataFrame(rows)
    write_csv(report, DATA / "validation_report.csv")
    failures = report[(report["status"] == "FAIL") & (report["blocking"] == True)]
    if not failures.empty:
        failed_names = ", ".join(failures["check"].tolist())
        raise RuntimeError(f"Publication validation failed: {failed_names}")
    return report


def finalize_output_manifest() -> pd.DataFrame:
    """Hash every finalized output and source file except the manifest itself."""

    source_candidates = [
        p
        for p in (
            Path(__file__).resolve(),
            SOURCE_DIR / "requirements.txt",
            SOURCE_DIR / "RUN_SIMULATION.md",
        )
        if p.is_file() and p.is_relative_to(BASE)
    ]
    authoritative_files = sorted(
        [path for path in DATA.rglob("*") if path.is_file() and path.name != "output_file_checksums.csv"]
        + [path for path in ARTIFACTS.rglob("*") if path.is_file()]
        + source_candidates,
        key=lambda path: str(path),
    )
    rows = []
    for path in authoritative_files:
        suffix = path.suffix.lower()
        rows.append(
            {
                "file": str(path.relative_to(BASE)),
                "artifact_class": (
                    "source"
                    if path in source_candidates
                    else "figure"
                    if suffix in (".png", ".pdf", ".svg")
                    else "paper_table"
                    if TABLES in path.parents
                    else "data_or_metadata"
                ),
                "bytes": path.stat().st_size,
                "sha256": file_checksum(path),
            }
        )
    manifest = pd.DataFrame(rows).drop_duplicates(subset=["file"], keep="last")
    write_csv(manifest, DATA / "output_file_checksums.csv")
    return manifest


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the HTGL design experiments and signed reference evidence pipeline."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Global deterministic seed (default: {SEED}).",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate publication tables/figures from existing authoritative CSV outputs.",
    )
    parser.add_argument("--version", action="version", version=SIMUP_CODE_VERSION)
    parser.add_argument(
        "--profile",
        choices=("publication", "verification"),
        default="publication",
        help="Publication uses 1,000 temporal replications. Verification exercises all stages with smaller temporal and witness samples.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SOURCE_DIR, help="Root for htgl_data and paper_artifacts."
    )
    parser.add_argument(
        "--submitted-paper",
        type=Path,
        help="Optional manuscript whose SHA-256 is recorded in the run manifest.",
    )
    parser.add_argument(
        "--trace-csv",
        type=Path,
        help="Optional governance-event trace with explicit observation exposure.",
    )
    parser.add_argument(
        "--trace-start", help="Inclusive ISO-8601 trace exposure start, with UTC offset."
    )
    parser.add_argument("--trace-end", help="Exclusive ISO-8601 trace exposure end, with UTC offset.")
    parser.add_argument("--trace-population", type=int)
    parser.add_argument("--trace-institutions", type=int)
    parser.add_argument("--trace-source", help="Trace provenance description supplied by its owner.")
    parser.add_argument(
        "--verify-witness",
        type=Path,
        help="Inspect a portable signed witness against separate trust and target files.",
    )
    parser.add_argument("--trust-registry", type=Path)
    parser.add_argument("--audit-target", type=Path)
    parser.add_argument(
        "--register-declaration",
        action="store_true",
        help="Register an otherwise valid declaration with persistent local replay state.",
    )
    parser.add_argument("--replay-db", type=Path)
    parser.add_argument("--verifier-id", help="16-byte verifier identity encoded in hexadecimal.")
    arguments = parser.parse_args(argv)
    trace_parameters = (
        arguments.trace_start,
        arguments.trace_end,
        arguments.trace_population,
        arguments.trace_institutions,
        arguments.trace_source,
    )
    if arguments.trace_csv and any(value is None for value in trace_parameters):
        parser.error(
            "--trace-csv requires --trace-start, --trace-end, --trace-population, --trace-institutions and --trace-source"
        )
    if any(value is not None for value in trace_parameters) and not arguments.trace_csv:
        parser.error("trace exposure options require --trace-csv")
    if arguments.verify_witness:
        if not arguments.trust_registry or not arguments.audit_target:
            parser.error("--verify-witness requires --trust-registry and --audit-target")
        if arguments.figures_only or arguments.trace_csv:
            parser.error(
                "witness inspection is a separate command from simulation and figure regeneration"
            )
        if arguments.register_declaration and (not arguments.replay_db or not arguments.verifier_id):
            parser.error("registration requires --replay-db and --verifier-id")
        if (arguments.replay_db or arguments.verifier_id) and not arguments.register_declaration:
            parser.error("replay state options require --register-declaration")
    elif (
        arguments.trust_registry
        or arguments.audit_target
        or arguments.register_declaration
        or arguments.replay_db
        or arguments.verifier_id
    ):
        parser.error("witness options require --verify-witness")
    if arguments.figures_only and arguments.trace_csv:
        parser.error("trace replay requires a full simulation invocation")
    return arguments


def verify_witness_command(arguments: argparse.Namespace) -> int:
    trust = TrustRegistry.from_manifest(json.loads(arguments.trust_registry.read_bytes()))
    target = AuditTarget.from_dict(json.loads(arguments.audit_target.read_bytes()))
    with arguments.verify_witness.open("rb") as handle:
        package = handle.read(MAX_JSON_PACKAGE_BYTES + 1)
    replay = None
    if arguments.register_declaration:
        replay = ReplayStore(arguments.replay_db, bytes.fromhex(arguments.verifier_id))
    try:
        mode = "register" if arguments.register_declaration else "inspect"
        accepted, reason = EvidenceVerifier(trust, replay).verify(package, target, mode)
        print(
            json.dumps(
                {
                    "accepted": accepted,
                    "reason": reason,
                    "operation": mode,
                    "evidence_scope": "declared structural and authorization predicates",
                }
            )
        )
        return 0 if accepted else 2
    finally:
        if replay is not None:
            replay.close()


def execute_simulation(arguments: argparse.Namespace) -> int:
    print(f"Running {SIMUP_CODE_VERSION}")
    print(f"Script path: {Path(__file__).resolve()}")
    print(f"Output base folder: {BASE}")
    print(f"Profile: {RUN_PROFILE}; deterministic statistical seed: {ACTIVE_SEED}")

    if arguments.figures_only:
        write_publication_tables()
        figure_manifest = generate_publication_figures()
        validation = run_validation_audit()
        manifest = finalize_output_manifest()
        print(f"Regenerated {len(figure_manifest)} figures (PNG + PDF).")
        print(f"Validation checks passed: {(validation['status'] == 'PASS').sum()}/{len(validation)}")
        print(f"Manifest entries: {len(manifest)}")
        return 0

    prepare_output_directories()
    if arguments.trace_csv:
        replay_governance_trace(
            arguments.trace_csv,
            utc_timestamp(arguments.trace_start, "trace start"),
            utc_timestamp(arguments.trace_end, "trace end"),
            arguments.trace_population,
            arguments.trace_institutions,
            arguments.trace_source,
            DATA / "trace_replay",
        )
    export_workload_rationale()
    export_protocol_specification()
    print("[1/8] Closed-form, independent-Poisson, and causal daily experiments")
    baseline_mc, scenario_results, population_results = analytical_and_daily_monte_carlo()
    print("[2/8] Batching, proof-size, and retained-storage design space")
    batching, _storage = batching_and_storage_analysis()
    print("[3/8] Correlated temporal, workflow-delay, batching, and queue experiments")
    temporal_summary, _queue_summary = temporal_and_capacity_analysis()
    print("[4/8] Latin-hypercube uncertainty and PRCC sensitivity")
    uncertainty_and_sensitivity_analysis()
    print("[5/8] Signed reference workflow, persistent replay and protocol faults")
    _formal_verification, formal_metrics = formal_model_experiment()
    batching_implementation_experiments()
    metadata_linkage_experiments()
    trace_conformance_experiment()
    export_reviewer_traceability(bool(arguments.trace_csv))
    export_result_reconciliation()
    utilization = validator_utilization_compatibility(
        baseline_mc, scenario_results, population_results, temporal_summary
    )
    write_paper_and_reproducibility_artifacts(
        baseline_mc, population_results, batching, formal_metrics, utilization
    )
    print("[6/8] Publication tables and figures")
    write_publication_tables()
    figure_manifest = generate_publication_figures()
    print("[7/8] Blocking numerical and artifact validation")
    validation = run_validation_audit()
    print("[8/8] Final SHA-256 artifact manifest")
    manifest = finalize_output_manifest()

    baseline = analytical_component_rates(REPRESENTED_POPULATIONS[0], SCENARIOS[0])
    highest = population_results[
        population_results["population"] == max(REPRESENTED_POPULATIONS)
    ]["TPS_p99"].max()
    print("\nSimulation completed.")
    print(f"Data folder: {DATA}")
    print(f"Paper artifact folder: {ARTIFACTS}")
    print(f"CSV files generated: {len(list(DATA.glob('*.csv')))}")
    print(f"Publication figures generated: {len(figure_manifest)} (PNG + PDF)")
    print(f"Validation checks passed: {(validation['status'] == 'PASS').sum()}/{len(validation)}")
    print(f"Manifest entries: {len(manifest)}")
    print(f"Baseline 50M logical workload: {baseline['Daily_Events']:,.2f} events/day")
    print(f"Baseline 50M logical-event TPS: {baseline['TPS']:.3f}")
    print(f"Maximum 450M population-scenario p99 TPS: {highest:.3f}")
    print("Evidence scope: conditional design models and measured local reference execution.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    global ACTIVE_SEED, RUN_PROFILE, TEMPORAL_REPLICATIONS, FORMAL_MODEL_EVENTS, PEAK_BOOTSTRAP_REPLICATIONS
    global BASE, DATA, ARTIFACTS, TABLES, FIGURES, SUBMITTED_PAPER
    arguments = parse_arguments(argv)
    if arguments.verify_witness:
        try:
            return verify_witness_command(arguments)
        except (ValueError, OSError, KeyError, TypeError, sqlite3.Error) as exc:
            print(
                json.dumps({"accepted": False, "reason": "INVALID_VERIFIER_INPUT", "detail": str(exc)}),
                file=sys.stderr,
            )
            return 2
    ACTIVE_SEED, RUN_PROFILE = arguments.seed, arguments.profile
    TEMPORAL_REPLICATIONS, FORMAL_MODEL_EVENTS, PEAK_BOOTSTRAP_REPLICATIONS = (
        (1000, 4096, 1000) if RUN_PROFILE == "publication" else (12, 37, 100)
    )
    BASE = arguments.output_dir.resolve()
    DATA, ARTIFACTS = BASE / "htgl_data", BASE / "paper_artifacts"
    TABLES, FIGURES = ARTIFACTS / "tables", ARTIFACTS / "figures"
    candidate = arguments.submitted_paper or SOURCE_DIR / "1571348117 paper.pdf"
    SUBMITTED_PAPER = candidate.resolve() if candidate.is_file() else None
    if arguments.submitted_paper and not SUBMITTED_PAPER:
        raise ValueError("the supplied manuscript is not a readable file")
    for source in (arguments.trace_csv, SUBMITTED_PAPER):
        if source and any(source.resolve().is_relative_to(directory) for directory in (DATA, ARTIFACTS)):
            raise ValueError("place source inputs outside the generated data and artifact folders")
    validate_configuration()
    BASE.mkdir(parents=True, exist_ok=True)
    # A separate durable SQLite file provides an OS-backed exclusive write
    # transaction. Interrupted runs release the lock with connection closure.
    guard = sqlite3.connect(str(BASE / ".htgl-run-lock.sqlite"), timeout=0, isolation_level=None)
    try:
        try:
            guard.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            raise RuntimeError(f"cannot acquire exclusive simulation output lock for {BASE}") from exc
        for directory in (DATA, ARTIFACTS, TABLES, FIGURES):
            directory.mkdir(parents=True, exist_ok=True)
        return execute_simulation(arguments)
    finally:
        guard.close()


if __name__ == "__main__":
    raise SystemExit(main())
