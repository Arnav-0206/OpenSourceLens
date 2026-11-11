"""Phase C Multi-Domain Analytics Orchestration & Publication Service (Step C.7).

Coordinates all six analytical domains (C.2–C.6) against a single pinned
snapshot transaction and frozen UTC reference time, aggregates domain-level
coverage, confidence scores/penalties, consolidated rules, and lineage metadata,
validates against Pydantic v2 contracts (AnalysisPayload, ConfidenceSummary),
and publishes atomically through AnalyticsJobQueue.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from backend.app.core.exceptions import (
    AttemptImmutableException,
    EntityNotFoundException,
    GhostWorkerRejectedException,
    InvalidParameterException,
    OpenSourceLensException,
)
from backend.app.database.models import (
    AnalysisRun,
    AnalysisRunAttempt,
    CollectionRun,
    Repository,
)
from backend.app.schemas.analytics_v2 import (
    AnalysisPayload,
    CURRENT_METHODOLOGY_VERSION,
    ConfidenceSummary,
    MetricCoverageItem,
    RuleInterpretation,
)
from backend.app.services.analytics.base import BaseMetricCalculator, CalculationResult
from backend.app.services.analytics.contributors import ContributorIntelligenceCalculator
from backend.app.services.analytics.cross_domain import CrossDomainSynthesisCalculator
from backend.app.services.analytics.development_activity import (
    DevelopmentActivityCalculator,
    _ensure_utc,
)
from backend.app.services.analytics.issues import IssueIntelligenceCalculator
from backend.app.services.analytics.pull_requests import PullRequestIntelligenceCalculator
from backend.app.services.analytics.releases import ReleaseEvolutionCalculator
from backend.app.services.analytics.rule_engine import consolidate_rules
from backend.app.services.analytics_queue import AnalyticsJobQueue

logger = logging.getLogger(__name__)

BASE_ANALYTICS_DOMAINS: Tuple[str, ...] = (
    "development_activity",
    "contributor_ecosystem",
    "pull_request_intelligence",
    "issue_intelligence",
    "release_evolution",
)
ALL_ANALYTICS_DOMAINS: Tuple[str, ...] = (
    *BASE_ANALYTICS_DOMAINS,
    "repository_synthesis",
)


class DomainCalculationError(OpenSourceLensException):
    """Raised when one of the analytical domain calculators fails during execution."""

    def __init__(
        self,
        domain: str,
        cause: Exception,
        *,
        retryable: bool = True,
    ) -> None:
        self.domain = domain
        self.cause = cause
        self.retryable = retryable
        super().__init__(
            message=f"Domain '{domain}' failed during analytics calculation: {type(cause).__name__}: {cause}",
            status_code=500,
            error_code="DOMAIN_CALCULATION_ERROR",
            details={"domain": domain, "cause_type": type(cause).__name__, "retryable": retryable},
        )


class NonRetryableAnalyticsError(OpenSourceLensException):
    """Raised when an analytical execution error cannot be resolved by retrying."""

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            status_code=422,
            error_code="NON_RETRYABLE_ANALYTICS_ERROR",
            details=details or {},
        )


class StaleAnalysisRevisionException(OpenSourceLensException):
    """Raised when a worker attempts to publish over a newer completed analysis revision."""

    def __init__(
        self,
        message: str,
        *,
        snapshot_revision: Optional[int] = None,
        newer_revision: Optional[int] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=409,
            error_code="STALE_ANALYSIS_REVISION",
            details={
                "snapshot_revision": snapshot_revision,
                "newer_revision": newer_revision,
            },
        )


@dataclass
class OrchestratedAnalysisBundle:
    """Validated multi-domain computation output ready for durable queue publication."""

    repository_id: int
    snapshot_revision: int
    requested_source_revision: Optional[int]
    methodology_version: str
    analysis_reference_time: datetime
    execution_time_ms: float
    domain_results: Dict[str, CalculationResult]
    validated_payload: AnalysisPayload
    metrics_payload: Dict[str, Any]
    coverage_summary: Dict[str, Any]
    confidence_summary: Dict[str, Any]
    rule_interpretations: List[Dict[str, Any]]
    unavailable_domains: List[str]
    is_bounded_or_partial: bool
    is_full_history: bool
    collection_lineage: Dict[str, Any]


@dataclass
class OrchestrationExecutionResult:
    """Structured outcome of processing a claimed or submitted analytics job."""

    status: str
    run_id: Optional[int]
    attempt_id: Optional[int]
    attempt_number: Optional[int]
    repository_id: Optional[int]
    snapshot_revision: Optional[int]
    requested_source_revision: Optional[int]
    published: bool
    stale_revision_detected: bool = False
    replacement_run_enqueued: bool = False
    failure_category: Optional[str] = None
    error_message: Optional[str] = None
    bundle: Optional[OrchestratedAnalysisBundle] = None


def _default_calculators() -> Dict[str, Optional[BaseMetricCalculator]]:
    """Instantiate the standard Phase C.2–C.6 analytical domain calculators."""
    return {
        "development_activity": DevelopmentActivityCalculator(),
        "contributor_ecosystem": ContributorIntelligenceCalculator(),
        "pull_request_intelligence": PullRequestIntelligenceCalculator(),
        "issue_intelligence": IssueIntelligenceCalculator(),
        "release_evolution": ReleaseEvolutionCalculator(),
        "repository_synthesis": CrossDomainSynthesisCalculator(),
    }


def _confidence_band(score: float, *, all_core_empty_or_unavailable: bool) -> str:
    """Map aggregate confidence score in [0.0, 1.0] to ConfidenceSummary band."""
    if all_core_empty_or_unavailable or score < 0.40:
        return "unreliable"
    if score >= 0.80:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def _make_unavailable_result(
    domain: str,
    snapshot_revision: int,
    reason: str,
    *,
    embedded_release: Optional[CalculationResult] = None,
) -> CalculationResult:
    """Construct an explicit unavailable CalculationResult without fabricating zero metrics."""
    coverage = MetricCoverageItem(
        domain=domain,
        status="unavailable",
        sample_size=0,
        min_sample_required=1,
        details={
            "available": False,
            "availability_state": "unavailable",
            "reason": reason,
        },
    )
    metrics: Dict[str, Any] = {
        "status": "unavailable",
        "available": False,
        "reason": reason,
        "coverage": coverage.model_dump(),
    }
    if domain == "release_evolution":
        # Preserve expected release subdomain status shapes when synthesis inspects unavailable release outputs
        metrics.update(
            {
                "scope": {"status": "unavailable", "mode": "unavailable", "is_bounded_or_partial": True},
                "cadence": {"status": "unavailable", "total_releases": 0, "regularity_classification": "unavailable"},
                "prerelease_surges": {"status": "unavailable", "valid_surge_releases_count": 0},
                "semver_adoption": {"status": "unavailable", "total_releases": 0, "semver_adoption_rate": None},
            }
        )
    elif domain == "repository_synthesis":
        metrics["chi_deferred"] = True
        metrics["chi_deferral_reason"] = (
            "Composite Health Index [0-100] is formally deferred in favor of the objective Multi-Domain Diagnostic Scorecard."
        )
        if embedded_release is not None:
            metrics["release_evolution"] = {
                **embedded_release.metrics,
                "coverage": embedded_release.coverage.model_dump(),
            }
    penalty_entry = {
        "rule": f"PENALTY-UNAVAILABLE-{domain.upper()}",
        "penalty": 1.0,
        "reason": reason,
    }
    rule = RuleInterpretation(
        rule_id="RULE-ORCH-UNAVAILABLE",
        domain=domain,
        severity="warning",
        finding=f"Domain '{domain}' unavailable: {reason}",
        evidence={"domain": domain, "availability_state": "unavailable"},
        recommendation="Ensure all required domain calculators and underlying ingestion data streams are configured.",
    )
    return CalculationResult(
        domain=domain,
        metrics=metrics,
        coverage=coverage,
        confidence_penalty=1.0,
        rules=[rule],
        details={"penalties": [penalty_entry], "snapshot_revision": snapshot_revision, "available": False},
    )


class AnalyticsOrchestrator:
    """Orchestrates snapshot-consistent execution and durable publication of all C.2–C.6 domains."""

    def __init__(
        self,
        job_queue: Optional[AnalyticsJobQueue] = None,
        methodology_version: str = CURRENT_METHODOLOGY_VERSION,
        calculators: Optional[Dict[str, Optional[BaseMetricCalculator]]] = None,
    ) -> None:
        """Initialize orchestrator with durable analytics queue, methodology version, and calculators."""
        if not methodology_version or not methodology_version.strip():
            raise InvalidParameterException("methodology_version must be a non-empty string.", parameter="methodology_version")
        self.queue = job_queue or AnalyticsJobQueue()
        self.methodology_version = methodology_version.strip()
        base_defaults = _default_calculators()
        if calculators is not None:
            self._calculators: Dict[str, Optional[BaseMetricCalculator]] = dict(calculators)
        else:
            self._calculators = base_defaults

    @staticmethod
    def _classify_exception(exc: Exception) -> Tuple[str, bool]:
        """Classify failure mode into ('lease_lost'|'stale_revision'|'non_retryable'|'retryable', retryable)."""
        if isinstance(exc, (GhostWorkerRejectedException, AttemptImmutableException)):
            return "lease_lost", False
        if isinstance(exc, StaleAnalysisRevisionException):
            return "stale_revision", False
        if isinstance(exc, DomainCalculationError):
            if not exc.retryable:
                return "non_retryable", False
            inner = exc.cause
            if isinstance(inner, (EntityNotFoundException, InvalidParameterException, NonRetryableAnalyticsError, ValidationError)):
                return "non_retryable", False
            return "retryable", True
        if isinstance(exc, (EntityNotFoundException, InvalidParameterException, NonRetryableAnalyticsError, ValidationError)):
            return "non_retryable", False
        return "retryable", True

    @staticmethod
    def _inspect_collection_lineage(
        latest_run: Optional[CollectionRun],
        explicit_collection_run_id: Optional[int],
    ) -> Dict[str, Any]:
        """Extract snapshot-consistent collection lineage and bounded vs full-history classification."""
        if latest_run is None:
            return {
                "collection_run_id": explicit_collection_run_id,
                "collection_mode": "unknown_unverified",
                "collection_status": None,
                "is_partial_collection": True,
                "is_bounded_or_partial": True,
                "is_full_history": False,
                "representativeness": "unverified_no_collection_run",
            }

        mode = (latest_run.mode or "unknown").lower().strip()
        run_status = (latest_run.status or "unknown").lower().strip()
        bounded_modes = {"fast_snapshot", "standard_diagnostic"}
        is_full_history = (
            mode == "full_history"
            and run_status == "completed"
            and not bool(latest_run.is_partial)
        )
        is_bounded_or_partial = not is_full_history
        if is_full_history:
            representativeness = "full_history"
        elif mode == "fast_snapshot":
            representativeness = "fast_snapshot_bounded"
        elif mode in bounded_modes:
            representativeness = "bounded_diagnostic"
        else:
            representativeness = "partial_or_incomplete_collection"

        return {
            "collection_run_id": explicit_collection_run_id if explicit_collection_run_id is not None else latest_run.id,
            "collection_mode": mode,
            "collection_status": run_status,
            "is_partial_collection": bool(latest_run.is_partial),
            "is_bounded_or_partial": is_bounded_or_partial,
            "is_full_history": is_full_history,
            "representativeness": representativeness,
        }

    def _run_domains_in_snapshot(
        self,
        repository_id: int,
        db: Session,
        ref_time: datetime,
        snapshot_revision: int,
        collection_run_id: Optional[int] = None,
    ) -> Tuple[Dict[str, CalculationResult], Dict[str, Any]]:
        """Execute all six domain calculators sequentially on one pinned snapshot transaction."""
        repo = db.scalar(select(Repository).where(Repository.id == repository_id))
        if repo is None:
            raise EntityNotFoundException(
                message=f"Repository ID {repository_id} not found inside analysis snapshot.",
                entity_type="Repository",
                entity_id=repository_id,
            )

        latest_collection_run = None
        if collection_run_id is not None:
            latest_collection_run = db.scalar(
                select(CollectionRun).where(
                    CollectionRun.id == collection_run_id,
                    CollectionRun.repository_id == repository_id,
                )
            )
        if latest_collection_run is None:
            latest_collection_run = db.scalar(
                select(CollectionRun)
                .where(CollectionRun.repository_id == repository_id)
                .order_by(CollectionRun.started_at.desc(), CollectionRun.id.desc())
                .limit(1)
            )

        lineage = self._inspect_collection_lineage(latest_collection_run, collection_run_id)
        domain_results: Dict[str, CalculationResult] = {}

        # Execute base domains C.2–C.6 (Module 5) against identical (db, ref_time, snapshot_revision)
        for key in BASE_ANALYTICS_DOMAINS:
            calc = self._calculators.get(key)
            if calc is None:
                domain_results[key] = _make_unavailable_result(
                    domain=key,
                    snapshot_revision=snapshot_revision,
                    reason=f"Calculator for domain '{key}' is not configured.",
                )
                continue
            try:
                res = calc.calculate(
                    repository_id=repository_id,
                    db=db,
                    analysis_reference_time=ref_time,
                    snapshot_revision=snapshot_revision,
                )
            except (EntityNotFoundException, InvalidParameterException, NonRetryableAnalyticsError):
                raise
            except Exception as exc:
                raise DomainCalculationError(domain=key, cause=exc, retryable=True) from exc

            if res is None:
                domain_results[key] = _make_unavailable_result(
                    domain=key,
                    snapshot_revision=snapshot_revision,
                    reason=f"Calculator for domain '{key}' returned no result.",
                )
            else:
                domain_results[key] = res

        # Execute Module 6 Cross-Domain Synthesis reusing precomputed domain results (no duplicate queries/computations)
        syn_calc = self._calculators.get("repository_synthesis")
        rel_res = domain_results["release_evolution"]
        if syn_calc is None:
            domain_results["repository_synthesis"] = _make_unavailable_result(
                domain="repository_synthesis",
                snapshot_revision=snapshot_revision,
                reason="Calculator for domain 'repository_synthesis' is not configured.",
                embedded_release=rel_res,
            )
        else:
            try:
                if isinstance(syn_calc, CrossDomainSynthesisCalculator):
                    syn_res = syn_calc.calculate(
                        repository_id=repository_id,
                        db=db,
                        analysis_reference_time=ref_time,
                        snapshot_revision=snapshot_revision,
                        precomputed_domains=domain_results,
                    )
                else:
                    # Support injected custom synthesis calculators adhering to BaseMetricCalculator contract
                    try:
                        syn_res = syn_calc.calculate(  # type: ignore[call-arg]
                            repository_id=repository_id,
                            db=db,
                            analysis_reference_time=ref_time,
                            snapshot_revision=snapshot_revision,
                            precomputed_domains=domain_results,
                        )
                    except TypeError:
                        syn_res = syn_calc.calculate(
                            repository_id=repository_id,
                            db=db,
                            analysis_reference_time=ref_time,
                            snapshot_revision=snapshot_revision,
                        )
            except (EntityNotFoundException, InvalidParameterException, NonRetryableAnalyticsError):
                raise
            except Exception as exc:
                raise DomainCalculationError(domain="repository_synthesis", cause=exc, retryable=True) from exc

            if syn_res is None:
                domain_results["repository_synthesis"] = _make_unavailable_result(
                    domain="repository_synthesis",
                    snapshot_revision=snapshot_revision,
                    reason="Calculator for domain 'repository_synthesis' returned no result.",
                    embedded_release=rel_res,
                )
            else:
                domain_results["repository_synthesis"] = syn_res

        return domain_results, lineage

    def build_publication_bundle(
        self,
        *,
        repository_id: int,
        snapshot_revision: int,
        analysis_reference_time: datetime,
        domain_results: Dict[str, Optional[CalculationResult]],
        collection_lineage: Optional[Dict[str, Any]] = None,
        requested_source_revision: Optional[int] = None,
        analysis_run_id: Optional[int] = None,
        attempt_number: Optional[int] = None,
        execution_time_ms: float = 0.0,
    ) -> OrchestratedAnalysisBundle:
        """Assemble, aggregate, and validate AnalysisPayload, coverage, confidence, and rules across all 6 domains."""
        ref_time = _ensure_utc(analysis_reference_time)
        if ref_time is None:
            raise InvalidParameterException("analysis_reference_time must not be None.", parameter="analysis_reference_time")

        lineage = dict(
            collection_lineage
            or {
                "collection_run_id": None,
                "collection_mode": "unknown_unverified",
                "collection_status": None,
                "is_partial_collection": True,
                "is_bounded_or_partial": True,
                "is_full_history": False,
                "representativeness": "unverified_no_collection_run",
            }
        )

        # 1. Normalize all 6 domain slots so missing/None domain results are explicitly marked unavailable
        normalized_results: Dict[str, CalculationResult] = {}
        unavailable_domains: List[str] = []
        for key in BASE_ANALYTICS_DOMAINS:
            raw_res = domain_results.get(key)
            if raw_res is None:
                unavailable_domains.append(key)
                normalized_results[key] = _make_unavailable_result(
                    domain=key,
                    snapshot_revision=snapshot_revision,
                    reason=f"Domain '{key}' result was missing from orchestrated domain outputs.",
                )
            else:
                if raw_res.coverage.status == "unavailable" or raw_res.metrics.get("available") is False:
                    unavailable_domains.append(key)
                normalized_results[key] = raw_res

        rel_res = normalized_results["release_evolution"]
        syn_raw = domain_results.get("repository_synthesis")
        if syn_raw is None:
            unavailable_domains.append("repository_synthesis")
            normalized_results["repository_synthesis"] = _make_unavailable_result(
                domain="repository_synthesis",
                snapshot_revision=snapshot_revision,
                reason="Domain 'repository_synthesis' result was missing from orchestrated domain outputs.",
                embedded_release=rel_res,
            )
        else:
            if syn_raw.coverage.status == "unavailable" or syn_raw.metrics.get("available") is False:
                unavailable_domains.append("repository_synthesis")
            normalized_results["repository_synthesis"] = syn_raw

        syn_res = normalized_results["repository_synthesis"]
        rel_penalties = [dict(p) for p in ((rel_res.details or {}).get("penalties") or [])]

        # 2. Construct locked 5-key AnalysisPayload with embedded release_evolution + release penalties & lineage
        syn_metrics = dict(syn_res.metrics)
        existing_rel_block = syn_metrics.get("release_evolution")
        if isinstance(existing_rel_block, dict):
            rel_embedded = dict(existing_rel_block)
        else:
            rel_embedded = {
                **rel_res.metrics,
                "coverage": rel_res.coverage.model_dump(),
            }
        rel_embedded.setdefault("coverage", rel_res.coverage.model_dump())
        rel_embedded["confidence"] = {
            "confidence_penalty": round(min(1.0, max(0.0, float(rel_res.confidence_penalty))), 3),
            "penalties": rel_penalties,
        }
        syn_metrics["release_evolution"] = rel_embedded

        # Preserve explicit orchestration lineage metadata without introducing any 6th top-level AnalysisPayload key
        # and strictly excluding last_synced_at.
        syn_metrics["orchestration_metadata"] = {
            "methodology_version": self.methodology_version,
            "snapshot_revision": snapshot_revision,
            "requested_source_revision": requested_source_revision,
            "analysis_reference_time": ref_time.isoformat(),
            "analysis_run_id": analysis_run_id,
            "attempt_number": attempt_number,
            "execution_time_ms": round(float(execution_time_ms), 3),
            "unavailable_domains": list(unavailable_domains),
            "collection_lineage": lineage,
        }

        payload_dict = {
            "development_activity": dict(normalized_results["development_activity"].metrics),
            "contributor_ecosystem": dict(normalized_results["contributor_ecosystem"].metrics),
            "pull_request_intelligence": dict(normalized_results["pull_request_intelligence"].metrics),
            "issue_intelligence": dict(normalized_results["issue_intelligence"].metrics),
            "repository_synthesis": syn_metrics,
        }

        # Validate complete payload with actual Pydantic v2 schema
        validated_payload = AnalysisPayload(**payload_dict)
        metrics_payload = validated_payload.model_dump()

        # Verify release_evolution remains present inside validated repository_synthesis
        if "release_evolution" not in (metrics_payload.get("repository_synthesis") or {}):
            raise NonRetryableAnalyticsError(
                "Validated AnalysisPayload is missing embedded 'release_evolution' inside 'repository_synthesis'."
            )

        # 3. Aggregate coverage across ALL six original CalculationResults
        domain_statuses: Dict[str, str] = {}
        status_counts: Dict[str, int] = {
            "sufficient": 0,
            "degraded": 0,
            "insufficient": 0,
            "empty": 0,
            "unavailable": 0,
        }
        coverage_summary: Dict[str, Any] = {}
        for key in ALL_ANALYTICS_DOMAINS:
            res = normalized_results[key]
            cov_dump = res.coverage.model_dump()
            coverage_summary[key] = cov_dump
            st = cov_dump["status"]
            domain_statuses[key] = st
            status_counts[st] = status_counts.get(st, 0) + 1

        core_statuses = [domain_statuses[k] for k in BASE_ANALYTICS_DOMAINS]
        if all(st in ("empty", "unavailable") for st in core_statuses) and unavailable_domains:
            overall_cov_status = "unavailable" if all(st == "unavailable" for st in core_statuses) else "insufficient"
        elif all(st == "empty" for st in domain_statuses.values()):
            overall_cov_status = "empty"
        elif all(st == "sufficient" for st in domain_statuses.values()):
            overall_cov_status = "sufficient"
        elif sum(1 for st in core_statuses if st in ("sufficient", "degraded")) == 0:
            overall_cov_status = "insufficient"
        else:
            overall_cov_status = "degraded"

        corr_block = syn_metrics.get("cross_domain_correlations")
        corr_cov_summary: Dict[str, Any] = {}
        if isinstance(corr_block, dict):
            corr_pairs = corr_block.get("pairs") or []
            p2_entry = next((p for p in corr_pairs if isinstance(p, dict) and p.get("pair_id") == "PAIR-02"), None)
            corr_cov_summary = {
                "status": corr_block.get("status"),
                "tested_pairs_count": corr_block.get("tested_pairs_count", 0),
                "pair_statuses": {
                    p["pair_id"]: p.get("status")
                    for p in corr_pairs
                    if isinstance(p, dict) and "pair_id" in p
                },
                "pair02_collection_coverage": p2_entry.get("collection_coverage") if p2_entry else None,
            }

        coverage_summary["summary"] = {
            "overall_status": overall_cov_status,
            "total_domains": len(ALL_ANALYTICS_DOMAINS),
            "domain_statuses": domain_statuses,
            "status_counts": status_counts,
            "unavailable_domains": list(unavailable_domains),
            "representativeness": lineage["representativeness"],
            "is_bounded_or_partial": bool(lineage["is_bounded_or_partial"]),
            "is_full_history": bool(lineage["is_full_history"]),
            "cross_domain_correlations": corr_cov_summary,
        }
        coverage_summary["lineage"] = {
            "repository_id": repository_id,
            "methodology_version": self.methodology_version,
            "snapshot_revision": snapshot_revision,
            "requested_source_revision": requested_source_revision,
            "analysis_reference_time": ref_time.isoformat(),
            "analysis_run_id": analysis_run_id,
            "attempt_number": attempt_number,
            "execution_time_ms": round(float(execution_time_ms), 3),
            **lineage,
        }

        # 4. Aggregate confidence across all six original CalculationResults, preserving all penalties
        aggregated_penalties: List[Dict[str, Any]] = []
        domain_breakdown: Dict[str, Any] = {}
        domain_scores: List[float] = []

        for key in ALL_ANALYTICS_DOMAINS:
            res = normalized_results[key]
            d_penalty = round(min(1.0, max(0.0, float(res.confidence_penalty))), 3)
            d_score = round(max(0.0, 1.0 - d_penalty), 3)
            domain_scores.append(d_score)

            raw_penalties = list((res.details or {}).get("penalties") or [])
            tagged_penalties: List[Dict[str, Any]] = []
            for entry in raw_penalties:
                p_copy = dict(entry)
                p_copy["domain"] = key
                tagged_penalties.append(p_copy)
            if d_penalty > 0.0 and not tagged_penalties:
                tagged_penalties.append(
                    {
                        "domain": key,
                        "rule": f"PENALTY-{key.upper()}",
                        "penalty": d_penalty,
                        "reason": f"Domain '{key}' applied confidence penalty {d_penalty}.",
                    }
                )
            aggregated_penalties.extend(tagged_penalties)
            domain_breakdown[key] = {
                "score": d_score,
                "confidence_penalty": d_penalty,
                "coverage_status": res.coverage.status,
                "penalties_count": len(tagged_penalties),
                "penalties": tagged_penalties,
            }

        overall_conf_score = round(sum(domain_scores) / float(len(domain_scores)), 3)
        all_core_empty_or_unavail = all(st in ("empty", "unavailable") for st in core_statuses)
        band = _confidence_band(
            overall_conf_score,
            all_core_empty_or_unavailable=all_core_empty_or_unavail,
        )

        validated_conf = ConfidenceSummary(
            score=overall_conf_score,
            band=band,
            penalties=aggregated_penalties,
        )
        confidence_summary: Dict[str, Any] = {
            **validated_conf.model_dump(),
            "total_domains_evaluated": len(ALL_ANALYTICS_DOMAINS),
            "unavailable_domains": list(unavailable_domains),
            "is_bounded_or_partial": bool(lineage["is_bounded_or_partial"]),
            "domain_breakdown": domain_breakdown,
        }

        # 5. Consolidate rules deterministically across all 6 original CalculationResults
        consolidated_rules = consolidate_rules(
            [normalized_results[k] for k in BASE_ANALYTICS_DOMAINS],
            list(syn_res.rules),
        )
        rule_interpretations = [r.model_dump() for r in consolidated_rules]

        return OrchestratedAnalysisBundle(
            repository_id=repository_id,
            snapshot_revision=snapshot_revision,
            requested_source_revision=requested_source_revision,
            methodology_version=self.methodology_version,
            analysis_reference_time=ref_time,
            execution_time_ms=round(float(execution_time_ms), 3),
            domain_results=normalized_results,
            validated_payload=validated_payload,
            metrics_payload=metrics_payload,
            coverage_summary=coverage_summary,
            confidence_summary=confidence_summary,
            rule_interpretations=rule_interpretations,
            unavailable_domains=unavailable_domains,
            is_bounded_or_partial=bool(lineage["is_bounded_or_partial"]),
            is_full_history=bool(lineage["is_full_history"]),
            collection_lineage=lineage,
        )

    def run_snapshot_analysis(
        self,
        repository_id: int,
        db: Session,
        *,
        analysis_reference_time: Optional[datetime] = None,
        snapshot_revision: Optional[int] = None,
        requested_source_revision: Optional[int] = None,
        analysis_run_id: Optional[int] = None,
        attempt_number: Optional[int] = None,
        collection_run_id: Optional[int] = None,
    ) -> OrchestratedAnalysisBundle:
        """Compute and validate a full 6-domain analytics bundle on a single snapshot.

        If `snapshot_revision` and `analysis_reference_time` are provided, assumes caller
        is already inside `AnalyticsJobQueue.open_analysis_snapshot(...)`; otherwise opens
        a pinned read snapshot cleanly via `self.queue.open_analysis_snapshot(...)`.
        """
        t0 = time.perf_counter()

        if snapshot_revision is not None and analysis_reference_time is not None:
            ref_time = _ensure_utc(analysis_reference_time)
            domain_results, lineage = self._run_domains_in_snapshot(
                repository_id=repository_id,
                db=db,
                ref_time=ref_time,  # type: ignore[arg-type]
                snapshot_revision=snapshot_revision,
                collection_run_id=collection_run_id,
            )
            exec_ms = (time.perf_counter() - t0) * 1000.0
            return self.build_publication_bundle(
                repository_id=repository_id,
                snapshot_revision=snapshot_revision,
                analysis_reference_time=ref_time,  # type: ignore[arg-type]
                domain_results=domain_results,
                collection_lineage=lineage,
                requested_source_revision=requested_source_revision,
                analysis_run_id=analysis_run_id,
                attempt_number=attempt_number,
                execution_time_ms=exec_ms,
            )

        # Clear any stray read-only SQLAlchemy autobegin transaction (e.g. from prior SELECTs)
        # so open_analysis_snapshot() emits BEGIN DEFERRED and pins a true WAL read snapshot.
        if db.in_transaction() and not db.new and not db.dirty and not db.deleted:
            db.rollback()

        with self.queue.open_analysis_snapshot(db=db, repository_id=repository_id) as (snap_rev, snap_ref):
            domain_results, lineage = self._run_domains_in_snapshot(
                repository_id=repository_id,
                db=db,
                ref_time=snap_ref,
                snapshot_revision=snap_rev,
                collection_run_id=collection_run_id,
            )
        exec_ms = (time.perf_counter() - t0) * 1000.0
        return self.build_publication_bundle(
            repository_id=repository_id,
            snapshot_revision=snap_rev,
            analysis_reference_time=snap_ref,
            domain_results=domain_results,
            collection_lineage=lineage,
            requested_source_revision=requested_source_revision,
            analysis_run_id=analysis_run_id,
            attempt_number=attempt_number,
            execution_time_ms=exec_ms,
        )

    def _abort_superseded_stale_run(
        self,
        *,
        run_id: int,
        attempt_id: int,
        worker_id: str,
        expected_lease_version: int,
        snapshot_revision: int,
        newer_revision: int,
        db: Session,
    ) -> str:
        """Prevent a stale worker from publishing over an already-completed newer revision."""
        reason = (
            f"Stale snapshot revision {snapshot_revision} superseded by newer completed "
            f"analysis at revision {newer_revision}; publication aborted."
        )
        now = datetime.now(timezone.utc)

        # Conditional update on AnalysisRun first while verifying lease ownership
        res_run = db.execute(
            update(AnalysisRun)
            .where(
                AnalysisRun.id == run_id,
                AnalysisRun.status == "running",
                AnalysisRun.worker_id == worker_id,
                AnalysisRun.lease_version == expected_lease_version,
            )
            .values(
                status="cancelled",
                worker_id=None,
                leased_until=None,
                completed_at=now,
                error_message=reason,
            )
        )
        if res_run.rowcount != 1:
            db.rollback()
            raise GhostWorkerRejectedException(
                f"Cannot abort stale run {run_id}: worker lease expired or worker identity mismatch."
            )

        # Finalize active attempt through application gatekeeper
        self.queue.finalize_attempt(
            attempt_id=attempt_id,
            outcome="interrupted",
            worker_id=worker_id,
            error_message=reason,
            db=db,
        )
        return reason

    def process_claimed_run(
        self,
        run: AnalysisRun,
        attempt: AnalysisRunAttempt,
        worker_id: str,
        db: Session,
    ) -> OrchestrationExecutionResult:
        """Execute all six domain calculators for a claimed AnalysisRun and publish atomically."""
        run_id = run.id
        attempt_id = attempt.id
        attempt_number = attempt.attempt_number
        expected_lease_version = run.lease_version
        repo_id = run.repository_id
        req_rev = run.requested_source_revision
        col_run_id = run.collection_run_id

        t0 = time.perf_counter()
        snapshot_revision: Optional[int] = None

        try:
            # Clear implicit SessionTransaction opened by claim_job()'s post-commit db.refresh()
            # so open_analysis_snapshot() executes BEGIN DEFERRED and pins the SQLite WAL read mark.
            if db.in_transaction() and not db.new and not db.dirty and not db.deleted:
                db.rollback()

            with self.queue.open_analysis_snapshot(db=db, repository_id=repo_id) as (snap_rev, ref_time):
                snapshot_revision = snap_rev
                if snap_rev < req_rev:
                    raise StaleAnalysisRevisionException(
                        f"Pinned snapshot revision ({snap_rev}) is older than requested revision ({req_rev}).",
                        snapshot_revision=snap_rev,
                        newer_revision=req_rev,
                    )

                domain_results, lineage = self._run_domains_in_snapshot(
                    repository_id=repo_id,
                    db=db,
                    ref_time=ref_time,
                    snapshot_revision=snap_rev,
                    collection_run_id=col_run_id,
                )

            exec_ms = (time.perf_counter() - t0) * 1000.0
            bundle = self.build_publication_bundle(
                repository_id=repo_id,
                snapshot_revision=snap_rev,
                analysis_reference_time=ref_time,
                domain_results=domain_results,
                collection_lineage=lineage,
                requested_source_revision=req_rev,
                analysis_run_id=run_id,
                attempt_number=attempt_number,
                execution_time_ms=exec_ms,
            )

            # Outside read snapshot: verify no newer completed analysis already superseded this snapshot revision
            newer_completed = db.scalar(
                select(AnalysisRun)
                .where(
                    AnalysisRun.repository_id == repo_id,
                    AnalysisRun.id != run_id,
                    AnalysisRun.status == "completed",
                    AnalysisRun.methodology_version == self.methodology_version,
                    AnalysisRun.analytics_source_revision > snap_rev,
                )
                .order_by(AnalysisRun.analytics_source_revision.desc(), AnalysisRun.completed_at.desc())
                .limit(1)
            )
            if newer_completed is not None:
                reason = self._abort_superseded_stale_run(
                    run_id=run_id,
                    attempt_id=attempt_id,
                    worker_id=worker_id,
                    expected_lease_version=expected_lease_version,
                    snapshot_revision=snap_rev,
                    newer_revision=int(newer_completed.analytics_source_revision or 0),
                    db=db,
                )
                logger.warning("Blocked stale publication for run ID=%d: %s", run_id, reason)
                return OrchestrationExecutionResult(
                    status="stale_blocked",
                    run_id=run_id,
                    attempt_id=attempt_id,
                    attempt_number=attempt_number,
                    repository_id=repo_id,
                    snapshot_revision=snap_rev,
                    requested_source_revision=req_rev,
                    published=False,
                    stale_revision_detected=True,
                    replacement_run_enqueued=False,
                    failure_category="stale_revision",
                    error_message=reason,
                    bundle=bundle,
                )

            # Check repository head before atomic publish so we can detect stale replacement enqueueing cleanly
            current_repo_rev = db.scalar(
                select(Repository.analytics_source_revision).where(Repository.id == repo_id)
            )

            self.queue.publish_and_finalize(
                run_id=run_id,
                attempt_id=attempt_id,
                worker_id=worker_id,
                expected_lease_version=expected_lease_version,
                attempt_number=attempt_number,
                snapshot_revision=snap_rev,
                actual_methodology=self.methodology_version,
                reference_time=ref_time,
                metrics_payload=bundle.metrics_payload,
                coverage_summary=bundle.coverage_summary,
                confidence_summary=bundle.confidence_summary,
                rule_interpretations=bundle.rule_interpretations,
                db=db,
            )

            stale_vs_repo = current_repo_rev is not None and snap_rev < current_repo_rev
            replacement_exists = False
            if stale_vs_repo:
                active_rep = db.scalar(
                    select(AnalysisRun.id).where(
                        AnalysisRun.repository_id == repo_id,
                        AnalysisRun.status.in_(["pending", "running"]),
                    ).limit(1)
                )
                replacement_exists = active_rep is not None

            return OrchestrationExecutionResult(
                status="completed_stale_replacement_enqueued" if stale_vs_repo else "completed",
                run_id=run_id,
                attempt_id=attempt_id,
                attempt_number=attempt_number,
                repository_id=repo_id,
                snapshot_revision=snap_rev,
                requested_source_revision=req_rev,
                published=True,
                stale_revision_detected=stale_vs_repo,
                replacement_run_enqueued=replacement_exists,
                bundle=bundle,
            )

        except (GhostWorkerRejectedException, AttemptImmutableException):
            if db.in_transaction():
                db.rollback()
            raise

        except StaleAnalysisRevisionException as exc:
            if db.in_transaction():
                db.rollback()
            err_msg = f"[STALE_REVISION] {exc}"
            self.queue.record_attempt_failure(
                run_id=run_id,
                attempt_id=attempt_id,
                worker_id=worker_id,
                expected_lease_version=expected_lease_version,
                attempt_number=attempt_number,
                error_message=err_msg,
                db=db,
                outcome="interrupted",
            )
            refreshed = db.scalar(select(AnalysisRun).where(AnalysisRun.id == run_id))
            final_status = "retry_scheduled" if (refreshed and refreshed.status == "pending") else "stale_blocked"
            return OrchestrationExecutionResult(
                status=final_status,
                run_id=run_id,
                attempt_id=attempt_id,
                attempt_number=attempt_number,
                repository_id=repo_id,
                snapshot_revision=snapshot_revision,
                requested_source_revision=req_rev,
                published=False,
                stale_revision_detected=True,
                failure_category="stale_revision",
                error_message=err_msg,
            )

        except Exception as exc:
            if db.in_transaction():
                db.rollback()
            category, is_retryable = self._classify_exception(exc)
            tag = "RETRYABLE" if is_retryable else "NON_RETRYABLE"
            err_msg = f"[{tag}] {type(exc).__name__}: {exc}"
            logger.exception("Analytics execution failed for run ID=%d (%s): %s", run_id, tag, exc)

            self.queue.record_attempt_failure(
                run_id=run_id,
                attempt_id=attempt_id,
                worker_id=worker_id,
                expected_lease_version=expected_lease_version,
                attempt_number=attempt_number,
                error_message=err_msg,
                db=db,
                outcome="failed",
                retryable=is_retryable,
                error_category=category,
            )
            refreshed = db.scalar(select(AnalysisRun).where(AnalysisRun.id == run_id))
            if refreshed and refreshed.status == "pending":
                result_status = "retry_scheduled"
                final_cat = category
            else:
                result_status = "failed"
                final_cat = "exhausted_retries" if is_retryable else category

            return OrchestrationExecutionResult(
                status=result_status,
                run_id=run_id,
                attempt_id=attempt_id,
                attempt_number=attempt_number,
                repository_id=repo_id,
                snapshot_revision=snapshot_revision,
                requested_source_revision=req_rev,
                published=False,
                failure_category=final_cat,
                error_message=err_msg,
            )

    def process_next_job(
        self,
        db: Session,
        worker_id: Optional[str] = None,
        lease_duration_seconds: int = AnalyticsJobQueue.DEFAULT_LEASE_DURATION_SECONDS,
    ) -> Optional[OrchestrationExecutionResult]:
        """Claim and execute the next pending or expired analytics job from AnalyticsJobQueue."""
        active_worker = worker_id or self.queue.generate_worker_id("analytics")
        claimed = self.queue.claim_job(
            worker_id=active_worker,
            db=db,
            lease_duration_seconds=lease_duration_seconds,
        )
        if claimed is None:
            return None
        run, attempt = claimed
        return self.process_claimed_run(
            run=run,
            attempt=attempt,
            worker_id=active_worker,
            db=db,
        )

    def enqueue_and_process(
        self,
        repository_id: int,
        db: Session,
        *,
        worker_id: Optional[str] = None,
        collection_run_id: Optional[int] = None,
        force: bool = False,
        max_retries: int = AnalyticsJobQueue.MAX_ALLOWED_RETRIES,
        lease_duration_seconds: int = AnalyticsJobQueue.DEFAULT_LEASE_DURATION_SECONDS,
    ) -> OrchestrationExecutionResult:
        """Enqueue analysis for a repository and execute/publish immediately if claimed."""
        run, is_new = self.queue.enqueue_analysis(
            repo_id=repository_id,
            db=db,
            collection_run_id=collection_run_id,
            requested_methodology=self.methodology_version,
            force=force,
            max_retries=max_retries,
        )

        if not is_new and run.status == "completed":
            return OrchestrationExecutionResult(
                status="already_completed",
                run_id=run.id,
                attempt_id=None,
                attempt_number=None,
                repository_id=run.repository_id,
                snapshot_revision=run.analytics_source_revision,
                requested_source_revision=run.requested_source_revision,
                published=False,
            )

        active_worker = worker_id or self.queue.generate_worker_id("analytics")
        claimed = self.queue.claim_job(
            worker_id=active_worker,
            db=db,
            lease_duration_seconds=lease_duration_seconds,
        )
        if claimed is None:
            return OrchestrationExecutionResult(
                status="active_or_unclaimed",
                run_id=run.id,
                attempt_id=None,
                attempt_number=None,
                repository_id=run.repository_id,
                snapshot_revision=run.analytics_source_revision,
                requested_source_revision=run.requested_source_revision,
                published=False,
            )
        claimed_run, attempt = claimed
        return self.process_claimed_run(
            run=claimed_run,
            attempt=attempt,
            worker_id=active_worker,
            db=db,
        )


AnalyticsService = AnalyticsOrchestrator
