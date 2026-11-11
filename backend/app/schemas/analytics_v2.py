"""Pydantic v2 schemas for Phase C Universal Repository Intelligence & Analytics.

Defines schemas for:
- Standardized methodology version constants.
- Multi-domain analytical metric payloads.
- Coverage assessments and sample size thresholds.
- Confidence scores and penalty breakdowns.
- Deterministic rule interpretations and diagnostics.
- AnalysisRun and AnalysisRunAttempt read and submission contracts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

CURRENT_METHODOLOGY_VERSION: str = "2026.1"


class MetricCoverageItem(BaseModel):
    """Data completeness and sufficiency evaluation for an analytical domain."""

    domain: str = Field(..., description="Analytical domain identifier")
    status: str = Field(..., description="Sufficiency status: 'sufficient', 'insufficient', 'empty', or 'degraded'")
    sample_size: int = Field(..., description="Number of observed entities in this domain")
    min_sample_required: int = Field(..., description="Minimum entity threshold required for statistical validity")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Detailed domain-specific coverage indicators")


class ConfidenceSummary(BaseModel):
    """Statistical confidence assessment and penalty telemetry."""

    score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score [0.0, 1.0]")
    band: str = Field(..., description="Confidence band: 'high', 'medium', 'low', or 'unreliable'")
    penalties: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of applied confidence penalties and their underlying triggers",
    )


class RuleInterpretation(BaseModel):
    """Deterministic intelligence finding and actionable diagnostics."""

    rule_id: str = Field(..., description="Deterministic rule identifier (e.g. 'RULE-DEV-001')")
    domain: str = Field(..., description="Target analytical domain")
    severity: str = Field(..., description="Finding severity: 'info', 'warning', 'critical', or 'positive'")
    finding: str = Field(..., description="Synthesized analytical finding")
    evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured supporting metrics and thresholds",
    )
    recommendation: Optional[str] = Field(
        default=None,
        description="Prescriptive recommendation for repository maintainers",
    )


class AnalysisPayload(BaseModel):
    """Structured multi-domain computed intelligence payload."""

    development_activity: Optional[Dict[str, Any]] = Field(default=None, description="Commit velocity & activity metrics")
    contributor_ecosystem: Optional[Dict[str, Any]] = Field(default=None, description="Contributor retention & concentration metrics")
    pull_request_intelligence: Optional[Dict[str, Any]] = Field(default=None, description="PR lifecycle & review metrics")
    issue_intelligence: Optional[Dict[str, Any]] = Field(default=None, description="NLP categorization & turnaround metrics")
    repository_synthesis: Optional[Dict[str, Any]] = Field(default=None, description="Cross-domain synthesis & health score")


class AnalysisRunSubmission(BaseModel):
    """Request contract for enqueueing or triggering an analytical run."""

    repository_id: int = Field(..., description="Target repository ID")
    collection_run_id: Optional[int] = Field(default=None, description="Triggering collection run lineage")
    requested_methodology_version: str = Field(
        default=CURRENT_METHODOLOGY_VERSION,
        description="Analytical methodology version requested",
    )
    force: bool = Field(
        default=False,
        description="Force execution even if an analysis already exists for current revision and methodology",
    )


class AnalysisRunAttemptRead(BaseModel):
    """Read contract for an individual execution attempt of an analysis run."""

    id: int = Field(..., description="Surrogate primary key of the attempt")
    analysis_run_id: int = Field(..., description="Associated analysis run ID")
    attempt_number: int = Field(..., description="Monotonic attempt sequence number (1, 2, 3, 4)")
    worker_id: str = Field(..., description="Identifier of the executing worker")
    outcome: str = Field(..., description="Execution outcome: 'running', 'completed', 'failed', 'lease_expired', 'interrupted'")
    error_category: Optional[str] = Field(
        default=None,
        description="Classified failure category if failed or interrupted ('retryable', 'non_retryable', 'stale_revision', 'lease_lost')",
    )
    error_message: Optional[str] = Field(default=None, description="Error diagnostics if failed or interrupted")
    interruption_reason: Optional[str] = Field(
        default=None,
        description="Reason code or explanation if the attempt was interrupted",
    )
    started_at: datetime = Field(..., description="Timestamp when the attempt started")
    completed_at: Optional[datetime] = Field(default=None, description="Timestamp when the attempt finalized")

    model_config = ConfigDict(from_attributes=True)


AnalyticsAttemptRead = AnalysisRunAttemptRead


class AnalysisRunRead(BaseModel):
    """Complete response model for an analytical run and its audit attempts."""

    id: int = Field(..., description="Surrogate primary key of the analysis run")
    repository_id: int = Field(..., description="Target repository ID")
    collection_run_id: Optional[int] = Field(default=None, description="Associated collection run ID")
    status: str = Field(..., description="Current status: 'pending', 'running', 'completed', 'failed', 'cancelled'")
    requested_source_revision: int = Field(..., description="Repository revision when run was enqueued")
    analytics_source_revision: Optional[int] = Field(default=None, description="Repository revision read inside worker snapshot")
    requested_methodology_version: Optional[str] = Field(default=None, description="Requested methodology version")
    methodology_version: Optional[str] = Field(default=None, description="Actual methodology version computed")
    analysis_reference_time: Optional[datetime] = Field(default=None, description="Frozen UTC reference time anchoring durations")
    source_content_digest: Optional[str] = Field(default=None, description="SHA-256 digest of analyzed entities")
    worker_id: Optional[str] = Field(default=None, description="Identifier of worker currently holding lease")
    lease_version: int = Field(..., description="Optimistic locking lease version")
    retry_count: int = Field(..., description="Retries attempted after initial execution (0 to 3)")
    max_retries: int = Field(..., description="Maximum allowed retries (3)")
    metrics_payload: Optional[Dict[str, Any]] = Field(default=None, description="Multi-domain computed metrics")
    coverage_summary: Optional[Dict[str, Any]] = Field(default=None, description="Domain coverage evaluations")
    confidence_summary: Optional[Dict[str, Any]] = Field(default=None, description="Confidence score and band")
    rule_interpretations: Optional[List[Dict[str, Any]]] = Field(default=None, description="Synthesized findings")
    started_at: datetime = Field(..., description="Timestamp when the run was enqueued")
    completed_at: Optional[datetime] = Field(default=None, description="Timestamp when the run reached terminal state")
    error_message: Optional[str] = Field(default=None, description="Terminal error description")
    execution_time_ms: float = Field(0.0, description="Execution duration in milliseconds")
    attempts: List[AnalysisRunAttemptRead] = Field(default_factory=list, description="Audit history of execution attempts")

    model_config = ConfigDict(from_attributes=True)


class RepositoryFreshnessEnvelope(BaseModel):
    """Collection and revision metadata exposed strictly in outer API envelopes (outside metrics_payload)."""

    repository_id: int = Field(..., description="Target repository ID")
    repo_name: str = Field(..., description="Canonical owner/repo identifier")
    analytics_source_revision: int = Field(..., description="Current repository analytics source revision")
    last_synced_at: Optional[datetime] = Field(default=None, description="Last GitHub collection sync timestamp")
    last_analyzed_at: Optional[datetime] = Field(default=None, description="Timestamp of last up-to-date analysis")
    primary_language: Optional[str] = Field(default=None, description="Primary repository language")
    stars_count: int = Field(0, description="GitHub stargazer count")
    forks_count: int = Field(0, description="GitHub forks count")
    is_archived: bool = Field(False, description="Whether repository is archived")


class AnalysisRecomputeRequest(BaseModel):
    """Request contract for manual analysis recomputation under owner/name routing."""

    force: bool = Field(
        default=False,
        description="Force a fresh run even if a valid completed analysis exists for the current revision and methodology",
    )
    requested_methodology_version: str = Field(
        default=CURRENT_METHODOLOGY_VERSION,
        description="Requested analytical methodology version",
    )


class AnalysisRecomputeResponse(BaseModel):
    """Response contract returned after requesting or deduplicating repository analysis."""

    repository: RepositoryFreshnessEnvelope
    run: AnalysisRunRead
    enqueued: bool = Field(..., description="True when a new pending AnalysisRun was created")
    already_active: bool = Field(False, description="True when an existing pending or running job was reused")
    cached: bool = Field(False, description="True when an up-to-date completed run was returned without enqueueing")
    message: str = Field(..., description="Human-readable enqueue or cache status summary")


class AnalysisStatusResponse(BaseModel):
    """Operational telemetry contract for polling repository analysis progress and staleness."""

    repository: RepositoryFreshnessEnvelope
    has_analysis: bool = Field(..., description="True if at least one AnalysisRun exists for this repository")
    is_stale: bool = Field(
        ...,
        description="True when the latest completed run analyzed an older revision than repo.analytics_source_revision",
    )
    current_run: Optional[AnalysisRunRead] = Field(
        default=None,
        description="Active run if in flight, otherwise most recent AnalysisRun",
    )
    latest_completed_run_id: Optional[int] = Field(
        default=None,
        description="ID of the most recent completed AnalysisRun for the requested methodology, if any",
    )


class AnalysisLatestResponse(BaseModel):
    """Latest completed multi-domain repository intelligence response envelope."""

    repository: RepositoryFreshnessEnvelope
    is_stale: bool = Field(
        ...,
        description="True when completed run revision lags repository.analytics_source_revision",
    )
    active_run_id: Optional[int] = Field(default=None, description="In-flight pending or running AnalysisRun ID if any")
    active_run_status: Optional[str] = Field(default=None, description="In-flight pending or running AnalysisRun status if any")
    run: AnalysisRunRead


class AnalysisHistoryResponse(BaseModel):
    """Paginated historical AnalysisRun listing for a repository."""

    repository: RepositoryFreshnessEnvelope
    total: int = Field(..., description="Total matching historical AnalysisRun records")
    limit: int = Field(..., description="Requested page size")
    offset: int = Field(..., description="Requested offset")
    items: List[AnalysisRunRead] = Field(default_factory=list)


class AnalysisDomainSliceResponse(BaseModel):
    """Single-domain analytical slice projected from a completed AnalysisRun."""

    repository: RepositoryFreshnessEnvelope
    run_id: int = Field(..., description="Completed AnalysisRun ID")
    domain: str = Field(..., description="Requested domain identifier")
    analytics_source_revision: Optional[int] = Field(default=None, description="Analyzed repository revision")
    methodology_version: Optional[str] = Field(default=None, description="Computed methodology version")
    analysis_reference_time: Optional[datetime] = Field(default=None, description="Frozen UTC reference time")
    is_stale: bool = Field(..., description="True when completed run revision lags current source revision")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Computed metrics slice for the requested domain")
    coverage: Optional[Dict[str, Any]] = Field(default=None, description="Domain coverage summary")
    confidence_breakdown: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Domain confidence score and applied penalties",
    )
    rules: List[Dict[str, Any]] = Field(default_factory=list, description="Consolidated rule interpretations for this domain")
