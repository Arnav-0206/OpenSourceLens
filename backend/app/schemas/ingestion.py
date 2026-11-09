"""Pydantic schemas for repository URL validation, eligibility, and confidence matrix."""

from __future__ import annotations

from enum import Enum
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ConfidenceLevel(str, Enum):
    """Statistical sample sufficiency tiers for analytical domains."""

    INSUFFICIENT_DATA = "insufficient_data"
    LIMITED_SAMPLE = "limited_sample"
    CONFIDENT = "confident"


class ParsedRepository(BaseModel):
    """Normalized and validated GitHub repository slug and canonical metadata."""

    model_config = ConfigDict(frozen=True)

    owner: str = Field(..., description="Repository owner/organization username.")
    repo: str = Field(..., description="Repository name.")
    full_name: str = Field(..., description="Normalized 'owner/repo' slug.")
    canonical_url: str = Field(..., description="Canonical HTTPS GitHub URL.")


class DomainConfidence(BaseModel):
    """Statistical confidence breakdown for a single analytical domain."""

    model_config = ConfigDict(from_attributes=True)

    domain: str = Field(..., description="Machine-readable domain identifier.")
    domain_name: str = Field(..., description="Human-readable domain title.")
    level: ConfidenceLevel = Field(
        ...,
        description="Assigned statistical sample sufficiency tier (independent of runtime model operational health).",
    )
    sample_count: int = Field(..., description="Observed sample volume for the key domain metric.")
    metric_name: str = Field(..., description="Name of the evaluated metric (e.g., commits, prs).")
    thresholds: Dict[str, Any] = Field(..., description="Threshold boundaries for this domain.")
    is_sufficient: bool = Field(..., description="Whether data meets minimal threshold for analysis.")
    message: str = Field(..., description="Explanatory description of confidence status.")
    warning: Optional[str] = Field(None, description="Actionable limitation warning for the UI.")


class RepositoryConfidenceReport(BaseModel):
    """Cross-domain synthesis of statistical sample confidence for a repository."""

    model_config = ConfigDict(from_attributes=True)

    overall_confidence: ConfidenceLevel = Field(
        ...,
        description="Overall repository statistical sample sufficiency tier.",
    )
    overall_score: float = Field(..., description="Composite confidence score between 0.0 and 1.0.")
    domains: Dict[str, DomainConfidence] = Field(..., description="Map of domain identifier to confidence.")
    summary: str = Field(..., description="Human-readable synthesis of repository diagnostic readiness.")
    sufficient_domains_count: int = Field(..., description="Count of domains with sufficient data.")
    total_domains_count: int = Field(..., description="Total evaluated analytical domains.")


class RepositoryEligibilityResult(BaseModel):
    """Comprehensive eligibility evaluation for dynamic repository ingestion."""

    model_config = ConfigDict(from_attributes=True)

    is_eligible: bool = Field(..., description="Whether the repository meets ingestion criteria.")
    repo_name: str = Field(..., description="Canonical 'owner/repo' identifier.")
    owner: str = Field(..., description="Repository owner username.")
    name: str = Field(..., description="Repository short name.")
    canonical_url: str = Field(..., description="Canonical GitHub URL.")
    github_id: Optional[int] = Field(None, description="GitHub repository numerical ID.")
    rejection_reason: Optional[str] = Field(None, description="Detailed reason if rejected.")
    
    # Metadata flags
    is_private: bool = Field(False, description="Whether the repository is private.")
    is_archived: bool = Field(False, description="Whether the repository is archived on GitHub.")
    is_fork: bool = Field(False, description="Whether the repository is a fork.")
    upstream_repo: Optional[str] = Field(None, description="Parent repository slug if forked.")
    is_template: bool = Field(False, description="Whether the repository is marked as a template.")
    is_empty: bool = Field(False, description="Whether the repository has no code or commits.")
    
    # Quantitative surface indicators
    stars_count: int = Field(0, description="Stargazer count.")
    forks_count: int = Field(0, description="Forks count.")
    open_issues_count: int = Field(0, description="Reported open issues count.")
    commits_count: Optional[int] = Field(None, description="Observed or reported commit count.")
    default_branch: Optional[str] = Field(None, description="Primary repository branch.")
    
    # Analytical readiness & badges
    confidence_report: Optional[RepositoryConfidenceReport] = Field(
        None, description="Decoupled statistical confidence breakdown."
    )
    badges: List[str] = Field(
        default_factory=list, description="Visual badge labels (e.g., 'Archived', 'Fork of ...')."
    )


class PaginatedResult(BaseModel):
    """Result envelope for multi-page GitHub REST API collection with truncation reporting."""

    model_config = ConfigDict(from_attributes=True)

    entity: str = Field(..., description="Entity category name (e.g., 'commits', 'pull_requests').")
    items: List[Dict[str, Any]] = Field(default_factory=list, description="Harvested record payloads.")
    total_requests: int = Field(0, description="Total HTTP requests consumed during pagination.")
    is_partial: bool = Field(False, description="Whether data was truncated before remote exhaustion.")
    partial_reason: Optional[str] = Field(None, description="Explanation if collection was truncated.")
    resumes_at: Optional[Any] = Field(None, description="Scheduled resumption timestamp if rate limited.")


class IngestionSummary(BaseModel):
    """Execution summary of a repository data collection run."""

    model_config = ConfigDict(from_attributes=True)

    repo_name: str = Field(..., description="Target repository slug (owner/repo).")
    mode: str = Field(..., description="Collection mode ('fast_snapshot' or 'standard_diagnostic').")
    is_partial: bool = Field(False, description="Whether collection was truncated before completion.")
    partial_reason: Optional[str] = Field(None, description="Explanation if truncated.")
    resumes_at: Optional[Any] = Field(None, description="Scheduled resumption timestamp if rate limited.")
    total_requests_consumed: int = Field(0, description="Total HTTP requests consumed.")
    items_collected: Dict[str, int] = Field(default_factory=dict, description="Counts of harvested entities.")
    entity_status: Dict[str, str] = Field(
        default_factory=dict,
        description="Execution status per entity stream ('completed', 'truncated', 'skipped', 'failed').",
    )
    records_mutated: int = Field(
        0,
        description="Total distinct database entity records inserted, updated, or deleted.",
    )
    execution_time_seconds: float = Field(0.0, description="Total execution duration in seconds.")
    errors: List[str] = Field(default_factory=list, description="Non-fatal warnings or error messages.")


class IngestionPreviewRequest(BaseModel):
    """Request payload for pre-flight repository ingestion evaluation and budgeting."""

    repo: str = Field(..., description="Repository URL or owner/repo slug to evaluate.", examples=["pallets/click"])


class IngestionPreviewResponse(BaseModel):
    """Response payload for pre-flight repository ingestion evaluation and budgeting."""

    model_config = ConfigDict(from_attributes=True)

    repo_name: str = Field(..., description="Canonical owner/repo slug.")
    owner: str = Field(..., description="Repository owner login.")
    name: str = Field(..., description="Repository short name.")
    canonical_url: str = Field(..., description="Canonical GitHub HTTPS URL.")
    is_eligible: bool = Field(..., description="Whether repository meets ingestion criteria.")
    rejection_reason: Optional[str] = Field(None, description="Detailed explanation if rejected.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Surface repository metadata indicators.")
    estimated_budgets: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Allocations and hard ceilings for fast_snapshot and standard_diagnostic modes.",
    )
    confidence_preview: Optional[RepositoryConfidenceReport] = Field(
        None, description="Analytical statistical sample confidence breakdown."
    )
    badges: List[str] = Field(default_factory=list, description="Descriptive indicators (e.g. 'Archived', 'Fork').")


class IngestionSubmitRequest(BaseModel):
    """Request payload for enqueuing a durable repository ingestion job."""

    repo: str = Field(..., description="Repository URL or owner/repo slug to ingest.", examples=["pallets/click"])
    mode: str = Field(
        "standard_diagnostic",
        description="Data harvesting depth: 'fast_snapshot' (<=10 requests) or 'standard_diagnostic' (<=50 requests).",
    )
    priority: int = Field(0, description="Queue priority level (higher values process earlier).")


class IngestionSubmitResponse(BaseModel):
    """Response payload returned when a durable ingestion job is created."""

    model_config = ConfigDict(from_attributes=True)

    job_id: int = Field(..., description="Unique persistent identifier for the collection job.")
    repo_name: str = Field(..., description="Target repository slug.")
    mode: str = Field(..., description="Configured collection mode.")
    status: str = Field(..., description="Initial job status (typically 'pending').")
    created_at: datetime = Field(..., description="Timestamp when job was registered in the queue.")
    message: str = Field(..., description="Human-readable submission status confirmation.")


class IngestionStatusResponse(BaseModel):
    """Comprehensive progress and execution telemetry for an ingestion job."""

    model_config = ConfigDict(from_attributes=True)

    job_id: int = Field(..., description="Unique persistent identifier for the collection job.")
    repo_name: str = Field(..., description="Target repository slug.")
    mode: str = Field(..., description="Configured collection mode.")
    status: str = Field(
        ...,
        description="Current job lifecycle status: 'pending', 'in_progress', 'completed', 'completed_partial', 'rate_limited', or 'failed'.",
    )
    worker_id: Optional[str] = Field(None, description="Identifier of the worker currently leasing the job.")
    leased_until: Optional[datetime] = Field(None, description="Lease expiration timestamp.")
    lease_version: int = Field(0, description="Fencing token incremented on each lease assignment.")
    retry_count: int = Field(0, description="Number of times this job has been retried after lease expiration.")
    max_retries: int = Field(2, description="Maximum retry threshold before failing abandoned jobs.")
    is_partial: bool = Field(False, description="Whether harvested dataset is partial/truncated.")
    partial_reason: Optional[str] = Field(None, description="Reason for truncation if partial.")
    resumes_at: Optional[datetime] = Field(None, description="Scheduled resumption timestamp if rate limited.")
    rate_limit_consumed: int = Field(0, description="Total API requests consumed by this collection run.")
    items_collected: Dict[str, int] = Field(default_factory=dict, description="Counts of harvested entities.")
    entity_status: Dict[str, str] = Field(
        default_factory=dict,
        description="Execution status per entity stream ('completed', 'truncated', 'skipped', 'failed').",
    )
    coverage_summary: Optional[Dict[str, Any]] = Field(None, description="Detailed coverage report.")
    error_message: Optional[str] = Field(None, description="Failure description if job failed.")
    started_at: datetime = Field(..., description="Timestamp when job was registered or started.")
    completed_at: Optional[datetime] = Field(None, description="Timestamp when job concluded.")
    duration_seconds: Optional[float] = Field(None, description="Total run duration in seconds.")

