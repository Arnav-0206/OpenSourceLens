"""Pydantic schemas for repository URL validation, eligibility, and confidence matrix."""

from __future__ import annotations

from enum import Enum
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
