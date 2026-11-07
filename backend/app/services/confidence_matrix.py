"""Decoupled Analytical Confidence Matrix.

Evaluates per-domain statistical sample sufficiency independently from basic
repository ingestion eligibility.

Note:
    Confidence levels strictly evaluate whether harvested data volume meets
    statistical sample size thresholds for meaningful analysis. They do NOT
    represent machine learning models (e.g., DistilBERT, Random Forest) as
    operational or loaded; runtime model availability and artifact health must
    be verified independently via model manager and service health checks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from backend.app.schemas.ingestion import (
    ConfidenceLevel,
    DomainConfidence,
    RepositoryConfidenceReport,
)


class ConfidenceMatrixEvaluator:
    """Evaluates statistical sample size sufficiency for each analytical domain."""

    # Domain keys
    DEVELOPMENT_ACTIVITY = "development_activity"
    CONTRIBUTOR_ECOSYSTEM = "contributor_ecosystem"
    PULL_REQUEST_LIFECYCLE = "pull_request_lifecycle"
    ISSUE_INTELLIGENCE = "issue_intelligence"
    RESOLUTION_VELOCITY = "resolution_velocity"

    @classmethod
    def evaluate_development_activity(cls, commits_count: int) -> DomainConfidence:
        """Evaluate commit history depth for rolling velocity and time-series decomposition."""
        count = max(0, commits_count)
        thresholds = {"insufficient_under": 3, "limited_under": 10, "confident_at_or_above": 10}

        if count < 3:
            return DomainConfidence(
                domain=cls.DEVELOPMENT_ACTIVITY,
                domain_name="Development Velocity & Time-Series",
                level=ConfidenceLevel.INSUFFICIENT_DATA,
                sample_count=count,
                metric_name="commits",
                thresholds=thresholds,
                is_sufficient=False,
                message="Requires at least 3 commits for rolling velocity and trend analysis.",
                warning="Unavailable (Requires ≥ 3 commits)",
            )
        elif count < 10:
            return DomainConfidence(
                domain=cls.DEVELOPMENT_ACTIVITY,
                domain_name="Development Velocity & Time-Series",
                level=ConfidenceLevel.LIMITED_SAMPLE,
                sample_count=count,
                metric_name="commits",
                thresholds=thresholds,
                is_sufficient=True,
                message="Commit sample supports basic activity tracking with rolling averages.",
                warning="Short history; rolling averages have high variance",
            )
        else:
            return DomainConfidence(
                domain=cls.DEVELOPMENT_ACTIVITY,
                domain_name="Development Velocity & Time-Series",
                level=ConfidenceLevel.CONFIDENT,
                sample_count=count,
                metric_name="commits",
                thresholds=thresholds,
                is_sufficient=True,
                message="Sufficient commit depth for time-series decomposition, velocity trends, and anomaly detection.",
                warning=None,
            )

    @classmethod
    def evaluate_contributor_ecosystem(cls, contributors_count: int) -> DomainConfidence:
        """Evaluate contributor sample size for Gini inequality, Lorenz curves, and retention cohorts."""
        count = max(0, contributors_count)
        thresholds = {"insufficient_under": 2, "limited_under": 5, "confident_at_or_above": 5}

        if count < 2:
            return DomainConfidence(
                domain=cls.CONTRIBUTOR_ECOSYSTEM,
                domain_name="Contributor Ecosystem & Inequality",
                level=ConfidenceLevel.INSUFFICIENT_DATA,
                sample_count=count,
                metric_name="contributors",
                thresholds=thresholds,
                is_sufficient=False,
                message="Gini inequality is mathematically degenerate for single-author repositories (G = 0).",
                warning="Low contributor count: single-author repository",
            )
        elif count < 5:
            return DomainConfidence(
                domain=cls.CONTRIBUTOR_ECOSYSTEM,
                domain_name="Contributor Ecosystem & Inequality",
                level=ConfidenceLevel.LIMITED_SAMPLE,
                sample_count=count,
                metric_name="contributors",
                thresholds=thresholds,
                is_sufficient=True,
                message="Contributor count supports basic cohort tracking and participation overview.",
                warning="Low contributor count (n < 5)",
            )
        else:
            return DomainConfidence(
                domain=cls.CONTRIBUTOR_ECOSYSTEM,
                domain_name="Contributor Ecosystem & Inequality",
                level=ConfidenceLevel.CONFIDENT,
                sample_count=count,
                metric_name="contributors",
                thresholds=thresholds,
                is_sufficient=True,
                message="Sufficient contributor pool for Gini inequality, Lorenz curves, retention cohorts, and bus factor modeling.",
                warning=None,
            )

    @classmethod
    def evaluate_pr_lifecycle(cls, prs_count: int) -> DomainConfidence:
        """Evaluate pull request volume for review turnaround percentiles and survival curves."""
        count = max(0, prs_count)
        thresholds = {"insufficient_under": 1, "limited_under": 10, "confident_at_or_above": 10}

        if count == 0:
            return DomainConfidence(
                domain=cls.PULL_REQUEST_LIFECYCLE,
                domain_name="Pull Request Lifecycle & Bottlenecks",
                level=ConfidenceLevel.INSUFFICIENT_DATA,
                sample_count=count,
                metric_name="pull_requests",
                thresholds=thresholds,
                is_sufficient=False,
                message="No pull requests recorded in repository.",
                warning="No pull requests recorded",
            )
        elif count < 10:
            return DomainConfidence(
                domain=cls.PULL_REQUEST_LIFECYCLE,
                domain_name="Pull Request Lifecycle & Bottlenecks",
                level=ConfidenceLevel.LIMITED_SAMPLE,
                sample_count=count,
                metric_name="pull_requests",
                thresholds=thresholds,
                is_sufficient=True,
                message="Sample supports individual PR turnaround inspection.",
                warning="Small PR sample (n < 10); survival curves and percentiles have wide confidence intervals",
            )
        else:
            return DomainConfidence(
                domain=cls.PULL_REQUEST_LIFECYCLE,
                domain_name="Pull Request Lifecycle & Bottlenecks",
                level=ConfidenceLevel.CONFIDENT,
                sample_count=count,
                metric_name="pull_requests",
                thresholds=thresholds,
                is_sufficient=True,
                message="Sufficient PR volume for Kaplan-Meier survival curves, review turnaround percentiles, and merge latency distributions.",
                warning=None,
            )

    @classmethod
    def evaluate_issue_intelligence(cls, issues_count: int) -> DomainConfidence:
        """Evaluate issue count for NLP classification and category distribution analysis."""
        count = max(0, issues_count)
        thresholds = {"insufficient_under": 1, "limited_under": 10, "confident_at_or_above": 10}

        if count == 0:
            return DomainConfidence(
                domain=cls.ISSUE_INTELLIGENCE,
                domain_name="Issue NLP Categorization",
                level=ConfidenceLevel.INSUFFICIENT_DATA,
                sample_count=count,
                metric_name="issues",
                thresholds=thresholds,
                is_sufficient=False,
                message="No issues recorded or issues disabled by maintainers.",
                warning="No issues recorded (or issues disabled by maintainers)",
            )
        elif count < 10:
            return DomainConfidence(
                domain=cls.ISSUE_INTELLIGENCE,
                domain_name="Issue NLP Categorization",
                level=ConfidenceLevel.LIMITED_SAMPLE,
                sample_count=count,
                metric_name="issues",
                thresholds=thresholds,
                is_sufficient=True,
                message="Sample volume supports individual issue classification; aggregate distributions remain subject to small-sample variance.",
                warning="Limited issue history (n < 10); aggregate distributions may not represent long-term trends",
            )
        else:
            return DomainConfidence(
                domain=cls.ISSUE_INTELLIGENCE,
                domain_name="Issue NLP Categorization",
                level=ConfidenceLevel.CONFIDENT,
                sample_count=count,
                metric_name="issues",
                thresholds=thresholds,
                is_sufficient=True,
                message="Sufficient issue volume for aggregate category distribution, multi-class validation, and backlog aging analysis.",
                warning=None,
            )

    @classmethod
    def evaluate_resolution_velocity(cls, closed_issues_count: int) -> DomainConfidence:
        """Evaluate closed issue history for empirical SLAs and resolution velocity modeling."""
        count = max(0, closed_issues_count)
        thresholds = {"insufficient_under": 5, "limited_under": 20, "confident_at_or_above": 20}

        if count < 5:
            return DomainConfidence(
                domain=cls.RESOLUTION_VELOCITY,
                domain_name="Resolution SLA Prediction",
                level=ConfidenceLevel.INSUFFICIENT_DATA,
                sample_count=count,
                metric_name="closed_issues",
                thresholds=thresholds,
                is_sufficient=False,
                message="Requires at least 5 closed issues for empirical SLA and resolution velocity prediction.",
                warning="Insufficient closed issue history (Requires ≥ 5 closed issues)",
            )
        elif count < 20:
            return DomainConfidence(
                domain=cls.RESOLUTION_VELOCITY,
                domain_name="Resolution SLA Prediction",
                level=ConfidenceLevel.LIMITED_SAMPLE,
                sample_count=count,
                metric_name="closed_issues",
                thresholds=thresholds,
                is_sufficient=True,
                message="Empirical SLA percentiles available with wide confidence bounds.",
                warning="Resolution velocity predictions have wide confidence bounds (5 <= n < 20 closed issues)",
            )
        else:
            return DomainConfidence(
                domain=cls.RESOLUTION_VELOCITY,
                domain_name="Resolution SLA Prediction",
                level=ConfidenceLevel.CONFIDENT,
                sample_count=count,
                metric_name="closed_issues",
                thresholds=thresholds,
                is_sufficient=True,
                message="Sufficient closed issue volume for empirical SLA quantile forecasting and resolution velocity modeling.",
                warning=None,
            )

    @classmethod
    def evaluate(
        cls,
        commits_count: int = 0,
        contributors_count: int = 0,
        prs_count: int = 0,
        issues_count: int = 0,
        closed_issues_count: int = 0,
    ) -> RepositoryConfidenceReport:
        """Evaluate the entire repository across all 5 analytical domains.

        Args:
            commits_count: Number of observed commits.
            contributors_count: Number of unique contributors.
            prs_count: Number of pull requests.
            issues_count: Number of issues.
            closed_issues_count: Number of closed issues.

        Returns:
            RepositoryConfidenceReport: Multi-domain assessment with composite score.
        """
        domains: Dict[str, DomainConfidence] = {
            cls.DEVELOPMENT_ACTIVITY: cls.evaluate_development_activity(commits_count),
            cls.CONTRIBUTOR_ECOSYSTEM: cls.evaluate_contributor_ecosystem(contributors_count),
            cls.PULL_REQUEST_LIFECYCLE: cls.evaluate_pr_lifecycle(prs_count),
            cls.ISSUE_INTELLIGENCE: cls.evaluate_issue_intelligence(issues_count),
            cls.RESOLUTION_VELOCITY: cls.evaluate_resolution_velocity(closed_issues_count),
        }

        # Weighting for composite score: CONFIDENT = 1.0, LIMITED_SAMPLE = 0.5, INSUFFICIENT_DATA = 0.0
        score_weights = {
            ConfidenceLevel.CONFIDENT: 1.0,
            ConfidenceLevel.LIMITED_SAMPLE: 0.5,
            ConfidenceLevel.INSUFFICIENT_DATA: 0.0,
        }

        scores = [score_weights[d.level] for d in domains.values()]
        composite_score = round(sum(scores) / len(scores), 2)
        sufficient_count = sum(1 for d in domains.values() if d.is_sufficient)
        total_count = len(domains)

        # Assign overall confidence level
        if composite_score >= 0.8:
            overall_confidence = ConfidenceLevel.CONFIDENT
            summary = (
                f"Repository has comprehensive analytical data coverage ({sufficient_count}/{total_count} "
                "domains sufficient). Full statistical and ML diagnostics are enabled."
            )
        elif composite_score >= 0.4:
            overall_confidence = ConfidenceLevel.LIMITED_SAMPLE
            summary = (
                f"Repository has moderate data coverage ({sufficient_count}/{total_count} "
                "domains sufficient). Certain domain models operate with widened confidence intervals."
            )
        else:
            overall_confidence = ConfidenceLevel.INSUFFICIENT_DATA
            summary = (
                f"Repository has sparse historical activity ({sufficient_count}/{total_count} "
                "domains sufficient). Primary focus is limited to raw event ingestion and basic cataloging."
            )

        return RepositoryConfidenceReport(
            overall_confidence=overall_confidence,
            overall_score=composite_score,
            domains=domains,
            summary=summary,
            sufficient_domains_count=sufficient_count,
            total_domains_count=total_count,
        )
