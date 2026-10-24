"""OpenSourceLens Exploratory Data Analysis (EDA) Package.

Provides analytical engines and reporting tools for dataset overview, state analysis,
label distributions, temporal trends, repository comparisons, text characteristics,
comment and discussion dynamics, resolution duration, feature correlation, publication visualizations,
and comprehensive EDA report/notebook generation.
"""

from ml.eda.overview import DatasetOverviewAnalyzer
from ml.eda.state_analysis import IssueStateAnalyzer
from ml.eda.label_analysis import IssueLabelAnalyzer
from ml.eda.temporal_analysis import IssueTemporalAnalyzer
from ml.eda.repo_comparison import RepositoryComparisonAnalyzer
from ml.eda.text_analysis import IssueTextAnalyzer
from ml.eda.comment_analysis import IssueCommentAnalyzer
from ml.eda.resolution_analysis import IssueResolutionAnalyzer
from ml.eda.correlation_analysis import IssueCorrelationAnalyzer
from ml.eda.visualizations import IssueVisualizer
from ml.eda.report import EDAFindingsReporter

__all__ = [
    "DatasetOverviewAnalyzer",
    "IssueStateAnalyzer",
    "IssueLabelAnalyzer",
    "IssueTemporalAnalyzer",
    "RepositoryComparisonAnalyzer",
    "IssueTextAnalyzer",
    "IssueCommentAnalyzer",
    "IssueResolutionAnalyzer",
    "IssueCorrelationAnalyzer",
    "IssueVisualizer",
    "EDAFindingsReporter",
]
