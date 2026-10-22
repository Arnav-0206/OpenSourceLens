"""OpenSourceLens Preprocessing Package."""

__all__ = [
    "RawDataLoader",
    "DatasetInspector",
    "IssueTextCleaner",
    "TimestampCleaner",
    "MissingValueHandler",
    "DataFilter",
    "LabelNormalizer",
    "IssueDatasetBuilder",
    "FeatureEngineer",
    "ProcessedDataStorage",
]


def __getattr__(name: str):
    if name == "RawDataLoader":
        from ml.preprocessing.loader import RawDataLoader
        return RawDataLoader
    elif name == "DatasetInspector":
        from ml.preprocessing.inspector import DatasetInspector
        return DatasetInspector
    elif name == "IssueTextCleaner":
        from ml.preprocessing.text_cleaner import IssueTextCleaner
        return IssueTextCleaner
    elif name == "TimestampCleaner":
        from ml.preprocessing.timestamp_cleaner import TimestampCleaner
        return TimestampCleaner
    elif name == "MissingValueHandler":
        from ml.preprocessing.missing_handler import MissingValueHandler
        return MissingValueHandler
    elif name == "DataFilter":
        from ml.preprocessing.filter import DataFilter
        return DataFilter
    elif name == "LabelNormalizer":
        from ml.preprocessing.label_normalizer import LabelNormalizer
        return LabelNormalizer
    elif name == "IssueDatasetBuilder":
        from ml.preprocessing.dataset_builder import IssueDatasetBuilder
        return IssueDatasetBuilder
    elif name == "FeatureEngineer":
        from ml.preprocessing.features import FeatureEngineer
        return FeatureEngineer
    elif name == "ProcessedDataStorage":
        from ml.preprocessing.storage_processed import ProcessedDataStorage
        return ProcessedDataStorage
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
