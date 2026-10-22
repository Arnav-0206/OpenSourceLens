"""Raw Data Storage Manager for OpenSourceLens.

Handles atomic file persistence, directory management, and loading of
raw GitHub API responses across repositories, issues, comments, and events.
"""

import os
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union


class RawDataStorage:
    """Manages the storage hierarchy under data/raw/."""

    CATEGORIES = ("repositories", "issues", "comments", "events")

    def __init__(self, base_dir: Union[str, Path] = "data/raw"):
        """Initialize storage manager with base directory and ensure subdirectories exist.

        Args:
            base_dir: Root directory for raw data (defaults to 'data/raw').
        """
        self.base_dir = Path(base_dir).resolve()
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create category subdirectories if they do not exist."""
        for cat in self.CATEGORIES:
            (self.base_dir / cat).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sanitize_repo_identifier(repo_identifier: str) -> str:
        """Convert 'owner/repo' or URL to safe file prefix 'owner__repo'."""
        cleaned = repo_identifier.strip().rstrip("/")
        if "github.com/" in cleaned:
            cleaned = cleaned.split("github.com/")[-1]
        parts = [p.strip() for p in cleaned.split("/") if p.strip()]
        if len(parts) >= 2:
            return f"{parts[-2]}__{parts[-1]}"
        return cleaned.replace("/", "__")

    def _atomic_write_json(self, target_path: Path, data: Any) -> str:
        """Write JSON data atomically using a temporary file and atomic replace.

        Args:
            target_path: Destination file path.
            data: Python dictionary or list to serialize.

        Returns:
            Resolved string path of the written file.
        """
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(f".tmp_{os.getpid()}")

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, target_path)
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

        return str(target_path)

    def save_repository(self, repo_identifier: str, raw_data: Dict[str, Any]) -> str:
        """Persist raw repository metadata.

        Args:
            repo_identifier: 'owner/repo' string.
            raw_data: Raw JSON dictionary from GitHub repository endpoint.

        Returns:
            Saved file path string.
        """
        sanitized = self.sanitize_repo_identifier(repo_identifier)
        target = self.base_dir / "repositories" / f"{sanitized}.json"
        return self._atomic_write_json(target, raw_data)

    def save_issues(
        self,
        repo_identifier: str,
        raw_issues: List[Dict[str, Any]],
        batch_id: Optional[str] = None,
    ) -> str:
        """Persist a batch of raw issues.

        Args:
            repo_identifier: 'owner/repo' string.
            raw_issues: List of raw issue JSON dictionaries.
            batch_id: Optional batch or page identifier (defaults to UTC timestamp).

        Returns:
            Saved file path string.
        """
        sanitized = self.sanitize_repo_identifier(repo_identifier)
        if not batch_id:
            batch_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        target = self.base_dir / "issues" / f"{sanitized}__issues_{batch_id}.json"
        return self._atomic_write_json(target, raw_issues)

    def save_comments(
        self,
        repo_identifier: str,
        issue_number: int,
        raw_comments: List[Dict[str, Any]],
    ) -> str:
        """Persist raw comments for an issue.

        Args:
            repo_identifier: 'owner/repo' string.
            issue_number: Issue number.
            raw_comments: List of raw comment JSON dictionaries.

        Returns:
            Saved file path string.
        """
        sanitized = self.sanitize_repo_identifier(repo_identifier)
        target = self.base_dir / "comments" / f"{sanitized}__issue_{issue_number}__comments.json"
        return self._atomic_write_json(target, raw_comments)

    def save_events(
        self,
        repo_identifier: str,
        issue_number: int,
        raw_events: List[Dict[str, Any]],
    ) -> str:
        """Persist raw timeline events for an issue.

        Args:
            repo_identifier: 'owner/repo' string.
            issue_number: Issue number.
            raw_events: List of raw event JSON dictionaries.

        Returns:
            Saved file path string.
        """
        sanitized = self.sanitize_repo_identifier(repo_identifier)
        target = self.base_dir / "events" / f"{sanitized}__issue_{issue_number}__events.json"
        return self._atomic_write_json(target, raw_events)

    def load_raw(self, file_path: Union[str, Path]) -> Any:
        """Load and parse a raw JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            Parsed JSON content.
        """
        path = Path(file_path).resolve()
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_files(
        self,
        category: str,
        repo_identifier: Optional[str] = None,
    ) -> List[str]:
        """List stored JSON files in a category, optionally filtered by repository.

        Args:
            category: One of ('repositories', 'issues', 'comments', 'events').
            repo_identifier: Optional 'owner/repo' filter.

        Returns:
            List of matching file path strings.
        """
        cat_dir = self.base_dir / category
        if not cat_dir.exists():
            return []

        prefix = f"{self.sanitize_repo_identifier(repo_identifier)}" if repo_identifier else ""
        matches = []
        for p in sorted(cat_dir.glob("*.json")):
            if not prefix or p.name.startswith(prefix):
                matches.append(str(p))
        return matches

    def get_storage_summary(self) -> Dict[str, Any]:
        """Calculate file counts and total byte sizes across all raw data categories.

        Returns:
            Dictionary with category breakdowns and total storage metrics.
        """
        summary: Dict[str, Any] = {
            "base_dir": str(self.base_dir),
            "categories": {},
            "total_files": 0,
            "total_bytes": 0,
        }

        for cat in self.CATEGORIES:
            cat_dir = self.base_dir / cat
            files = list(cat_dir.glob("*.json"))
            cat_bytes = sum(f.stat().st_size for f in files) if files else 0

            summary["categories"][cat] = {
                "file_count": len(files),
                "bytes": cat_bytes,
                "size_kb": round(cat_bytes / 1024, 2),
            }
            summary["total_files"] += len(files)
            summary["total_bytes"] += cat_bytes

        summary["total_size_mb"] = round(summary["total_bytes"] / (1024 * 1024), 2)
        return summary
