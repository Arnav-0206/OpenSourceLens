"""Enhanced Text Preprocessing Module for OpenSourceLens (Phase 5 NLP Enhancement).

Provides advanced text normalization specifically engineered for GitHub issues and software
engineering text: normalizes hex addresses, commit hashes, version numbers, issue references,
and file paths; strips issue template markdown boilerplate; condenses repetitive stack traces;
preserves crucial negation indicators; and filters conversational GitHub noise.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Set
import re

import pandas as pd


class EnhancedIssueTextPreprocessor:
    """Advanced NLP preprocessor tailored for GitHub issue titles, descriptions, and code artifacts."""

    # Software and technical entity regular expressions
    HEX_ADDR_RE = re.compile(r"\b0x[0-9a-fA-F]{4,16}\b")
    COMMIT_HASH_RE = re.compile(r"\b[0-9a-f]{7,40}\b")
    VERSION_RE = re.compile(r"\bv?\d+(\.\d+)+(-[a-zA-Z0-9.]+)?\b")
    ISSUE_REF_RE = re.compile(r"(?<!\w)#\d+\b")
    FILE_PATH_RE = re.compile(r"\b(?:[a-zA-Z0-9_\-./]+)?\.(py|ts|js|go|cpp|c|h|json|yaml|yml|md|rs|java|sh)\b", re.IGNORECASE)

    # Common GitHub issue form headers and template boilerplate
    TEMPLATE_HEADERS_RE = re.compile(
        r"(?mi)^\s*#{1,6}\s*(steps\s+to\s+reproduce|expected\s+behavior|actual\s+behavior|expected\s+result|"
        r"actual\s+result|environment(?:\s+info)?|system\s+info|versions?|description|"
        r"motivation(?:\s+and\s+context)?|additional\s+context|how\s+to\s+reproduce|"
        r"reproduction\s+steps|error\s+log|stack\s+trace|requirements|context)\s*:?\s*$",
    )

    # Stack trace and runtime log patterns
    TRACEBACK_LINE_RE = re.compile(r'^\s*File\s+"[^"]+",\s+line\s+\d+.*$', re.MULTILINE)
    JAVA_STACK_LINE_RE = re.compile(r"^\s*at\s+[a-zA-Z0-9_$.]+\([^)]+\)\s*$", re.MULTILINE)
    GO_STACK_LINE_RE = re.compile(r"^\s*(?:goroutine\s+\d+|created\s+by\s+.*|\t[a-zA-Z0-9_./]+\.go:\d+.*)$", re.MULTILINE)

    # Common exception / fatal error signatures to extract and preserve
    EXCEPTION_TYPE_RE = re.compile(
        r"\b([A-Z][a-zA-Z0-9]*(?:Exception|Error|Fault|Panic|Failure)|SIGSEGV|SIGABRT|SIGILL|SIGBUS|CUDA\s+error)\b"
    )

    # Conversational filler and triage noise
    GITHUB_CHATTER_RE = re.compile(
        r"\b(thanks\s+in\s+advance|thank\s+you\s+very\s+much|thanks|thank\s+you|please\s+help|"
        r"any\s+updates|any\s+update|ping|bump|cheers|regards|hey\s+guys|hey\s+team|"
        r"hello\s+everyone|hi\s+all|anyone\s+know|anybody\s+know)\b",
        re.IGNORECASE,
    )

    # Negation words that must NOT be treated as stop words
    NEGATION_WORDS: Set[str] = {
        "not", "no", "never", "cannot", "cant", "wont", "doesnt", "isnt",
        "didnt", "shouldnt", "couldnt", "wouldnt", "failed", "failure",
        "without", "none", "neither", "nor", "nothing"
    }

    # Custom domain stop words (noise terms with zero category discriminative power)
    DOMAIN_STOP_WORDS: Set[str] = {
        "hi", "hello", "hey", "please", "thanks", "thank", "advance", "regards",
        "cheers", "anyone", "somebody", "someone", "etc", "also", "already",
        "im", "ive", "id", "maybe", "probably", "would", "could", "should"
    }

    def __init__(
        self,
        strip_template_headers: bool = True,
        normalize_code_entities: bool = True,
        condense_stack_traces: bool = True,
        preserve_negations: bool = True,
        filter_github_noise: bool = True,
    ):
        """Initialize the enhanced preprocessor.

        Args:
            strip_template_headers: Strip common GitHub issue markdown headers (default: True).
            normalize_code_entities: Replace hex addresses, SHAs, versions, and paths with tokens (default: True).
            condense_stack_traces: Condense verbose stack trace frames to exception signatures (default: True).
            preserve_negations: Retain negation terms (default: True).
            filter_github_noise: Remove conversational filler words (default: True).
        """
        self.strip_template_headers = strip_template_headers
        self.normalize_code_entities = normalize_code_entities
        self.condense_stack_traces = condense_stack_traces
        self.preserve_negations = preserve_negations
        self.filter_github_noise = filter_github_noise

    def clean_text(self, text: Optional[str], is_title: bool = False) -> str:
        """Execute full enhanced cleaning pipeline on raw issue text.

        Args:
            text: Raw input text string.
            is_title: Whether the input is an issue title (preserves shorter phrasing) (default: False).

        Returns:
            Cleaned and normalized text string.
        """
        if not text or not isinstance(text, str):
            return ""

        s = text

        # 1. Strip Template Headers (descriptions only, titles don't have them)
        if self.strip_template_headers and not is_title:
            s = self.TEMPLATE_HEADERS_RE.sub(" ", s)

        # 2. Condense Stack Traces and Logs
        if self.condense_stack_traces and not is_title:
            # Extract and preserve all exception types before stripping frame lines
            exceptions = self.EXCEPTION_TYPE_RE.findall(s)
            s = self.TRACEBACK_LINE_RE.sub(" ", s)
            s = self.JAVA_STACK_LINE_RE.sub(" ", s)
            s = self.GO_STACK_LINE_RE.sub(" ", s)

            # If exceptions were found, append them explicitly to preserve failure signatures
            if exceptions:
                unique_exc = " ".join(sorted(set(exceptions)))
                s += f" exception_tag {unique_exc}"

        # 3. Filter Conversational Chatter
        if self.filter_github_noise:
            s = self.GITHUB_CHATTER_RE.sub(" ", s)

        # 4. Normalize Code & Technical Entities
        if self.normalize_code_entities:
            s = self.HEX_ADDR_RE.sub(" hex_addr ", s)
            s = self.COMMIT_HASH_RE.sub(" commit_hash ", s)
            s = self.VERSION_RE.sub(" version_num ", s)
            s = self.ISSUE_REF_RE.sub(" issue_ref ", s)
            s = self.FILE_PATH_RE.sub(" file_path ", s)

        # 5. Lowercase and Basic Punctuation Cleaning
        s = s.lower()
        # Keep letters, numbers, underscores, and hyphens
        s = re.sub(r"[^\w\s-]", " ", s)
        # Collapse multiple dashes or underscores
        s = re.sub(r"[-_]{2,}", " ", s)

        # 6. Word-level Filtering (Stop-words with negation preservation)
        tokens = s.split()
        filtered_tokens = []
        for t in tokens:
            if len(t) <= 1 and t not in ("a", "i"):
                continue

            # Remove custom domain noise words
            if self.filter_github_noise and t in self.DOMAIN_STOP_WORDS:
                continue

            # Check negations
            if self.preserve_negations and t in self.NEGATION_WORDS:
                filtered_tokens.append(t)
                continue

            filtered_tokens.append(t)

        return " ".join(filtered_tokens).strip()

    def preprocess_issue(self, title: Optional[str], body: Optional[str]) -> Dict[str, str]:
        """Preprocess title and body, returning clean individual and combined strings.

        Args:
            title: Raw or basic clean issue title.
            body: Raw or basic clean issue body.

        Returns:
            Dictionary containing 'clean_title', 'clean_body', and 'combined_text'.
        """
        c_title = self.clean_text(title, is_title=True)
        c_body = self.clean_text(body, is_title=False)
        combined = f"{c_title} {c_body}".strip()

        return {
            "clean_title": c_title,
            "clean_body": c_body,
            "combined_text": combined,
        }

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        title_col: str = "clean_title",
        body_col: str = "clean_body",
        prefix: str = "enhanced_",
    ) -> pd.DataFrame:
        """Apply enhanced preprocessing across a pandas DataFrame.

        Args:
            df: Input DataFrame.
            title_col: Name of title column (default: 'clean_title').
            body_col: Name of body column (default: 'clean_body').
            prefix: Column name prefix for enhanced outputs (default: 'enhanced_').

        Returns:
            DataFrame augmented with new enhanced text columns and word count metrics.
        """
        result_df = df.copy()

        # Handle column fallbacks
        t_col = title_col if title_col in result_df.columns else ("title" if "title" in result_df.columns else None)
        b_col = body_col if body_col in result_df.columns else ("body" if "body" in result_df.columns else None)

        titles = result_df[t_col].fillna("").astype(str) if t_col else pd.Series([""] * len(result_df))
        bodies = result_df[b_col].fillna("").astype(str) if b_col else pd.Series([""] * len(result_df))

        clean_titles = titles.apply(lambda t: self.clean_text(t, is_title=True))
        clean_bodies = bodies.apply(lambda b: self.clean_text(b, is_title=False))
        combined = (clean_titles + " " + clean_bodies).str.strip()

        result_df[f"{prefix}title"] = clean_titles
        result_df[f"{prefix}body"] = clean_bodies
        result_df[f"{prefix}combined_text"] = combined
        result_df[f"{prefix}title_word_count"] = clean_titles.apply(lambda s: len(s.split()))
        result_df[f"{prefix}body_word_count"] = clean_bodies.apply(lambda s: len(s.split()))
        result_df[f"{prefix}total_word_count"] = (
            result_df[f"{prefix}title_word_count"] + result_df[f"{prefix}body_word_count"]
        )

        return result_df
