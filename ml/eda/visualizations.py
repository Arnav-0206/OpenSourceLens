"""Final Exploratory Data Analysis (EDA) Visualization Suite for OpenSourceLens.

Generates high-resolution (300 DPI) publication-quality figures representing the analytical
findings from Steps 3.1 through 3.9: label taxonomies, state lifecycles, cross-repository
metrics, temporal trajectories, text characteristics, discussion comments, resolution turnaround
dynamics, correlation heatmaps, and a consolidated master executive dashboard.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


class IssueVisualizer:
    """Automated visualizer producing publication-grade charts for OpenSourceLens EDA."""

    DEFAULT_FIGURES_DIR = "ml/eda/figures"

    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path] = "data/processed/issues_master.csv",
        output_dir: Union[str, Path] = DEFAULT_FIGURES_DIR,
    ):
        """Initialize visualizer with DataFrame or path to CSV/Parquet file, and output directory.

        Args:
            data: Loaded pd.DataFrame or path to processed dataset file.
            output_dir: Directory where generated image files (.png) will be saved.

        Raises:
            FileNotFoundError: If input file path does not exist.
            ValueError: If dataset is empty.
        """
        if isinstance(data, (str, Path)):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found at: {path}")

            if path.suffix == ".parquet":
                try:
                    self.df = pd.read_parquet(path)
                except Exception:
                    csv_alt = path.with_suffix(".csv")
                    if csv_alt.exists():
                        self.df = pd.read_csv(csv_alt)
                    else:
                        raise
            elif path.suffix == ".csv":
                self.df = pd.read_csv(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}. Expected .parquet or .csv")
            self.source_path = str(path)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
            self.source_path = "DataFrame in-memory"
        else:
            raise TypeError(f"Expected DataFrame, str, or Path, got {type(data).__name__}")

        if self.df.empty:
            raise ValueError("Provided dataset is empty (0 rows).")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Apply global visualization style
        sns.set_theme(style="whitegrid", palette="deep")
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
        })

    def plot_label_distribution(self, save: bool = True) -> Path:
        """Figure 1: Canonical Label Distribution (Counts & Percentages)."""
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        try:
            label_col = "canonical_label" if "canonical_label" in self.df.columns else "primary_label"
            if label_col not in self.df.columns:
                counts = pd.Series([len(self.df)], index=["all"])
            else:
                counts = self.df[label_col].value_counts()

            total = counts.sum()
            y_pos = np.arange(len(counts))
            bars = ax.barh(y_pos, counts.values, color=sns.color_palette("mako", len(counts)), edgecolor="none", height=0.6)

            ax.set_yticks(y_pos)
            ax.set_yticklabels([str(k) for k in counts.index])
            ax.invert_yaxis()
            ax.set_xlabel("Number of Issues")
            ax.set_title("Canonical Issue Label Distribution (Class Imbalance)")

            for bar in bars:
                w = bar.get_width()
                pct = (w / total) * 100
                ax.text(w + (total * 0.01), bar.get_y() + bar.get_height() / 2, f"{int(w):,} ({pct:.1f}%)",
                        va="center", ha="left", fontsize=8.5, color="#333333")

            ax.set_xlim(0, max(counts.values) * 1.20)
            plt.tight_layout()

            target_path = self.output_dir / "fig1_label_distribution.png"
            if save:
                fig.savefig(target_path, dpi=300)
            return target_path
        finally:
            plt.close(fig)

    def plot_state_and_reasons(self, save: bool = True) -> Path:
        """Figure 2: Issue State Lifecycle and Closure Reasons."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
        try:
            # 1. State proportions
            if "state" in self.df.columns:
                state_counts = self.df["state"].value_counts()
                colors_state = ["#4C72B0", "#55A868"] if len(state_counts) == 2 else sns.color_palette("muted", len(state_counts))
                ax1.pie(state_counts.values, labels=state_counts.index, autopct="%1.1f%%",
                        startangle=140, colors=colors_state, wedgeprops=dict(width=0.45, edgecolor="white"))
                ax1.set_title("Issue Lifecycle State (Open vs. Closed)")
            else:
                ax1.text(0.5, 0.5, "No 'state' column", ha="center")

            # 2. State reason breakdown
            if "state_reason" in self.df.columns:
                reasons = self.df["state_reason"].value_counts()
                sns.barplot(x=reasons.values, y=reasons.index, ax=ax2, palette="viridis", hue=reasons.index, legend=False)
                ax2.set_title("State Reason Decomposition")
                ax2.set_xlabel("Count")
                for i, v in enumerate(reasons.values):
                    ax2.text(v + (max(reasons.values) * 0.02), i, f"{int(v):,}", va="center", fontsize=8.5)
                ax2.set_xlim(0, max(reasons.values) * 1.15)
            else:
                ax2.text(0.5, 0.5, "No 'state_reason' column", ha="center")

            plt.tight_layout()
            target_path = self.output_dir / "fig2_state_reasons.png"
            if save:
                fig.savefig(target_path, dpi=300)
            return target_path
        finally:
            plt.close(fig)

    def plot_repository_comparison(self, save: bool = True) -> Path:
        """Figure 3: Four-Panel Cross-Repository Engagement Comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
        try:
            repo_col = "repo_name"
            if repo_col not in self.df.columns:
                target_path = self.output_dir / "fig3_repo_comparison.png"
                fig.savefig(target_path, dpi=300)
                return target_path

            # Panel 1: Total Issues per Repo
            repo_counts = self.df[repo_col].value_counts()
            sns.barplot(x=repo_counts.index, y=repo_counts.values, ax=axes[0, 0], palette="crest", hue=repo_counts.index, legend=False)
            axes[0, 0].set_title("Total Harvested Issues per Repository")
            axes[0, 0].set_ylabel("Issues Count")
            axes[0, 0].tick_params(axis="x", rotation=15)

            # Panel 2: Closed Ratio (%) per Repo
            closed_ratios = []
            repos = repo_counts.index.tolist()
            for r in repos:
                sub = self.df[self.df[repo_col] == r]
                c_rate = (sub["state"] == "closed").mean() * 100 if "state" in sub.columns else 0.0
                closed_ratios.append(c_rate)
            sns.barplot(x=repos, y=closed_ratios, ax=axes[0, 1], palette="flare", hue=repos, legend=False)
            axes[0, 1].set_title("Closure Rate (%) by Repository")
            axes[0, 1].set_ylabel("Closure Rate (%)")
            axes[0, 1].set_ylim(0, 100)
            axes[0, 1].tick_params(axis="x", rotation=15)

            # Panel 3: Median Resolution Days (Closed Issues)
            closed_df = self.df[self.df["state"] == "closed"] if "state" in self.df.columns else self.df
            med_res = []
            for r in repos:
                sub = closed_df[closed_df[repo_col] == r]
                m = pd.to_numeric(sub.get("resolution_time_days"), errors="coerce").dropna().median() if "resolution_time_days" in sub.columns else 0.0
                med_res.append(m if pd.notna(m) else 0.0)
            sns.barplot(x=repos, y=med_res, ax=axes[1, 0], palette="magma", hue=repos, legend=False)
            axes[1, 0].set_title("Median Resolution Turnaround (Days)")
            axes[1, 0].set_ylabel("Median Days")
            axes[1, 0].tick_params(axis="x", rotation=15)

            # Panel 4: Median Comments per Issue
            med_comments = []
            for r in repos:
                sub = self.df[self.df[repo_col] == r]
                c = pd.to_numeric(sub.get("comments_count"), errors="coerce").dropna().median() if "comments_count" in sub.columns else 0.0
                med_comments.append(c if pd.notna(c) else 0.0)
            sns.barplot(x=repos, y=med_comments, ax=axes[1, 1], palette="viridis", hue=repos, legend=False)
            axes[1, 1].set_title("Median Comments / Issue (Discussion Volume)")
            axes[1, 1].set_ylabel("Median Comments")
            axes[1, 1].tick_params(axis="x", rotation=15)

            plt.tight_layout()
            target_path = self.output_dir / "fig3_repo_comparison.png"
            if save:
                fig.savefig(target_path, dpi=300)
            return target_path
        finally:
            plt.close(fig)

    def plot_temporal_trends(self, save: bool = True) -> Path:
        """Figure 4: Temporal Issue Creation Trajectories Over Time."""
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
        try:
            if "created_at" in self.df.columns:
                df_temp = self.df.copy()
                df_temp["created_dt"] = pd.to_datetime(df_temp["created_at"], errors="coerce", utc=True)
                df_temp = df_temp.dropna(subset=["created_dt"])
                df_temp["year_month"] = df_temp["created_dt"].dt.tz_localize(None).dt.to_period("M").astype(str)

                if "repo_name" in df_temp.columns:
                    grouped = df_temp.groupby(["year_month", "repo_name"]).size().unstack(fill_value=0)
                    grouped.plot(kind="line", ax=ax, marker="o", linewidth=2.0)
                else:
                    grouped = df_temp.groupby("year_month").size()
                    grouped.plot(kind="line", ax=ax, marker="o", color="#3470a3", linewidth=2.5)

                ax.set_title("Monthly Issue Creation Volume by Repository")
                ax.set_xlabel("Year-Month")
                ax.set_ylabel("New Issues Count")
                ax.tick_params(axis="x", rotation=45)
                ax.legend(title="Repository", frameon=True)
            else:
                ax.text(0.5, 0.5, "No 'created_at' column", ha="center")

            plt.tight_layout()
            target_path = self.output_dir / "fig4_temporal_volume.png"
            if save:
                fig.savefig(target_path, dpi=300)
            return target_path
        finally:
            plt.close(fig)

    def plot_text_distributions(self, save: bool = True) -> Path:
        """Figure 5: Title & Body Word Count Distributions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
        try:
            # 1. Title word count
            title_words = pd.to_numeric(self.df.get("clean_title_word_count"), errors="coerce").dropna() if "clean_title_word_count" in self.df.columns else pd.Series(dtype=float)
            if not title_words.empty:
                sns.histplot(title_words, kde=True, ax=ax1, color="#2b5c8f", bins=20)
                med = title_words.median()
                ax1.axvline(med, color="red", linestyle="--", linewidth=1.5, label=f"Median ({med:.1f} words)")
                ax1.set_title("Clean Title Word Count Distribution")
                ax1.set_xlabel("Word Count")
                ax1.set_ylabel("Issue Frequency")
                ax1.legend()
            else:
                ax1.text(0.5, 0.5, "No title word count data", ha="center")

            # 2. Body word count (log-scaled)
            body_words = pd.to_numeric(self.df.get("clean_body_word_count"), errors="coerce").dropna() if "clean_body_word_count" in self.df.columns else pd.Series(dtype=float)
            if not body_words.empty:
                body_pos = body_words[body_words > 0]
                sns.histplot(body_pos, kde=True, ax=ax2, color="#2d7f5e", log_scale=True, bins=25)
                med_b = body_words.median()
                ax2.axvline(med_b, color="red", linestyle="--", linewidth=1.5, label=f"Median ({med_b:.1f} words)")
                ax2.set_title("Clean Body Word Count Distribution (Log Scale)")
                ax2.set_xlabel("Word Count (Log Scale)")
                ax2.set_ylabel("Issue Frequency")
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, "No body word count data", ha="center")

            plt.tight_layout()
            target_path = self.output_dir / "fig5_text_distributions.png"
            if save:
                fig.savefig(target_path, dpi=300)
            return target_path
        finally:
            plt.close(fig)

    def plot_comment_buckets(self, save: bool = True) -> Path:
        """Figure 6: Discussion Intensity Buckets & Cumulative Percentage Curve."""
        fig, ax1 = plt.subplots(figsize=(9, 4.8), dpi=300)
        try:
            if "comments_count" in self.df.columns:
                bins = [-1, 0, 1, 3, 5, 10, np.inf]
                labels = ["0 Comments", "1 Comment", "2-3 Comments", "4-5 Comments", "6-10 Comments", "11+ Comments"]
                binned = pd.cut(self.df["comments_count"], bins=bins, labels=labels)
                counts = binned.value_counts()[labels]
                cum_pct = (counts.cumsum() / len(self.df)) * 100

                x = np.arange(len(labels))
                bars = ax1.bar(x, counts.values, color="#48639c", width=0.55, label="Issue Count")
                ax1.set_ylabel("Number of Issues", color="#48639c")
                ax1.set_xticks(x)
                ax1.set_xticklabels(labels, rotation=15)
                ax1.set_title("Issue Discussion Intensity (Comment Buckets & Cumulative Reach)")

                # Twin axis for cumulative percentage
                ax2 = ax1.twinx()
                ax2.plot(x, cum_pct.values, color="#d95f02", marker="o", linewidth=2.0, label="Cumulative %")
                ax2.set_ylabel("Cumulative Percentage (%)", color="#d95f02")
                ax2.set_ylim(0, 105)
                ax2.grid(False)

                for bar in bars:
                    h = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width() / 2, h + (max(counts.values) * 0.01), f"{int(h):,}",
                             ha="center", va="bottom", fontsize=8.5)
            else:
                ax1.text(0.5, 0.5, "No 'comments_count' column", ha="center")

            plt.tight_layout()
            target_path = self.output_dir / "fig6_comment_buckets.png"
            if save:
                fig.savefig(target_path, dpi=300)
            return target_path
        finally:
            plt.close(fig)

    def plot_resolution_dynamics(self, save: bool = True) -> Path:
        """Figure 7: Resolution Turnaround Speed Tiers and Distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
        try:
            closed = self.df[self.df["state"] == "closed"] if "state" in self.df.columns else self.df
            if "resolution_time_days" in closed.columns and not closed.empty:
                res_days = pd.to_numeric(closed["resolution_time_days"], errors="coerce").dropna()
                res_days = res_days[res_days >= 0]

                # 1. Speed tiers
                speed_bins = [-0.001, 1.0, 3.0, 7.0, 14.0, 30.0, np.inf]
                speed_labels = ["< 1 Day", "1-3 Days", "3-7 Days", "7-14 Days", "14-30 Days", "30+ Days"]
                binned_speed = pd.cut(res_days, bins=speed_bins, labels=speed_labels)
                speed_counts = binned_speed.value_counts()[speed_labels]

                sns.barplot(x=speed_counts.values, y=speed_counts.index, ax=ax1, palette="crest", hue=speed_counts.index, legend=False)
                ax1.set_title("Operational Turnaround Speed Tiers")
                ax1.set_xlabel("Closed Issues")

                # 2. Resolution days boxplot across canonical labels
                label_col = "canonical_label" if "canonical_label" in closed.columns else "primary_label"
                if label_col in closed.columns:
                    top_labels = closed[label_col].value_counts().head(4).index.tolist()
                    sub_box = closed[closed[label_col].isin(top_labels)].copy()
                    sub_box["res_days_clean"] = pd.to_numeric(sub_box["resolution_time_days"], errors="coerce")
                    sub_box = sub_box.dropna(subset=["res_days_clean"])
                    sub_box = sub_box[sub_box["res_days_clean"] >= 0]

                    sns.boxplot(x=label_col, y="res_days_clean", data=sub_box, ax=ax2, palette="Set2", hue=label_col, legend=False)
                    ax2.set_title("Resolution Duration by Canonical Label")
                    ax2.set_ylabel("Resolution Days (Capped at 95th Percentile)")
                    cap = sub_box["res_days_clean"].quantile(0.95)
                    ax2.set_ylim(-1, cap * 1.1)
                else:
                    ax2.text(0.5, 0.5, "No label column for boxplot", ha="center")
            else:
                ax1.text(0.5, 0.5, "No closed resolution data", ha="center")
                ax2.text(0.5, 0.5, "No closed resolution data", ha="center")

            plt.tight_layout()
            target_path = self.output_dir / "fig7_resolution_dynamics.png"
            if save:
                fig.savefig(target_path, dpi=300)
            return target_path
        finally:
            plt.close(fig)

    def plot_correlation_heatmap(self, save: bool = True, method: str = "spearman") -> Path:
        """Figure 8: Annotated Correlation Heatmap across Core Features."""
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        try:
            core_cols = [
                "clean_title_word_count", "clean_body_word_count", "comments_count",
                "reactions_count", "interaction_total", "labels_count", "code_block_count",
                "url_count", "resolution_time_days", "has_code_block", "is_weekend"
            ]
            avail = [c for c in core_cols if c in self.df.columns]
            if len(avail) >= 3:
                num_df = self.df[avail].apply(pd.to_numeric, errors="coerce")
                if method.lower() == "spearman":
                    corr = num_df.rank().corr(method="pearson").round(2)
                else:
                    corr = num_df.corr(method=method).round(2)

                mask = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(corr, mask=mask, cmap="vlag", vmin=-1.0, vmax=1.0, center=0.0,
                            annot=True, fmt=".2f", square=True, linewidths=0.5, ax=ax,
                            cbar_kws={"shrink": 0.8, "label": f"{method.capitalize()} Correlation"})
                ax.set_title(f"Inter-Feature {method.capitalize()} Correlation Heatmap", pad=12)
            else:
                ax.text(0.5, 0.5, "Insufficient numeric features for correlation", ha="center")

            plt.tight_layout()
            target_path = self.output_dir / "fig8_correlation_heatmap.png"
            if save:
                fig.savefig(target_path, dpi=300)
            return target_path
        finally:
            plt.close(fig)

    def plot_master_dashboard(self, save: bool = True) -> Path:
        """Figure 9: Consolidated 6-Panel Executive EDA Summary Dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300)
        try:
            # 1. Label distribution
            if "canonical_label" in self.df.columns:
                lc = self.df["canonical_label"].value_counts()
                sns.barplot(x=lc.values, y=lc.index, ax=axes[0, 0], palette="mako", hue=lc.index, legend=False)
                axes[0, 0].set_title("1. Canonical Taxonomy Balance")
                axes[0, 0].set_xlabel("Count")
            else:
                axes[0, 0].text(0.5, 0.5, "No labels", ha="center")

            # 2. State balance
            if "state" in self.df.columns:
                sc = self.df["state"].value_counts()
                axes[0, 1].pie(sc.values, labels=sc.index, autopct="%1.1f%%", colors=["#4C72B0", "#55A868"],
                               startangle=90, wedgeprops=dict(width=0.45, edgecolor="white"))
                axes[0, 1].set_title("2. Issue Lifecycle (Open/Closed)")
            else:
                axes[0, 1].text(0.5, 0.5, "No state", ha="center")

            # 3. Repositories
            if "repo_name" in self.df.columns:
                rc = self.df["repo_name"].value_counts()
                sns.barplot(x=rc.index, y=rc.values, ax=axes[0, 2], palette="crest", hue=rc.index, legend=False)
                axes[0, 2].set_title("3. Target Repositories Volume")
                axes[0, 2].tick_params(axis="x", rotation=25)
            else:
                axes[0, 2].text(0.5, 0.5, "No repo", ha="center")

            # 4. Text length KDE
            if "clean_body_word_count" in self.df.columns:
                bw = pd.to_numeric(self.df["clean_body_word_count"], errors="coerce").dropna()
                bw_pos = bw[bw > 0]
                sns.histplot(bw_pos, kde=True, ax=axes[1, 0], color="#2d7f5e", log_scale=True, bins=20)
                axes[1, 0].set_title("4. Body Word Count Distribution")
                axes[1, 0].set_xlabel("Words (Log Scale)")
            else:
                axes[1, 0].text(0.5, 0.5, "No body words", ha="center")

            # 5. Comment Buckets
            if "comments_count" in self.df.columns:
                cb_bins = [-1, 0, 1, 3, 5, 10, np.inf]
                cb_labels = ["0", "1", "2-3", "4-5", "6-10", "11+"]
                cb = pd.cut(self.df["comments_count"], bins=cb_bins, labels=cb_labels).value_counts()[cb_labels]
                sns.barplot(x=cb.index, y=cb.values, ax=axes[1, 1], palette="flare", hue=cb.index, legend=False)
                axes[1, 1].set_title("5. Discussion Intensity (Comments)")
                axes[1, 1].set_xlabel("Comments Count Bucket")
            else:
                axes[1, 1].text(0.5, 0.5, "No comments", ha="center")

            # 6. Resolution Speed Tiers
            closed = self.df[self.df["state"] == "closed"] if "state" in self.df.columns else self.df
            if "resolution_time_days" in closed.columns and not closed.empty:
                rd = pd.to_numeric(closed["resolution_time_days"], errors="coerce").dropna()
                rd_clean = rd[rd >= 0]
                sp_bins = [-0.001, 1.0, 3.0, 7.0, 14.0, 30.0, np.inf]
                sp_labels = ["<1d", "1-3d", "3-7d", "7-14d", "14-30d", "30d+"]
                sp = pd.cut(rd_clean, bins=sp_bins, labels=sp_labels).value_counts()[sp_labels]
                sns.barplot(x=sp.index, y=sp.values, ax=axes[1, 2], palette="magma", hue=sp.index, legend=False)
                axes[1, 2].set_title("6. Turnaround Speed Tiers")
                axes[1, 2].set_xlabel("Duration Tier")
            else:
                axes[1, 2].text(0.5, 0.5, "No resolution data", ha="center")

            plt.suptitle("OpenSourceLens — Master Exploratory Data Analysis Dashboard", fontsize=15, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            target_path = self.output_dir / "fig9_master_eda_dashboard.png"
            if save:
                fig.savefig(target_path, dpi=300)
            return target_path
        finally:
            plt.close(fig)

    def generate_all_visualizations(self) -> Dict[str, str]:
        """Generate all 9 figures in batch and return dictionary mapping figure names to file paths."""
        results = {
            "fig1_label_distribution": str(self.plot_label_distribution()),
            "fig2_state_reasons": str(self.plot_state_and_reasons()),
            "fig3_repo_comparison": str(self.plot_repository_comparison()),
            "fig4_temporal_volume": str(self.plot_temporal_trends()),
            "fig5_text_distributions": str(self.plot_text_distributions()),
            "fig6_comment_buckets": str(self.plot_comment_buckets()),
            "fig7_resolution_dynamics": str(self.plot_resolution_dynamics()),
            "fig8_correlation_heatmap": str(self.plot_correlation_heatmap()),
            "fig9_master_eda_dashboard": str(self.plot_master_dashboard()),
        }
        return results
