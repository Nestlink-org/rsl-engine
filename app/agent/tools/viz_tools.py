"""Visualization tools for the RSL chat agent.

Each tool queries the DB, builds a plot with matplotlib/seaborn,
saves it to static/visualizations/, and returns the URL path.

The chat agent calls these tools when users ask for charts or visual analysis.
"""

import json
import logging
import os
import time
import uuid
from datetime import date, timedelta
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

VIZ_DIR = "static/visualizations"
VIZ_URL_PREFIX = "/static/visualizations"


def _save_fig(fig, prefix: str) -> str:
    """Save matplotlib figure, return URL path."""
    os.makedirs(VIZ_DIR, exist_ok=True)
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(VIZ_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return f"{VIZ_URL_PREFIX}/{filename}"


def _setup_style():
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#f8f9fa"})


# ── Tool 1: Facility anomaly score bar chart ──────────────────────────────────

@tool
async def plot_facility_anomaly_scores(top_n: int = 10, weeks: int = 12) -> str:
    """
    Plot a bar chart of the top N facilities ranked by average anomaly score.
    Returns the URL of the saved chart image.
    """
    from sqlalchemy import func, select
    from app.db.sessions import async_session
    from app.models.facility_metric import FacilityWeeklyMetric
    import matplotlib.pyplot as plt
    import numpy as np

    _setup_style()
    cutoff = date.today() - timedelta(weeks=weeks)

    async with async_session() as session:
        result = await session.execute(
            select(
                FacilityWeeklyMetric.facility_id,
                func.avg(FacilityWeeklyMetric.avg_anomaly_score).label("avg_score"),
                func.sum(FacilityWeeklyMetric.flagged_claims).label("flagged"),
                func.sum(FacilityWeeklyMetric.claim_volume).label("volume"),
            )
            .where(FacilityWeeklyMetric.week_start_date >= cutoff)
            .group_by(FacilityWeeklyMetric.facility_id)
            .order_by(func.avg(FacilityWeeklyMetric.avg_anomaly_score).desc())
            .limit(top_n)
        )
        rows = result.all()

    if not rows:
        return "No facility data available to plot."

    facilities = [r[0] for r in rows]
    scores = [float(r[1] or 0) for r in rows]
    flagged = [int(r[2] or 0) for r in rows]

    colors = ["#e74c3c" if s > 0.7 else "#f39c12" if s > 0.4 else "#2ecc71" for s in scores]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(facilities[::-1], scores[::-1], color=colors[::-1], edgecolor="white", height=0.6)
    ax.set_xlabel("Average Anomaly Score", fontsize=12)
    ax.set_title(f"Top {top_n} Facilities by Anomaly Score (Last {weeks} Weeks)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.axvline(0.5, color="#e74c3c", linestyle="--", alpha=0.5, label="Anomaly threshold (0.5)")
    ax.axvline(0.7, color="#c0392b", linestyle=":", alpha=0.5, label="High risk (0.7)")

    for bar, score, flag in zip(bars, scores[::-1], flagged[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f} ({flag} flagged)", va="center", fontsize=9)

    ax.legend(fontsize=9)
    plt.tight_layout()
    url = _save_fig(fig, "facility_anomaly")
    return f"Facility anomaly chart saved. View at: {url}"


# ── Tool 2: Patient trajectory anomaly heatmap ────────────────────────────────

@tool
async def plot_patient_trajectory(patient_id: str) -> str:
    """
    Plot a heatmap of a patient's visit trajectory showing per-visit reconstruction errors
    across CBC features. Returns the URL of the saved chart image.
    """
    from sqlalchemy import select
    from app.db.sessions import async_session
    from app.models.patient_trajectory import PatientTrajectory
    import matplotlib.pyplot as plt
    import numpy as np

    _setup_style()

    async with async_session() as session:
        result = await session.execute(
            select(PatientTrajectory).where(PatientTrajectory.patient_id == patient_id)
        )
        traj = result.scalar_one_or_none()

    if not traj:
        return f"No trajectory data found for patient {patient_id}."

    visits = json.loads(traj.visit_sequence or "[]")
    per_visit_errors = json.loads(traj.per_visit_errors or "[]")

    if not visits:
        return f"Patient {patient_id} has no visit history."

    features = ["age", "sex_encoded", "HGB", "HCT", "MCV", "MCHC", "NEU", "LYM", "EOS", "BAS", "MON", "PLT", "length_of_stay"]
    n_visits = len(visits)
    data = []
    for v in visits:
        row = [float(v.get(f, 0)) for f in features]
        data.append(row)

    import numpy as np
    data_arr = np.array(data)
    # Normalize each feature to 0-1 for heatmap
    col_min = data_arr.min(axis=0)
    col_max = data_arr.max(axis=0)
    col_range = np.where(col_max - col_min == 0, 1, col_max - col_min)
    data_norm = (data_arr - col_min) / col_range

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: feature heatmap
    import seaborn as sns
    sns.heatmap(
        data_norm.T,
        ax=axes[0],
        xticklabels=[f"Visit {i+1}" for i in range(n_visits)],
        yticklabels=features,
        cmap="RdYlGn_r",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Normalized Value"},
    )
    axes[0].set_title(f"Patient {patient_id} — CBC Feature Heatmap", fontweight="bold")

    # Right: per-visit reconstruction error
    if per_visit_errors:
        visit_labels = [f"Visit {i+1}" for i in range(len(per_visit_errors))]
        bar_colors = ["#e74c3c" if e > 0.3 else "#f39c12" if e > 0.1 else "#2ecc71" for e in per_visit_errors]
        axes[1].bar(visit_labels, per_visit_errors, color=bar_colors, edgecolor="white")
        axes[1].axhline(0.295, color="#e74c3c", linestyle="--", label="Anomaly threshold")
        axes[1].set_ylabel("Reconstruction Error")
        axes[1].set_title("Per-Visit Reconstruction Error", fontweight="bold")
        axes[1].legend()
        score = traj.trajectory_anomaly_score or 0
        status = "⚠ ANOMALOUS" if traj.is_trajectory_anomaly else "✓ NORMAL"
        axes[1].set_xlabel(f"Trajectory Score: {score:.3f} — {status}")
    else:
        axes[1].text(0.5, 0.5, "No reconstruction errors available", ha="center", va="center")

    plt.tight_layout()
    url = _save_fig(fig, f"patient_traj_{patient_id[:8]}")
    return f"Patient trajectory chart for {patient_id} saved. View at: {url}"


# ── Tool 3: Disease distribution pie/bar ─────────────────────────────────────

@tool
async def plot_disease_distribution(weeks: int = 12) -> str:
    """
    Plot disease category and diagnosis distribution from Model 2 predictions.
    Returns the URL of the saved chart image.
    """
    from sqlalchemy import select
    from app.db.sessions import async_session
    from app.models.fraud_flag import FraudFlag
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter

    _setup_style()

    async with async_session() as session:
        result = await session.execute(
            select(
                FraudFlag.predicted_category,
                FraudFlag.predicted_diagnosis,
                FraudFlag.is_anomaly,
                FraudFlag.diagnosis_confidence,
            ).where(
                FraudFlag.model_id == 2,
                FraudFlag.predicted_diagnosis.isnot(None),
            )
        )
        rows = result.all()

    if not rows:
        return "No disease classification data available."

    categories = Counter(r[0] for r in rows if r[0])
    diagnoses = Counter(r[1] for r in rows if r[1])
    flagged_by_diag = Counter(r[1] for r in rows if r[1] and r[2])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Category pie
    cat_labels = list(categories.keys())
    cat_vals = list(categories.values())
    cat_colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    axes[0].pie(cat_vals, labels=cat_labels, autopct="%1.1f%%",
                colors=cat_colors[:len(cat_labels)], startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[0].set_title("Disease Category Distribution", fontweight="bold")

    # Diagnosis bar
    diag_labels = list(diagnoses.keys())
    diag_vals = list(diagnoses.values())
    diag_flagged = [flagged_by_diag.get(d, 0) for d in diag_labels]
    x = range(len(diag_labels))
    axes[1].bar(x, diag_vals, label="Total", color="#3498db", alpha=0.7)
    axes[1].bar(x, diag_flagged, label="Flagged", color="#e74c3c", alpha=0.9)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(diag_labels, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("Claim Count")
    axes[1].set_title("Diagnosis: Total vs Flagged", fontweight="bold")
    axes[1].legend()

    # Flag rate by diagnosis
    flag_rates = [flagged_by_diag.get(d, 0) / max(diagnoses[d], 1) for d in diag_labels]
    bar_colors = ["#e74c3c" if r > 0.5 else "#f39c12" if r > 0.2 else "#2ecc71" for r in flag_rates]
    axes[2].bar(diag_labels, flag_rates, color=bar_colors, edgecolor="white")
    axes[2].set_xticklabels(diag_labels, rotation=30, ha="right", fontsize=9)
    axes[2].set_ylabel("Flag Rate")
    axes[2].set_ylim(0, 1.1)
    axes[2].axhline(0.5, color="#e74c3c", linestyle="--", alpha=0.6, label="50% threshold")
    axes[2].set_title("Flag Rate by Diagnosis", fontweight="bold")
    axes[2].legend()

    plt.suptitle("Disease Analysis — RSL Fraud Detection", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    url = _save_fig(fig, "disease_dist")
    return f"Disease distribution chart saved. View at: {url}"


# ── Tool 4: Anomaly score trend line ─────────────────────────────────────────

@tool
async def plot_anomaly_trend(period: str = "weekly", weeks: int = 12) -> str:
    """
    Plot anomaly score trend over time (weekly/monthly).
    Returns the URL of the saved chart image.
    """
    from sqlalchemy import func, select
    from app.db.sessions import async_session
    from app.models.facility_metric import FacilityWeeklyMetric
    import matplotlib.pyplot as plt
    import numpy as np

    _setup_style()
    cutoff = date.today() - timedelta(weeks=weeks)

    async with async_session() as session:
        result = await session.execute(
            select(
                FacilityWeeklyMetric.week_start_date,
                func.sum(FacilityWeeklyMetric.claim_volume).label("vol"),
                func.sum(FacilityWeeklyMetric.flagged_claims).label("flagged"),
                func.avg(FacilityWeeklyMetric.avg_anomaly_score).label("avg_score"),
            )
            .where(FacilityWeeklyMetric.week_start_date >= cutoff)
            .group_by(FacilityWeeklyMetric.week_start_date)
            .order_by(FacilityWeeklyMetric.week_start_date)
        )
        rows = result.all()

    if not rows:
        return "No trend data available."

    dates = [str(r[0]) for r in rows]
    volumes = [int(r[1] or 0) for r in rows]
    flagged = [int(r[2] or 0) for r in rows]
    scores = [float(r[3] or 0) for r in rows]
    flag_rates = [f / max(v, 1) for f, v in zip(flagged, volumes)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: claim volume + flagged
    axes[0].fill_between(dates, volumes, alpha=0.3, color="#3498db", label="Total Claims")
    axes[0].plot(dates, volumes, color="#3498db", linewidth=2)
    axes[0].fill_between(dates, flagged, alpha=0.5, color="#e74c3c", label="Flagged Claims")
    axes[0].plot(dates, flagged, color="#e74c3c", linewidth=2)
    axes[0].set_ylabel("Claim Count")
    axes[0].set_title("Claim Volume & Fraud Flags Over Time", fontweight="bold")
    axes[0].legend()

    # Bottom: anomaly score + flag rate
    ax2 = axes[1].twinx()
    axes[1].plot(dates, scores, color="#9b59b6", linewidth=2.5, marker="o", markersize=4, label="Avg Anomaly Score")
    axes[1].axhline(0.5, color="#e74c3c", linestyle="--", alpha=0.5)
    axes[1].set_ylabel("Avg Anomaly Score", color="#9b59b6")
    axes[1].set_ylim(0, 1.1)
    ax2.plot(dates, flag_rates, color="#f39c12", linewidth=2, linestyle="--", marker="s", markersize=4, label="Flag Rate")
    ax2.set_ylabel("Flag Rate", color="#f39c12")
    ax2.set_ylim(0, 1.1)
    axes[1].set_title("Anomaly Score & Flag Rate Trend", fontweight="bold")

    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    tick_step = max(1, len(dates) // 8)
    axes[1].set_xticks(range(0, len(dates), tick_step))
    axes[1].set_xticklabels(dates[::tick_step], rotation=30, ha="right", fontsize=9)

    plt.tight_layout()
    url = _save_fig(fig, "anomaly_trend")
    return f"Anomaly trend chart saved. View at: {url}"


# ── Tool 5: Claim CBC radar chart ─────────────────────────────────────────────

@tool
async def plot_claim_cbc_profile(claim_id: str) -> str:
    """
    Plot a radar chart of a claim's CBC values vs normal reference ranges.
    Returns the URL of the saved chart image.
    """
    from sqlalchemy import select
    from app.db.sessions import async_session
    from app.models.claim import CBCData, Claim
    from app.models.fraud_flag import FraudFlag
    import matplotlib.pyplot as plt
    import numpy as np

    _setup_style()

    async with async_session() as session:
        claim_result = await session.execute(select(Claim).where(Claim.claim_id == claim_id))
        claim = claim_result.scalar_one_or_none()
        if not claim:
            return f"Claim {claim_id} not found."

        cbc_result = await session.execute(select(CBCData).where(CBCData.claim_id == claim.id))
        cbc = cbc_result.scalar_one_or_none()
        if not cbc:
            return f"No CBC data for claim {claim_id}."

        flags_result = await session.execute(
            select(FraudFlag).where(FraudFlag.claim_id == claim.id)
        )
        flags = flags_result.scalars().all()

    # Normal reference ranges (midpoint used for normalization)
    normal_mid = {"HGB": 14.0, "HCT": 43.0, "MCV": 89.0, "MCHC": 33.5,
                  "NEU": 55.0, "LYM": 32.0, "EOS": 3.0, "BAS": 1.0, "MON": 6.0, "PLT": 275.0}
    normal_range = {"HGB": (12, 17), "HCT": (36, 50), "MCV": (80, 100), "MCHC": (31, 36),
                    "NEU": (40, 70), "LYM": (20, 45), "EOS": (1, 5), "BAS": (0.5, 1.5),
                    "MON": (2, 10), "PLT": (150, 400)}

    labs = list(normal_mid.keys())
    values = [getattr(cbc, lab, 0) for lab in labs]
    # Normalize: 1.0 = at midpoint of normal range
    norm_values = [v / normal_mid[lab] for v, lab in zip(values, labs)]

    N = len(labs)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    norm_values_plot = norm_values + norm_values[:1]
    normal_line = [1.0] * N + [1.0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Radar
    ax = fig.add_subplot(121, polar=True)
    ax.plot(angles, normal_line, "g--", linewidth=1, alpha=0.5, label="Normal midpoint")
    ax.fill(angles, normal_line, alpha=0.1, color="green")
    ax.plot(angles, norm_values_plot, "b-", linewidth=2, label="Patient values")
    ax.fill(angles, norm_values_plot, alpha=0.25, color="blue")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labs, fontsize=10)
    ax.set_title(f"CBC Profile — Claim {claim_id[:12]}", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # Fraud flags table
    ax2 = axes[1]
    ax2.axis("off")
    flag_data = []
    for f in flags:
        flag_data.append([
            f"Model {f.model_id}",
            f"{f.anomaly_score:.3f}",
            "⚠ YES" if f.is_anomaly else "✓ NO",
            f.severity.upper(),
            f.flag_reason[:50] + "..." if len(f.flag_reason) > 50 else f.flag_reason,
        ])

    if flag_data:
        table = ax2.table(
            cellText=flag_data,
            colLabels=["Model", "Score", "Anomaly", "Severity", "Reason"],
            cellLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#ecf0f1")
        ax2.set_title(f"Fraud Detection Results\nDiagnosis: {claim.claimed_diagnosis}", fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No fraud flags found", ha="center", va="center", fontsize=12)

    plt.tight_layout()
    url = _save_fig(fig, f"claim_cbc_{claim_id[:8]}")
    return f"CBC profile chart for claim {claim_id} saved. View at: {url}"


# ── Tool 6: Top anomalous patients bar chart ──────────────────────────────────

@tool
async def plot_top_anomalous_patients(top_n: int = 10) -> str:
    """
    Plot a bar chart of the top N patients with highest trajectory anomaly scores.
    Returns the URL of the saved chart image.
    """
    from sqlalchemy import select
    from app.db.sessions import async_session
    from app.models.patient_trajectory import PatientTrajectory
    import matplotlib.pyplot as plt

    _setup_style()

    async with async_session() as session:
        result = await session.execute(
            select(PatientTrajectory)
            .where(PatientTrajectory.trajectory_anomaly_score.isnot(None))
            .order_by(PatientTrajectory.trajectory_anomaly_score.desc())
            .limit(top_n)
        )
        patients = result.scalars().all()

    if not patients:
        return "No patient trajectory data available."

    pids = [p.patient_id for p in patients]
    scores = [float(p.trajectory_anomaly_score or 0) for p in patients]
    is_anomaly = [bool(p.is_trajectory_anomaly) for p in patients]
    colors = ["#e74c3c" if a else "#f39c12" for a in is_anomaly]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(pids[::-1], scores[::-1], color=colors[::-1], edgecolor="white", height=0.6)
    ax.set_xlabel("Trajectory Anomaly Score", fontsize=12)
    ax.set_title(f"Top {top_n} Patients by Trajectory Anomaly Score", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.1)
    ax.axvline(0.5, color="#e74c3c", linestyle="--", alpha=0.5, label="Anomaly threshold")

    for bar, score, anom in zip(bars, scores[::-1], is_anomaly[::-1]):
        label = f"{score:.3f} {'⚠' if anom else '✓'}"
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9)

    ax.legend()
    plt.tight_layout()
    url = _save_fig(fig, "top_patients")
    return f"Top anomalous patients chart saved. View at: {url}"


# ── All visualization tools list ──────────────────────────────────────────────

VIZ_TOOLS = [
    plot_facility_anomaly_scores,
    plot_patient_trajectory,
    plot_disease_distribution,
    plot_anomaly_trend,
    plot_claim_cbc_profile,
    plot_top_anomalous_patients,
]
