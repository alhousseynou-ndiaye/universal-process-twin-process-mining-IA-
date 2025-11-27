# process_analyzer.py

from typing import Dict, Any, List
from collections import Counter

import pandas as pd


def compute_kpis(events: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcule des KPI avancés SI une colonne 'timestamp' est disponible.

    - Cycle time (par cas)
    - Durée moyenne par étape (step)
    - Durée moyenne par transition (step -> next_step)
    """
    if "timestamp" not in events.columns:
        return {"has_timestamp": False}

    df = events.copy()

    # S'assurer que timestamp est bien en datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if df["timestamp"].isna().all():
        return {"has_timestamp": False}

    # On enlève les lignes sans timestamp exploitable
    df = df.dropna(subset=["timestamp"]).copy()

    # ===== KPI 1 : Cycle time par cas =====
    cycle = df.groupby("case_id")["timestamp"].agg(["min", "max"])
    cycle["delta_hours"] = (cycle["max"] - cycle["min"]).dt.total_seconds() / 3600.0

    if len(cycle) > 0:
        avg_cycle = float(cycle["delta_hours"].mean())
        min_cycle = float(cycle["delta_hours"].min())
        max_cycle = float(cycle["delta_hours"].max())
    else:
        avg_cycle = min_cycle = max_cycle = None

    # ===== KPI 2 & 3 : Durées entre événements =====
    # Tri par cas + timestamp
    df = df.sort_values(["case_id", "timestamp"]).copy()
    df["next_timestamp"] = df.groupby("case_id")["timestamp"].shift(-1)
    df["next_step"] = df.groupby("case_id")["step"].shift(-1)

    df["delta_hours"] = (
        df["next_timestamp"] - df["timestamp"]
    ).dt.total_seconds() / 3600.0

    # On enlève les deltas négatifs ou NaN (dernier événement, anomalies)
    mask = df["delta_hours"].notna() & (df["delta_hours"] >= 0)
    df_dur = df[mask].copy()

    # Durée moyenne par STEP
    step_stats = (
        df_dur.groupby("step")["delta_hours"]
        .agg(["count", "mean"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    steps_duration: List[Dict[str, Any]] = [
        {
            "step": row["step"],
            "avg_hours": round(float(row["mean"]), 2),
            "count": int(row["count"]),
        }
        for _, row in step_stats.iterrows()
    ]
    slowest_steps = steps_duration[:3]

    # Durée moyenne par TRANSITION (step -> next_step)
    trans_stats = (
        df_dur.groupby(["step", "next_step"])["delta_hours"]
        .agg(["count", "mean"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    transitions_duration: List[Dict[str, Any]] = [
        {
            "from": row["step"],
            "to": row["next_step"],
            "avg_hours": round(float(row["mean"]), 2),
            "count": int(row["count"]),
        }
        for _, row in trans_stats.iterrows()
    ]
    slowest_transitions = transitions_duration[:3]

    return {
        "has_timestamp": True,
        "cycle_time": {
            "avg_hours": round(avg_cycle, 2) if avg_cycle is not None else None,
            "min_hours": round(min_cycle, 2) if min_cycle is not None else None,
            "max_hours": round(max_cycle, 2) if max_cycle is not None else None,
        },
        "steps_duration": steps_duration,
        "slowest_steps": slowest_steps,
        "transitions_duration": transitions_duration,
        "slowest_transitions": slowest_transitions,
    }


def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyse un DataFrame standardisé avec au minimum :
    - case_id
    - step

    Optionnel :
    - timestamp (pour KPI temps)
    """
    if not {"case_id", "step"}.issubset(df.columns):
        raise ValueError("Le DataFrame doit contenir au moins 'case_id' et 'step'.")

    events = df.copy()

    # S'il y a un timestamp, on tente de le parser
    if "timestamp" in events.columns:
        events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce")

    # Tri pour respecter l'ordre du processus
    if "timestamp" in events.columns:
        events = events.sort_values(["case_id", "timestamp"])
    else:
        events = events.sort_values(["case_id"]).reset_index(drop=True)

    # ===== STATS DE BASE =====
    num_cases = int(events["case_id"].nunique())
    num_events = int(len(events))
    step_counts = events["step"].value_counts()
    num_steps = int(len(step_counts))

    # ===== GRAPH : NOEUDS =====
    nodes = [
        {"id": step, "count": int(count)} for step, count in step_counts.items()
    ]

    # ===== GRAPH : TRANSITIONS =====
    edge_counter: Counter = Counter()
    for case_id, group in events.groupby("case_id"):
        steps_seq = group["step"].tolist()
        for a, b in zip(steps_seq, steps_seq[1:]):
            edge_counter[(a, b)] += 1

    edges = [
        {"from": src, "to": dst, "count": int(count)}
        for (src, dst), count in edge_counter.items()
    ]

    stats = {
        "num_cases": num_cases,
        "num_events": num_events,
        "num_steps": num_steps,
        "num_transitions": len(edges),
    }

    graph = {
        "nodes": nodes,
        "edges": edges,
    }

    # ===== KPI AVANCÉS =====
    kpi = compute_kpis(events)

    return {
        "stats": stats,
        "graph": graph,
        "kpi": kpi,
    }
