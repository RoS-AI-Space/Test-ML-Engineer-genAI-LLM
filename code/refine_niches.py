#!/usr/bin/env python3
"""
Post-refinement: LLM validation per niche + embedding-based reassignment of outliers.

Reads result/niches.json and cache/embeddings.npy (same preprocess as segment.py).
Writes NEW files with suffix _refined (does not overwrite niches.json, etc.):
  niches_refined.json, app_niche_mapping_refined.csv, metrics_refined.json, umap_clusters_refined.html

Usage:
  python refine_niches.py
  python refine_niches.py --no-umap
  python refine_niches.py --tag _refined_v2

Environment: same .env as segment.py (LLM keys). Optional:
  REFINE_REASSIGN_THRESHOLD=0.58  (min cosine sim to another niche centroid to move)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Default output: stem + tag + ext  e.g. niches_refined.json
DEFAULT_TAG = "_refined"
REVIEW_NICHE_NAME = "Refinement / Unassigned"
REVIEW_NICHE_DESC = (
    "Apps moved during refinement: no other niche centroid exceeded the similarity threshold."
)


def _output_path(result_dir: Path, filename: str, tag: str) -> Path:
    """Build e.g. niches.json + _refined -> niches_refined.json"""
    p = Path(filename)
    return result_dir / f"{p.stem}{tag}{p.suffix}"


def _load_niche_membership(niches_path: Path) -> dict[str, list[str]]:
    """niche_name -> ordered list of trackName."""
    with open(niches_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, list[str]] = {}
    for block in data.get("niches", []):
        name = block.get("niche_name", "")
        comps = block.get("competitors") or []
        if name:
            seen: set[str] = set()
            uniq: list[str] = []
            for c in comps:
                c = str(c).strip()
                if c and c not in seen:
                    seen.add(c)
                    uniq.append(c)
            out[name] = uniq
    return out


def _build_name_to_row(df: pd.DataFrame) -> dict[str, int]:
    """First occurrence index per trackName."""
    m: dict[str, int] = {}
    for i in range(len(df)):
        tn = str(df.iloc[i].get("trackName", "")).strip()
        if tn and tn not in m:
            m[tn] = i
    return m


def _validation_prompt(
    niche_name: str,
    niche_description: str,
    apps: list[tuple[str, str]],
) -> str:
    """Build user prompt: (trackName, overview_snippet)."""
    lines = []
    for tn, snip in apps:
        lines.append("- %s | %s" % (tn, snip[:180].replace("\n", " ")))
    return (
        "Niche name: %s\nNiche description: %s\n\nApps in this cluster:\n%s\n\n"
        "Return JSON only with keys: is_homogeneous (boolean), outliers (array of trackName "
        "strings that are NOT direct competitors with the core group or are clearly wrong here), "
        "notes (optional string)."
        % (niche_name, niche_description, "\n".join(lines))
    )


VALIDATION_SYSTEM = """You are a mobile app market analyst. Direct competitors solve the same user problem and substitute each other in product research.

Return only valid JSON, no markdown."""


def _validate_one_niche(args: tuple) -> tuple[str, dict | None]:
    """Worker: returns (niche_name, parsed dict or None)."""
    niche_name, niche_description, apps_tuples = args
    if len(apps_tuples) <= 1:
        return niche_name, {"is_homogeneous": True, "outliers": [], "notes": "single_app"}

    import segment as seg

    prompt = _validation_prompt(niche_name, niche_description, apps_tuples)
    raw = seg.llm_call_json(prompt, VALIDATION_SYSTEM, temperature=0.0)
    if raw is None:
        log.warning("LLM validation failed for niche=%s", niche_name[:60])
        return niche_name, None
    if not isinstance(raw, dict):
        return niche_name, None
    outliers = raw.get("outliers") or []
    if not isinstance(outliers, list):
        outliers = []
    hom = raw.get("is_homogeneous")
    if hom is None:
        hom = len(outliers) == 0
    return niche_name, {
        "is_homogeneous": bool(hom),
        "outliers": [str(x).strip() for x in outliers if str(x).strip()],
        "notes": raw.get("notes", ""),
    }


def _sample_apps_for_prompt(
    names: list[str],
    name_to_idx: dict[str, int],
    df: pd.DataFrame,
    max_apps: int,
) -> list[tuple[str, str]]:
    """If too many apps, keep all names in outlier resolution but shorten prompt: use first max_apps."""
    out: list[tuple[str, str]] = []
    for tn in names[:max_apps]:
        if tn not in name_to_idx:
            continue
        row = df.iloc[name_to_idx[tn]]
        snip = str(row.get("overview_clean", row.get("overview", "")))
        out.append((tn, snip))
    return out


def _centroids(
    membership: dict[str, list[str]],
    name_to_idx: dict[str, int],
    embs: np.ndarray,
) -> dict[str, np.ndarray]:
    """Normalized mean embedding per niche (non-empty)."""
    out: dict[str, np.ndarray] = {}
    for niche, names in membership.items():
        idxs = [name_to_idx[n] for n in names if n in name_to_idx]
        if not idxs:
            continue
        v = embs[idxs].mean(axis=0)
        nrm = np.linalg.norm(v)
        if nrm > 0:
            v = v / nrm
        out[niche] = v
    return out


def _best_reassignment(
    outlier: str,
    source_niche: str,
    emb_idx: int,
    embs: np.ndarray,
    centroids: dict[str, np.ndarray],
    threshold: float,
) -> str | None:
    """Return target niche name or None -> review bucket."""
    v = embs[emb_idx : emb_idx + 1]
    best_n = None
    best_s = -1.0
    for niche, c in centroids.items():
        if niche == source_niche:
            continue
        s = float(cosine_similarity(v, c.reshape(1, -1))[0, 0])
        if s > best_s:
            best_s = s
            best_n = niche
    if best_n is not None and best_s >= threshold:
        return best_n
    return None


def _load_niche_meta(niches_path: Path) -> dict[str, str]:
    """niche_name -> description from original niches.json."""
    with open(niches_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, str] = {}
    for block in data.get("niches", []):
        n = block.get("niche_name", "")
        d = block.get("niche_description", "")
        if n:
            out[n] = d
    return out


def run_refinement(
    tag: str = DEFAULT_TAG,
    write_umap: bool = True,
    max_prompt_apps: int = 45,
    threshold: float | None = None,
) -> None:
    """Run full refinement pipeline."""
    threshold = threshold if threshold is not None else float(
        os.environ.get("REFINE_REASSIGN_THRESHOLD", "0.58")
    )

    import segment as seg

    niches_path = config.RESULT_DIR / "niches.json"
    if not niches_path.exists():
        log.error("Missing %s", niches_path)
        sys.exit(1)
    emb_path = config.CACHE_DIR / "embeddings.npy"
    if not emb_path.exists():
        log.error("Missing embeddings %s — run segment.py first.", emb_path)
        sys.exit(1)

    log.info("[1/5] Load data, embeddings, niches.json (same preprocess as segment)...")
    df = seg.load_data()
    df = seg.preprocess(df)
    embs = np.load(str(emb_path))
    if embs.shape[0] != len(df):
        log.error("Embeddings rows %s != df rows %s", embs.shape[0], len(df))
        sys.exit(1)

    name_to_idx = _build_name_to_row(df)
    membership = _load_niche_membership(niches_path)
    niche_descriptions = _load_niche_meta(niches_path)

    log.info(
        "[1/5] Done: %d apps in dataframe, %d niches in %s",
        len(df),
        len(membership),
        niches_path.name,
    )
    workers = max(1, config.LLM_ENRICHMENT_CONCURRENCY)

    tasks = []
    for niche_name, app_names in membership.items():
        apps_tuples = _sample_apps_for_prompt(
            app_names, name_to_idx, df, max_apps=max_prompt_apps
        )
        desc = niche_descriptions.get(niche_name, "")
        tasks.append((niche_name, desc, apps_tuples))

    total_llm = len(tasks)
    validation: dict[str, dict | None] = {}
    log.info(
        "[2/5] LLM validation: %d niches, %d parallel workers (this stage is the slowest)",
        total_llm,
        workers,
    )
    done_llm = 0
    progress_every = max(1, min(25, total_llm // 20 or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_validate_one_niche, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            n, res = fut.result()
            validation[n] = res
            done_llm += 1
            if done_llm == 1 or done_llm == total_llm or done_llm % progress_every == 0:
                log.info(
                    "[2/5] LLM progress: %d/%d (%.1f%%)",
                    done_llm,
                    total_llm,
                    100.0 * done_llm / total_llm,
                )
    ok_llm = sum(1 for v in validation.values() if v is not None)
    log.info(
        "[2/5] LLM validation finished: %d/%d niches returned JSON, %d empty/failed",
        ok_llm,
        total_llm,
        total_llm - ok_llm,
    )

    log.info("[3/5] Remove outliers, reassign by embedding centroids (threshold=%.3f)...", threshold)
    # Copy membership for mutation
    mem = {k: list(v) for k, v in membership.items()}
    moves: list[tuple[str, str, str]] = []  # outlier, from_niche, to_niche or REVIEW

    for niche_name, app_names in list(mem.items()):
        vr = validation.get(niche_name)
        if not vr:
            continue
        outliers = [o for o in vr.get("outliers", []) if o in app_names]
        for o in outliers:
            if o not in name_to_idx:
                log.warning("Outlier not in df: %s", o[:80])
                continue
            mem[niche_name].remove(o)
            moves.append((o, niche_name, "_pending"))

    # Recompute centroids from membership after removals (before reassign)
    centroids = _centroids(mem, name_to_idx, embs)

    for i, (outlier, src, _) in enumerate(moves):
        ei = name_to_idx.get(outlier)
        if ei is None:
            continue
        tgt = _best_reassignment(outlier, src, ei, embs, centroids, threshold)
        if tgt is None:
            tgt = REVIEW_NICHE_NAME
        moves[i] = (outlier, src, tgt)
        if tgt not in mem:
            mem[tgt] = []
        if outlier not in mem[tgt]:
            mem[tgt].append(outlier)

    # Drop empty niches (except we might have emptied some)
    mem = {k: v for k, v in mem.items() if len(v) > 0}

    if REVIEW_NICHE_NAME in mem and len(mem[REVIEW_NICHE_NAME]) == 0:
        del mem[REVIEW_NICHE_NAME]

    # Apps present in df but missing from any niche (should be rare)
    for i in range(len(df)):
        tn = str(df.iloc[i].get("trackName", "")).strip()
        if not tn:
            continue
        in_mem = any(tn in names for names in mem.values())
        if not in_mem:
            mem.setdefault(REVIEW_NICHE_NAME, []).append(tn)

    mem = {k: v for k, v in mem.items() if len(v) > 0}

    log.info(
        "[3/5] Done: %d outlier moves queued, %d niches after cleanup",
        len(moves),
        len(mem),
    )

    # Build niches.json structure
    niches_out = []
    for niche_name, comps in sorted(mem.items(), key=lambda x: (-len(x[1]), x[0])):
        desc = niche_descriptions.get(niche_name, "")
        if niche_name == REVIEW_NICHE_NAME and not desc:
            desc = REVIEW_NICHE_DESC
        niches_out.append(
            {
                "niche_name": niche_name,
                "niche_description": desc,
                "competitors": comps,
                "metadata": {"cluster_size": len(comps)},
            }
        )

    result_dir = config.RESULT_DIR
    result_dir.mkdir(parents=True, exist_ok=True)

    log.info("[4/5] Write outputs (tag=%s): JSON, CSV, metrics...", tag)
    p_json = _output_path(result_dir, "niches.json", tag)
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump({"niches": niches_out}, f, ensure_ascii=False, indent=2)
    log.info("Wrote %s", p_json)

    # trackName -> niche_name
    assignment: dict[str, str] = {}
    for niche_name, comps in mem.items():
        for tn in comps:
            assignment[tn] = niche_name

    rows = []
    for i in range(len(df)):
        tn = str(df.iloc[i].get("trackName", "")).strip()
        sub = assignment.get(tn, REVIEW_NICHE_NAME)
        rows.append(
            {
                "AppName": tn,
                "Description": str(df.iloc[i].get("overview_clean", ""))[:200],
                "SubNicheName": sub,
            }
        )

    p_csv = _output_path(result_dir, "app_niche_mapping.csv", tag)
    pd.DataFrame(rows).to_csv(p_csv, index=False, encoding="utf-8-sig")
    log.info("Wrote %s", p_csv)

    # Numeric labels for metrics
    unique_names = sorted(set(assignment.values()))
    nm_to_id = {n: i for i, n in enumerate(unique_names)}
    labels = np.array(
        [
            nm_to_id[
                assignment.get(str(df.iloc[i].get("trackName", "")).strip(), REVIEW_NICHE_NAME)
            ]
            for i in range(len(df))
        ]
    )

    metrics = seg.compute_metrics(embs, labels)
    p_met = _output_path(result_dir, "metrics.json", tag)
    with open(p_met, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    log.info("Wrote %s", p_met)

    log.info(
        "[4/5] Summary: outliers processed=%d, cosine threshold=%.3f, "
        "clusters in assignment=%d",
        len(moves),
        threshold,
        len(unique_names),
    )

    if write_umap:
        import umap
        import plotly.express as px

        log.info("[5/5] Building UMAP 2D (may take 1–3 min on ~4k points)...")
        reducer = umap.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
        )
        coords = reducer.fit_transform(embs)
        plot_df = pd.DataFrame(
            {
                "x": coords[:, 0],
                "y": coords[:, 1],
                "trackName": df["trackName"].values,
                "niche_name": [assignment.get(str(t).strip(), REVIEW_NICHE_NAME) for t in df["trackName"].values],
            }
        )
        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color="niche_name",
            hover_data=["trackName"],
            title="App Segmentation — UMAP 2D (refined)",
            width=1600,
            height=1000,
        )
        fig.update_traces(marker=dict(size=3, opacity=0.7))
        fig.update_layout(
            showlegend=True,
            legend=dict(font=dict(size=6)),
            legend_title=dict(font=dict(size=8)),
        )
        p_html = _output_path(result_dir, "umap_clusters.html", tag)
        fig.write_html(str(p_html))
        log.info("[5/5] Wrote %s — refinement complete.", p_html)
    else:
        log.info("[5/5] Skipped UMAP (--no-umap) — refinement complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine niches via LLM + embeddings; write *_refined files.")
    parser.add_argument(
        "--tag",
        type=str,
        default=DEFAULT_TAG,
        help='Suffix before extension (default: "%s")' % DEFAULT_TAG,
    )
    parser.add_argument("--no-umap", action="store_true", help="Skip writing UMAP HTML.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override REFINE_REASSIGN_THRESHOLD cosine similarity.",
    )
    args = parser.parse_args()

    if not config.has_llm_access():
        log.error("No LLM access — set keys and LLM_MODEL in .env")
        sys.exit(1)

    run_refinement(tag=args.tag, write_umap=not args.no_umap, threshold=args.threshold)


if __name__ == "__main__":
    main()
