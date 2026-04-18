#!/usr/bin/env python3
"""
App Segmentation Pipeline — LLM-Enhanced Coarse-to-Fine Segmentation

Architecture:
  Stage 0: Data prep, dedup
  Stage 1: LLM signal enrichment (optional, via litellm — graceful fallback)
  Stage 2: Embedding (local bge-small or API-based via litellm — graceful fallback)
  Stage 3: Agglomerative clustering (complete linkage) + recursive refinement
  Stage 4: LLM naming (optional, via litellm — graceful fallback) / rule-based
  Stage 5: Quality metrics
  Stage 6: Export JSON/CSV + UMAP visualization

Usage:
  python segment.py                       # try LLM if API key present, fallback to rule-based
  python segment.py --no-llm              # force no LLM calls at all
  python segment.py --no-llm-enrichment   # skip LLM enrichment
  python segment.py --no-llm-naming        # skip LLM naming
  python segment.py --help

Configuration:
  Copy .env.example to .env and fill in OPENROUTER_API_KEY.
  See config.py for all configurable parameters.
"""

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

import config

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# ─── LLM FALLBACK STATE ────────────────────────────────────────
# If LLM calls fail repeatedly, we disable LLM for the rest of the run
_llm_disabled = False
_llm_fail_count = 0
_LLM_MAX_FAILS = 5  # after this many consecutive failures, disable LLM
_llm_state_lock = threading.Lock()


def _llm_available():
    """Check if LLM is usable (has key and not disabled by too many failures)."""
    with _llm_state_lock:
        disabled = _llm_disabled
    return not disabled and config.has_llm_access()


def _llm_failed():
    """Record a failed LLM call. Disable LLM after too many failures."""
    global _llm_disabled, _llm_fail_count
    with _llm_state_lock:
        _llm_fail_count += 1
        n = _llm_fail_count
        if _llm_fail_count >= _LLM_MAX_FAILS:
            _llm_disabled = True
    log.warning("LLM call failed (%d consecutive failures)", n)
    if n >= _LLM_MAX_FAILS:
        log.warning("Too many LLM failures (%d). Disabling LLM for remainder of run.", n)


def _llm_succeeded():
    """Record a successful LLM call. Reset failure counter."""
    global _llm_fail_count
    with _llm_state_lock:
        _llm_fail_count = 0


# ══════════════════════════════════════════════════════════════════
# STAGE 0: DATA PREPARATION
# ══════════════════════════════════════════════════════════════════

def load_data(filepath=None):
    filepath = Path(filepath) if filepath else config.DATA_FILE
    log.info("Loading data from %s", filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    log.info("Loaded %d apps", len(df))
    return df


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w\s.,;:!?\-\u2019()/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def deduplicate(df):
    from rapidfuzz import fuzz as rfuzz
    from rapidfuzz import process as rprocess
    names = df["trackName"].fillna("").str.strip().tolist()
    dup = set()
    for i, name in enumerate(names):
        if i in dup or len(name) < 3:
            continue
        matches = rprocess.extract(name, names, scorer=rfuzz.token_sort_ratio, limit=None, score_cutoff=92)
        for _, score, idx in matches:
            if idx != i and score >= 92 and idx not in dup:
                dup.add(idx)
    if dup:
        log.info("Removing %d near-duplicate apps", len(dup))
        df = df.drop(index=list(dup)).reset_index(drop=True)
    return df


def prepare_features(features):
    if isinstance(features, list):
        return "; ".join(str(f).strip() for f in features[:20] if str(f).strip())
    return str(features) if features else ""


def preprocess(df):
    log.info("Preprocessing %d apps...", len(df))
    df["overview_clean"] = df["overview"].fillna("").apply(clean_text)
    df["description_clean"] = df["description"].fillna("").apply(clean_text)
    df["features_text"] = df["features"].apply(prepare_features)
    empty = df["overview_clean"].str.strip() == ""
    if empty.any():
        df.loc[empty, "overview_clean"] = df.loc[empty, "description_clean"].str[:500]
    df["features_text"] = df["features_text"].fillna("")
    df["trackName"] = df["trackName"].fillna("Unknown").str.strip()
    df = deduplicate(df).reset_index(drop=True)
    log.info("After preprocessing: %d apps remaining", len(df))
    return df


def build_canonical_text(row, enrichment=None):
    parts = []
    tn = str(row.get("trackName", "")).strip()
    ov = str(row.get("overview_clean", "")).strip()
    ft = str(row.get("features_text", "")).strip()
    if tn:
        parts.append("App name: " + tn)
        parts.append("App name: " + tn)
        parts.append("App name: " + tn)
    if ov:
        parts.append("Overview: " + ov[:600])
    if ft:
        parts.append("Features: " + ft[:800])
    if enrichment:
        for key, label in [("primary_jtbd", "Core purpose"), ("target_user", "Target user"),
                           ("category_narrow", "Narrow category"), ("core_value", "Value proposition")]:
            val = enrichment.get(key, "")
            if val:
                parts.append(f"{label}: {val}")
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════
# STAGE 1: LLM ENRICHMENT — with graceful fallback
# ══════════════════════════════════════════════════════════════════

def _try_litellm_completion(system_prompt, user_prompt, temperature=0.0, max_tokens=1024):
    """Try LLM call via litellm with fallback chain. Returns (raw_text, model_used) or (None, None)."""
    try:
        import litellm
        litellm.suppress_debug_info = True
    except ImportError:
        log.warning("litellm not installed. Cannot use LLM.")
        return None, None

    chain = config.get_llm_chain()
    if not chain:
        log.warning("No LLM models configured (set LLM_MODEL in .env).")
        return None, None

    timeout = getattr(config, "LLM_TIMEOUT", config.HTTP_TIMEOUT)
    for model_name, api_key, api_base in chain:
        if not model_name:
            continue
        try:
            log.debug("Trying LLM model: %s", model_name)
            response = litellm.completion(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
            )
            content = response.choices[0].message.content
            if content is None:
                content = ""
            content = content.strip()
            if not content and hasattr(response.choices[0].message, 'reasoning_content'):
                rc = response.choices[0].message.reasoning_content
                if rc and rc.strip():
                    content = rc.strip()
                    log.debug("Using reasoning_content from %s (content was empty)", model_name)
            if not content:
                log.warning("LLM model %s returned empty content", model_name)
                continue
            if model_name != config.LLM_MODEL_NAME:
                log.info("Fallback model %s succeeded", model_name)
            return content, model_name
        except Exception as e:
            log.warning("LLM model %s failed: %s", model_name, str(e)[:200])
            continue

    log.warning("All LLM models in fallback chain failed.")
    return None, None


def _try_openai_completion(system_prompt, user_prompt, temperature=0.0):
    """Fallback: try OpenAI directly if litellm fails."""
    if not config.OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        timeout = getattr(config, "LLM_TIMEOUT", config.HTTP_TIMEOUT)
        client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=timeout)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.warning("OpenAI fallback also failed: %s", str(e)[:200])
        return None


def llm_call_json(prompt_text, system_prompt, temperature=0.0):
    """Universal LLM call that returns parsed JSON or None. Tries fallback chain first, then OpenAI."""
    if not _llm_available():
        return None

    raw, _model_used = _try_litellm_completion(system_prompt, prompt_text, temperature)

    if raw is None and config.OPENAI_API_KEY:
        log.info("All litellm models failed, trying OpenAI directly...")
        raw = _try_openai_completion(system_prompt, prompt_text, temperature)

    if raw is None:
        _llm_failed()
        return None

    # Parse JSON - handle both single object and array
    json_str = raw
    # Extract from code blocks first
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0].strip()
    
    # Try parse as-is
    try:
        result = json.loads(json_str)
        _llm_succeeded()
        return result
    except json.JSONDecodeError as e:
        # Try to find JSON by looking for [ and ]
        start = json_str.find('[')
        end = json_str.rfind(']')
        if start >= 0 and end > start:
            try:
                result = json.loads(json_str[start:end+1])
                _llm_succeeded()
                return result
            except:
                pass
    
    log.warning("Failed to parse LLM JSON: %s...", raw[:200])
    _llm_failed()
    return None


ENRICHMENT_SYSTEM_PROMPT = """You are an expert app market analyst. Extract structured competitive attributes from mobile apps.

Rules:
- category_narrow must be granular: "AI Headshot Generator" NOT "Photo App"
- Focus on the PRIMARY use case
- Reply with ONLY a GitHub-flavored markdown table (no JSON, no code fences, no extra commentary before/after the table)
- Required columns in order: # | trackName | primary_jtbd | target_user | core_value | category_narrow
- Each cell: short phrases (2-5 words) where appropriate; trackName must match the app name from the prompt

Example row:
| 1 | Logo Maker AI | Create custom logos | Small businesses | Pro logos without designer | AI Logo Maker"""

ENRICHMENT_BATCH_TEMPLATE = """Extract data for these apps. Return as simple markdown table:

| # | trackName | primary_jtbd | target_user | core_value | category_narrow |
|---|-----------|--------------|-------------|------------|-----------------|
| 1 | AppName1 | ... | ... | ... | ... |
| 2 | AppName2 | ... | ... | ... | ... |

{apps}

Fill in the table with real data. Use short phrases (2-5 words)."""


def parse_enrichment_table(text, batch_apps):
    """Parse markdown table response from LLM. Returns list of enrichment dicts."""
    results = []
    lines = text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line or not line.startswith("|") or "---" in line:
            continue

        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]

        if len(parts) < 6:
            continue
        if parts[0] == "#" or not parts[0].isdigit():
            continue

        try:
            result = {
                "trackName": parts[1] if len(parts) > 1 else "",
                "primary_jtbd": parts[2] if len(parts) > 2 else "",
                "target_user": parts[3] if len(parts) > 3 else "",
                "core_value": parts[4] if len(parts) > 4 else "",
                "category_narrow": parts[5] if len(parts) > 5 else "",
            }
            if result["trackName"] and result["category_narrow"]:
                results.append(result)
        except Exception:
            continue

    return results


def _enrich_one_batch(df, batch_start, batch_end, batch_size):
    """Single enrichment batch (for parallel workers). Returns (batch_start, batch_end, items)."""
    batch_apps = []
    for idx in range(batch_start, batch_end):
        row = df.iloc[idx]
        batch_apps.append(
            "%d. %s: %s"
            % (idx + 1, row.get("trackName", ""), str(row.get("overview_clean", ""))[:250])
        )

    prompt = ENRICHMENT_BATCH_TEMPLATE.format(apps="\n".join(batch_apps))
    max_tok = max(2048, min(8192, batch_size * 100))
    raw, _model_used = _try_litellm_completion(
        ENRICHMENT_SYSTEM_PROMPT,
        prompt,
        temperature=0.0,
        max_tokens=max_tok,
    )
    if not raw:
        _llm_failed()
        return batch_start, batch_end, []

    parsed_items = parse_enrichment_table(raw, batch_apps)
    if not parsed_items:
        log.warning("Enrichment batch %d-%d: markdown table parse returned no rows", batch_start, batch_end)
        _llm_failed()
        return batch_start, batch_end, []

    _llm_succeeded()
    return batch_start, batch_end, parsed_items


def llm_enrich_apps(df, batch_size=20):
    """LLM enrichment with batching and parallel API calls (markdown table only)."""
    if not _llm_available():
        log.info("LLM enrichment skipped (no API key or disabled). Using text-only embeddings.")
        return [None] * len(df)

    log.info(
        "Starting LLM enrichment for %d apps (batch_size=%d, concurrency=%d)...",
        len(df),
        batch_size,
        config.LLM_ENRICHMENT_CONCURRENCY,
    )
    enrichments = [None] * len(df)

    tasks = [(s, min(s + batch_size, len(df))) for s in range(0, len(df), batch_size)]
    workers = min(config.LLM_ENRICHMENT_CONCURRENCY, max(1, len(tasks)))

    if workers <= 1:
        for batch_start, batch_end in tasks:
            _bs, _be, parsed_items = _enrich_one_batch(df, batch_start, batch_end, batch_size)
            for item in parsed_items:
                track_name = item.get("trackName", "")
                if not track_name:
                    continue
                for idx in range(_bs, _be):
                    if df.iloc[idx].get("trackName", "") == track_name:
                        if enrichments[idx] is None:
                            enrichments[idx] = item
                        break
            success = sum(1 for e in enrichments if e is not None)
            log.info("  Enriched %d/%d (%d successful)", _be, len(df), success)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_enrich_one_batch, df, bs, be, batch_size): (bs, be)
                for bs, be in tasks
            }
            for fut in as_completed(futures):
                batch_start, batch_end, parsed_items = fut.result()
                for item in parsed_items:
                    track_name = item.get("trackName", "")
                    if not track_name:
                        continue
                    for idx in range(batch_start, batch_end):
                        if df.iloc[idx].get("trackName", "") == track_name:
                            if enrichments[idx] is None:
                                enrichments[idx] = item
                            break
                success = sum(1 for e in enrichments if e is not None)
                log.info("  Enriched (parallel) up to %d/%d (%d successful)", batch_end, len(df), success)

    final_success = sum(1 for e in enrichments if e is not None)
    log.info(
        "LLM enrichment complete: %d/%d successful (%.1f%%)",
        final_success,
        len(df),
        100 * final_success / max(1, len(df)),
    )

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CACHE_DIR / "enrichments.json", "w", encoding="utf-8") as f:
        json.dump(enrichments, f, ensure_ascii=False, indent=2)

    return enrichments


# ══════════════════════════════════════════════════════════════════
# STAGE 2: EMBEDDING — local or API, with fallback
# ══════════════════════════════════════════════════════════════════

def compute_embeddings(df):
    """Compute embeddings. Tries models in fallback chain: primary API -> fallback API -> local."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    chain = config.get_embedding_chain()
    last_err = None

    for model_name, api_key, api_base, is_local in chain:
        if is_local:
            log.info("Using local embedding model: %s", model_name)
            return _compute_embeddings_local(df, config.CACHE_DIR / "embeddings.npy")

        try:
            log.info("Trying embedding model: %s ...", model_name)
            embs = _compute_embeddings_api(df, config.CACHE_DIR / "embeddings.npy",
                                           model_name=model_name, api_key=api_key, api_base=api_base)
            log.info("Embedding model %s succeeded.", model_name)
            return embs
        except Exception as e:
            last_err = e
            log.warning("Embedding model %s failed: %s", model_name, str(e)[:200])
            # Remove broken cache so next model doesn't load mismatched embeddings
            cache_file = config.CACHE_DIR / "embeddings.npy"
            if cache_file.exists():
                cache_file.unlink()
                log.info("Removed mismatched cache file.")
            continue

    # Should not reach here (local always works), but just in case
    log.error("All embedding models failed. Last error: %s", str(last_err)[:300])
    raise RuntimeError(f"All embedding models failed. Last: {last_err}")


def _compute_embeddings_local(df, cache_path):
    from sentence_transformers import SentenceTransformer

    if cache_path.exists():
        embs = np.load(str(cache_path))
        if embs.shape[0] == len(df):
            log.info("Loaded cached local embeddings: %s", embs.shape)
            return embs
        log.warning("Cache mismatch, recomputing")

    model_name = config.LOCAL_EMBEDDING_MODEL
    log.info("Computing embeddings locally with %s...", model_name)
    model = SentenceTransformer(model_name)
    texts = df["canonical_text"].tolist()
    embs = model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    embs = np.array(embs, dtype=np.float32)
    log.info("Embeddings shape: %s", embs.shape)
    np.save(str(cache_path), embs)
    return embs


def _compute_embeddings_api(df, cache_path, model_name=None, api_key=None, api_base=None):
    import litellm
    litellm.suppress_debug_info = True

    if cache_path.exists():
        embs = np.load(str(cache_path))
        if embs.shape[0] == len(df):
            log.info("Loaded cached API embeddings: %s", embs.shape)
            return embs

    if model_name is None:
        model_name = config.get_embedding_model_name()
    if api_key is None:
        api_key = config.get_embedding_api_key()
    if api_base is None:
        api_base = config.get_embedding_api_base()

    log.info("Computing embeddings via API with %s...", model_name)
    texts = df["canonical_text"].tolist()

    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = litellm.embedding(
            model=model_name,
            input=batch,
            api_key=api_key,
            api_base=api_base,
            timeout=config.EMBEDDING_TIMEOUT,
        )
        batch_embs = [item["embedding"] for item in response.data]
        all_embeddings.extend(batch_embs)
        log.info("  Embedded batch %d/%d (%d vectors)", i // batch_size + 1,
                 (len(texts) + batch_size - 1) // batch_size, len(batch_embs))

    embs = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embs = embs / norms

    log.info("Embeddings shape: %s", embs.shape)
    np.save(str(cache_path), embs)
    return embs


# ══════════════════════════════════════════════════════════════════
# STAGE 3: CLUSTERING
# ══════════════════════════════════════════════════════════════════

def cluster_pipeline(embeddings):
    log.info("=" * 50)
    log.info("STAGE 3: Clustering (%d apps)", embeddings.shape[0])
    log.info("=" * 50)

    dist_thresh = config.DISTANCE_THRESHOLD
    max_size = config.MAX_CLUSTER_SIZE
    merge_thresh = config.MERGE_THRESHOLD
    noise_thresh = config.NOISE_REASSIGN_THRESHOLD

    # Adapt thresholds for embedding dimensionality
    # Higher-dimensional embeddings (e.g. 2048) tend to have lower cosine similarity
    # compared to lower-dimensional ones (e.g. 384), so we need looser thresholds
    emb_dim = embeddings.shape[1]
    if emb_dim > 1000:
        # High-dim embeddings (1024d qwen, 2048d nvidia) — need looser thresholds
        dist_thresh = 0.48
        noise_thresh = 0.50
        merge_thresh = 0.94
        log.info("Using adapted thresholds for %d-dim embeddings (dist=%.2f, noise=%.2f, merge=%.2f)",
                 emb_dim, dist_thresh, noise_thresh, merge_thresh)
    elif emb_dim > 500:
        # Mid-dim embeddings (e.g. 768d) — slightly looser than local 384d
        dist_thresh = 0.38
        noise_thresh = 0.55
        merge_thresh = 0.95
        log.info("Using adapted thresholds for %d-dim embeddings (dist=%.2f, noise=%.2f, merge=%.2f)",
                 emb_dim, dist_thresh, noise_thresh, merge_thresh)

    log.info("Step 3.1: Agglomerative clustering (complete linkage, thresh=%.2f)...", dist_thresh)
    agg = AgglomerativeClustering(
        n_clusters=None, distance_threshold=dist_thresh,
        metric="cosine", linkage="complete",
    )
    primary_labels = agg.fit_predict(embeddings)
    n_primary = len(set(primary_labels))
    prim_sizes = [int(np.sum(primary_labels == c)) for c in range(n_primary)]
    log.info("  Primary: %d clusters, min=%d, median=%d, max=%d",
             n_primary, min(prim_sizes), sorted(prim_sizes)[len(prim_sizes)//2], max(prim_sizes))

    # Filter small clusters
    labels = primary_labels.copy()
    for c in range(n_primary):
        if np.sum(primary_labels == c) < config.MIN_CLUSTER_SIZE:
            labels[labels == c] = config.NOISE_LABEL

    unique = sorted(set(labels) - {config.NOISE_LABEL})
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap.get(l, config.NOISE_LABEL) for l in labels])

    n_after = len(set(labels) - {config.NOISE_LABEL})
    n_noise = int((labels == config.NOISE_LABEL).sum())
    log.info("  After filtering: %d clusters, %d noise", n_after, n_noise)

    # Recursive refinement
    log.info("Step 3.2: Recursive refinement (max_size=%d)...", max_size)
    labels = _recursive_refine(embeddings, labels, depth=0, max_depth=config.RECURSIVE_MAX_DEPTH, max_size=max_size)
    n_refine = len(set(labels) - {config.NOISE_LABEL})
    log.info("  After refinement: %d clusters", n_refine)

    # Noise reassignment
    log.info("Step 3.3: Noise reassignment (threshold=%.2f)...", noise_thresh)
    labels = _reassign_noise(embeddings, labels, threshold=noise_thresh)
    n_after_noise = len(set(labels) - {config.NOISE_LABEL})
    n_remaining_noise = int((labels == config.NOISE_LABEL).sum())
    log.info("  After noise: %d clusters, %d unassigned", n_after_noise, n_remaining_noise)

    # Merge similar
    log.info("Step 3.4: Merging similar clusters (threshold=%.2f)...", merge_thresh)
    labels = _merge_similar(embeddings, labels, threshold=merge_thresh)
    n_final = len(set(labels) - {config.NOISE_LABEL})
    n_final_noise = int((labels == config.NOISE_LABEL).sum())
    log.info("FINAL: %d clusters, %d unassigned", n_final, n_final_noise)

    return labels


def _recursive_refine(embeddings, labels, depth=0, max_depth=5, max_size=20):
    if depth >= max_depth:
        return labels

    unique = sorted(set(labels) - {config.NOISE_LABEL})
    if not unique:
        return labels

    refined = labels.copy()
    next_label = max(unique) + 1
    did_refine = False

    for cid in unique:
        indices = np.where(labels == cid)[0]
        if len(indices) <= max_size:
            continue

        did_refine = True
        cluster_embs = embeddings[indices]
        stricter = max(0.15, config.DISTANCE_THRESHOLD - 0.04 * depth)
        agg = AgglomerativeClustering(
            n_clusters=None, distance_threshold=stricter,
            metric="cosine", linkage="complete",
        )
        sub_labels = agg.fit_predict(cluster_embs)

        for sub_id in sorted(set(sub_labels)):
            sub_indices = np.where(sub_labels == sub_id)[0]
            if len(sub_indices) >= config.MIN_CLUSTER_SIZE:
                refined[indices[sub_indices]] = next_label
                next_label += 1
            else:
                refined[indices[sub_indices]] = config.NOISE_LABEL

    if did_refine:
        unique2 = sorted(set(refined) - {config.NOISE_LABEL})
        remap = {old: new for new, old in enumerate(unique2)}
        refined = np.array([remap.get(l, config.NOISE_LABEL) for l in refined])
        refined = _recursive_refine(embeddings, refined, depth + 1, max_depth, max_size)

    return refined


def _reassign_noise(embeddings, labels, threshold=0.60):
    noise_idx = np.where(labels == config.NOISE_LABEL)[0]
    if len(noise_idx) == 0:
        return labels

    unique = sorted(set(labels) - {config.NOISE_LABEL})
    if not unique:
        return labels

    centroids = {}
    for cid in unique:
        idx = np.where(labels == cid)[0]
        centroids[cid] = embeddings[idx].mean(axis=0)

    centroid_matrix = normalize(np.array([centroids[c] for c in unique]))
    noise_embs = normalize(embeddings[noise_idx])
    sims = cosine_similarity(noise_embs, centroid_matrix)

    result = labels.copy()
    reassigned = 0
    for i, ni in enumerate(noise_idx):
        best_sim = sims[i].max()
        best_cidx = sims[i].argmax()
        if best_sim > threshold:
            result[ni] = unique[best_cidx]
            reassigned += 1

    log.info("  Reassigned %d noise points", reassigned)
    return result


def _merge_similar(embeddings, labels, threshold=0.96):
    unique = sorted(set(labels) - {config.NOISE_LABEL})
    if len(unique) <= 1:
        return labels

    centroids = {}
    for cid in unique:
        idx = np.where(labels == cid)[0]
        centroids[cid] = embeddings[idx].mean(axis=0)

    centroid_matrix = normalize(np.array([centroids[c] for c in unique]))
    sim_matrix = cosine_similarity(centroid_matrix)

    merge_map = {}
    for i, c1 in enumerate(unique):
        for j, c2 in enumerate(unique):
            if i >= j:
                continue
            if sim_matrix[i, j] > threshold:
                s1, s2 = np.sum(labels == c1), np.sum(labels == c2)
                smaller, larger = (c1, c2) if s1 < s2 else (c2, c1)
                if smaller not in merge_map and larger not in merge_map:
                    merge_map[smaller] = larger

    result = labels.copy()
    for c_from, c_to in merge_map.items():
        result[result == c_from] = c_to

    if merge_map:
        new_unique = sorted(set(result) - {config.NOISE_LABEL})
        remap = {old: new for new, old in enumerate(new_unique)}
        final = np.full_like(result, config.NOISE_LABEL)
        for old, new in remap.items():
            final[result == old] = new
        log.info("  Merged %d pairs -> %d final clusters", len(merge_map), len(new_unique))
        return final

    return labels


# ══════════════════════════════════════════════════════════════════
# STAGE 4: NAMING — LLM with graceful fallback
# ══════════════════════════════════════════════════════════════════

NAMING_SYSTEM = """You are an expert mobile app market analyst naming competitive sub-niches.

Rules:
- niche_name: 2-4 words, VERY specific (e.g., "AI Headshot Generators", NOT "Photo Apps")
- niche_description: 1-2 sentences explaining what these apps do and for whom
- Be GRANULAR: "Keto Diet Trackers" NOT "Health Apps", "AI Meeting Notetakers" NOT "Productivity Tools"
- Output ONLY valid JSON: {"niche_name": "...", "niche_description": "..."}"""

NAMING_BATCH_TEMPLATE = """Name each cluster with a specific sub-niche.

For each cluster return JSON:
{{"cluster_id": N, "niche_name": "...", "niche_description": "..."}}

Clusters:
{clusters}

Return JSON array for all clusters."""


def _extract_keywords(df, labels, cluster_id):
    indices = np.where(labels == cluster_id)[0]
    if len(indices) == 0:
        return [], [], []

    stop = {"app", "free", "pro", "plus", "best", "new", "mobile", "the", "and", "for", "with",
            "your", "you", "from", "all", "get", "one", "also", "can", "use", "make", "like",
            "more", "most", "just", "very", "easy", "simple", "create", "using", "this", "that"}

    names = df.iloc[indices]["trackName"].tolist()
    feats = " ".join(df.iloc[indices]["features_text"].tolist()).lower()

    name_words = [w for w in re.findall(r"[a-z]{3,}", " ".join(names).lower()) if w not in stop]
    feat_words = [w for w in re.findall(r"[a-z]{3,}", feats) if w not in stop]

    combined = Counter()
    for w, c in Counter(name_words).items():
        combined[w] += c * 3
    for w, c in Counter(feat_words).items():
        combined[w] += c * 2

    return [w for w, _ in combined.most_common(30)], names[:5], [w for w, _ in Counter(feat_words).most_common(10)]


NICHE_KEYWORDS = {
    "headshot": "AI Headshot Generators", "avatar": "AI Avatar Creators", "logo": "AI Logo Makers",
    "invoice": "Invoice & Billing Apps", "fasting": "Intermittent Fasting Trackers", "keto": "Keto Diet Trackers",
    "meeting": "AI Meeting Notetakers", "notetaker": "AI Meeting Notetakers", "transcription": "Transcription Apps",
    "cleaner": "Phone Storage Cleaners", "storage": "Phone Storage Cleaners", "vpn": "VPN Services",
    "password": "Password Managers", "meditation": "Meditation & Mindfulness Apps", "sleep": "Sleep Tracking Apps",
    "workout": "Workout & Fitness Apps", "fitness": "Fitness Tracking Apps", "yoga": "Yoga Apps",
    "running": "Running & Jogging Apps", "dating": "Dating Apps", "pregnancy": "Pregnancy Trackers",
    "period": "Period Tracking Apps", "interior": "AI Interior Design Apps", "design": "Design & Creative Apps",
    "language": "Language Learning Apps", "translator": "Translation Apps", "weather": "Weather Apps",
    "calendar": "Calendar & Planner Apps", "scanner": "Document Scanner Apps", "pdf": "PDF Tools",
    "photo": "Photo Editing Apps", "video": "Video Editing Apps", "music": "Music & Audio Apps",
    "podcast": "Podcast Apps", "budget": "Budget & Expense Tracking",
    "recipe": "Recipe & Cooking Apps", "meal": "Meal Planning Apps",
    "therapy": "Therapy & Mental Health Apps", "anxiety": "Anxiety & Stress Relief Apps",
    "browser": "Web Browser Apps", "widget": "Widget & Customization",
    "wallpaper": "Wallpaper Apps", "keyboard": "Keyboard Apps", "battery": "Battery & Optimization Apps",
    "recorder": "Audio & Video Recording Apps", "writing": "AI Writing Assistants",
    "alarm": "Alarm & Reminder Apps", "timer": "Timer Apps",
    "notes": "Note-Taking Apps", "notepad": "Note-Taking Apps",
    "study": "Study & Flashcard Apps", "flashcard": "Flashcard Apps",
    "drawing": "Drawing & Sketch Apps", "horoscope": "Horoscope & Astrology Apps",
    "pet": "Pet Care Apps", "dog": "Dog Care Apps", "cat": "Cat Care Apps",
    "gps": "GPS & Navigation Apps", "navigation": "GPS & Navigation Apps",
    "streaming": "Streaming Apps", "book": "Book & Reading Apps", "audiobook": "Audiobook Apps",
    "news": "News & Magazine Apps", "chess": "Chess Apps", "puzzle": "Puzzle Games",
    "paint": "Painting Apps", "ringtone": "Ringtone Apps",
    "plant": "Plant Care & Identification", "garden": "Gardening Apps",
    "parking": "Parking Apps", "dialer": "Phone & Dialer Apps",
    "screen": "Screen Recording Apps", "hair": "Hair & Beauty Apps",
    "beauty": "Beauty & Makeup Apps", "makeup": "Makeup Apps",
    "habit": "Habit Tracker Apps", "journal": "Journal & Diary Apps",
    "diary": "Journal & Diary Apps", "calorie": "Calorie Counter Apps",
    "water": "Water Intake Tracker Apps", "prayer": "Prayer & Religious Apps",
    "bible": "Bible & Religious Apps", "mental": "Mental Health Apps",
    "driving": "Driving & Navigation Apps", "identif": "Nature Identification Apps",
}


def _smart_label(keywords, names, feat_words):
    names_str = " ".join(names).lower()

    for kw in keywords[:10]:
        if kw.lower() in NICHE_KEYWORDS:
            niche_name = NICHE_KEYWORDS[kw.lower()]
            return {"niche_name": niche_name, "niche_description": f"Apps for {niche_name.lower()}, including {', '.join(n for n in names[:3])}."}

    name_words_all = []
    for name in names:
        for w in re.findall(r"[A-Za-z]{3,}", name):
            if w.lower() not in {"app", "free", "pro", "plus", "best", "new", "mobile", "the", "and", "for", "with"}:
                name_words_all.append(w)
    name_ctr = Counter(w.lower() for w in name_words_all)
    for word, count in name_ctr.most_common(3):
        if count >= max(2, len(names) * 0.3):
            return {"niche_name": f"{word.title()} Apps", "niche_description": f"Apps centered around {word.lower()}, including {', '.join(n for n in names[:3])}."}

    if keywords:
        return {"niche_name": f"{keywords[0].title()} Apps", "niche_description": f"Apps focused on {keywords[0].lower()}, including {', '.join(n for n in names[:3])}."}

    return {"niche_name": names[0] + " & Similar" if names else "Unknown Niche", "niche_description": f"A group of mobile apps."}


def name_clusters(df, labels, use_llm=True, naming_batch_size=10):
    """Name clusters. Falls back to rule-based if LLM unavailable or fails."""
    if use_llm and _llm_available():
        log.info("Attempting LLM naming via %s (batch_size=%d)...", config.LLM_MODEL_NAME, naming_batch_size)
        try:
            result = _name_clusters_llm(df, labels, batch_size=naming_batch_size)
            if result:
                return result
        except Exception as e:
            log.warning("LLM naming failed entirely: %s. Falling back to rule-based.", str(e)[:200])

    log.info("Using rule-based naming for all clusters.")
    return name_clusters_rule_based(df, labels)


def _name_one_cluster_batch(df, labels, batch_ids):
    """Build prompt and call LLM for one naming batch. Returns (batch_ids, result_list_or_none)."""
    batch_lines = []
    for cid in batch_ids:
        indices = np.where(labels == cid)[0]
        if len(indices) == 0:
            continue
        reps = _get_reps(df, labels, cid, n=5)
        apps = ", ".join([n for n, d in reps[:5]])
        batch_lines.append("Cluster %s: %s" % (cid, apps))

    prompt = NAMING_BATCH_TEMPLATE.format(clusters="\n".join(batch_lines))
    result = llm_call_json(prompt, NAMING_SYSTEM)
    return batch_ids, result


def _name_clusters_llm(df, labels, batch_size=10):
    """Name clusters with LLM using batching and parallel API calls (JSON array)."""
    unique = sorted(set(labels) - {config.NOISE_LABEL})
    cluster_info = {}
    llm_success = 0
    llm_fail = 0

    batches = [unique[i : i + batch_size] for i in range(0, len(unique), batch_size)]
    workers = min(config.LLM_NAMING_CONCURRENCY, max(1, len(batches)))

    log.info(
        "Naming %d clusters with LLM (batch_size=%d, concurrency=%d)...",
        len(unique),
        batch_size,
        workers,
    )

    def _apply_batch(batch_ids, result):
        nonlocal llm_success, llm_fail
        if result and isinstance(result, list):
            for item in result:
                if isinstance(item, dict) and "niche_name" in item:
                    cluster_id = item.get("cluster_id")
                    if cluster_id is not None and cluster_id in batch_ids:
                        cluster_info[cluster_id] = {
                            "niche_name": item["niche_name"],
                            "niche_description": item.get("niche_description", ""),
                        }
                        llm_success += 1
        else:
            for cid in batch_ids:
                if cid not in cluster_info:
                    kws, names, feats = _extract_keywords(df, labels, cid)
                    cluster_info[cid] = _smart_label(kws, names, feats)
                    llm_fail += 1

    if workers <= 1:
        for batch_ids in batches:
            batch_ids, result = _name_one_cluster_batch(df, labels, batch_ids)
            _apply_batch(batch_ids, result)
            done = len(cluster_info)
            log.info(
                "  Named %d/%d clusters (LLM: %d, rule: %d)",
                done,
                len(unique),
                llm_success,
                llm_fail,
            )
    else:
        batch_results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_name_one_cluster_batch, df, labels, bid) for bid in batches]
            for fut in as_completed(futures):
                batch_results.append(fut.result())
        for batch_ids, result in batch_results:
            _apply_batch(batch_ids, result)
            done = len(cluster_info)
            log.info(
                "  Named %d/%d clusters (LLM: %d, rule: %d)",
                done,
                len(unique),
                llm_success,
                llm_fail,
            )

    for cid in unique:
        if cid not in cluster_info:
            kws, names, feats = _extract_keywords(df, labels, cid)
            cluster_info[cid] = _smart_label(kws, names, feats)

    log.info("LLM naming complete: %d LLM, %d rule-based", llm_success, llm_fail)
    return cluster_info


def _get_reps(df, labels, cluster_id, n=8):
    indices = np.where(labels == cluster_id)[0]
    if len(indices) == 0:
        return []

    emb_cache = config.CACHE_DIR / "embeddings.npy"
    if emb_cache.exists():
        all_embs = np.load(str(emb_cache))
        if all_embs.shape[0] == len(df):
            cluster_embs = all_embs[indices]
            centroid = cluster_embs.mean(axis=0)
            dists = np.linalg.norm(cluster_embs - centroid, axis=1)
            closest = indices[np.argsort(dists)[:n]]
            return [(df.iloc[i]["trackName"], str(df.iloc[i].get("overview_clean", ""))[:150]) for i in closest]

    return [(df.iloc[i]["trackName"], str(df.iloc[i].get("overview_clean", ""))[:150]) for i in indices[:n]]


def name_clusters_rule_based(df, labels):
    unique = sorted(set(labels) - {config.NOISE_LABEL})
    cluster_info = {}
    for cid in unique:
        kws, names, feats = _extract_keywords(df, labels, cid)
        cluster_info[cid] = _smart_label(kws, names, feats)
    return cluster_info


# ══════════════════════════════════════════════════════════════════
# STAGE 5: METRICS
# ══════════════════════════════════════════════════════════════════

def compute_metrics(embeddings, labels):
    mask = labels != config.NOISE_LABEL
    inlier_embs = embeddings[mask]
    inlier_labels = labels[mask]
    unique = sorted(set(inlier_labels))

    if len(unique) < 2:
        return {"n_clusters": len(unique), "n_noise": int((labels == config.NOISE_LABEL).sum())}

    sil = silhouette_score(inlier_embs, inlier_labels, metric="cosine", sample_size=min(5000, len(inlier_embs)))
    sizes = [int(np.sum(inlier_labels == c)) for c in unique]

    sample = unique[:40] if len(unique) > 40 else unique
    sims = []
    for cid in sample:
        idx = np.where(labels == cid)[0]
        if len(idx) < 2:
            continue
        sim_mat = cosine_similarity(embeddings[idx])
        np.fill_diagonal(sim_mat, 0)
        sims.append(float(sim_mat.sum() / (len(idx) * (len(idx) - 1))))

    return {
        "n_clusters": len(unique),
        "n_noise": int((labels == config.NOISE_LABEL).sum()),
        "noise_ratio": round(float((labels == config.NOISE_LABEL).sum() / len(labels)), 3),
        "silhouette": round(float(sil), 4),
        "avg_intra_similarity": round(float(np.mean(sims)), 4) if sims else 0,
        "median_cluster_size": float(np.median(sizes)),
        "cluster_size_distribution": {"min": int(min(sizes)), "max": int(max(sizes)), "mean": round(float(np.mean(sizes)), 1)},
    }


# ══════════════════════════════════════════════════════════════════
# STAGE 6: EXPORT
# ══════════════════════════════════════════════════════════════════

def export_results(df, labels, cluster_info):
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    unique = sorted(set(labels) - {config.NOISE_LABEL})
    niches = []
    for cid in unique:
        indices = np.where(labels == cid)[0]
        info = cluster_info.get(cid, {"niche_name": f"Niche_{cid}", "niche_description": ""})
        niches.append({
            "niche_name": info["niche_name"],
            "niche_description": info["niche_description"],
            "competitors": df.iloc[indices]["trackName"].tolist(),
            "metadata": {"cluster_size": int(len(indices))},
        })

    noise_idx = np.where(labels == config.NOISE_LABEL)[0]
    if len(noise_idx) > 0:
        niches.append({
            "niche_name": "Unclassified / Unique Apps",
            "niche_description": "Apps that could not be confidently assigned to a specific competitive niche.",
            "competitors": df.iloc[noise_idx]["trackName"].tolist(),
            "metadata": {"cluster_size": int(len(noise_idx))},
        })

    with open(config.RESULT_DIR / "niches.json", "w", encoding="utf-8") as f:
        json.dump({"niches": niches}, f, ensure_ascii=False, indent=2)
    log.info("Saved niches.json (%d niches)", len(niches))

    rows = []
    for cid in unique:
        info = cluster_info.get(cid, {"niche_name": f"Niche_{cid}"})
        for idx in np.where(labels == cid)[0]:
            rows.append({"AppName": df.iloc[idx]["trackName"], "Description": str(df.iloc[idx].get("overview_clean", ""))[:200], "SubNicheName": info["niche_name"]})
    for idx in noise_idx:
        rows.append({"AppName": df.iloc[idx]["trackName"], "Description": str(df.iloc[idx].get("overview_clean", ""))[:200], "SubNicheName": "Unclassified"})
    csv_path = config.RESULT_DIR / "app_niche_mapping.csv"
    for attempt in range(3):
        try:
            pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
            log.info("Saved app_niche_mapping.csv")
            break
        except PermissionError:
            if attempt < 2:
                log.warning("CSV file locked, retrying in 2s...")
                time.sleep(2)
            else:
                log.error("Could not save CSV (file locked). Skipping.")

    embeddings = np.load(str(config.CACHE_DIR / "embeddings.npy"))
    metrics = compute_metrics(embeddings, labels)
    with open(config.RESULT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    log.info("Saved metrics.json")
    return metrics


def create_visualization(df, labels, cluster_info):
    import umap
    import plotly.express as px

    log.info("Creating UMAP visualization...")
    embeddings = np.load(str(config.CACHE_DIR / "embeddings.npy"))

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(embeddings)

    name_map = {}
    for cid in sorted(set(labels) - {config.NOISE_LABEL}):
        info = cluster_info.get(cid, {"niche_name": f"Cluster {cid}"})
        name_map[cid] = info["niche_name"]

    plot_df = pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1],
        "trackName": df["trackName"].values,
        "niche_name": [name_map.get(l, "Noise") for l in labels],
    })

    fig = px.scatter(plot_df, x="x", y="y", color="niche_name", hover_data=["trackName"],
                     title="App Segmentation — UMAP 2D", width=1600, height=1000)
    fig.update_traces(marker=dict(size=3, opacity=0.7))
    fig.update_layout(showlegend=True, legend=dict(font=dict(size=6)), legend_title=dict(font=dict(size=8)))
    fig.write_html(str(config.RESULT_DIR / "umap_clusters.html"))
    log.info("Saved umap_clusters.html")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="App Segmentation Pipeline")
    parser.add_argument("--no-llm-enrichment", action="store_true", help="Skip LLM enrichment stage")
    parser.add_argument("--no-llm-naming", action="store_true", help="Skip LLM naming, use rule-based")
    parser.add_argument("--no-llm", action="store_true", help="Skip ALL LLM calls (force rule-based)")
    parser.add_argument("--enrichment-batch-size", type=int, default=20, help="Number of apps per LLM enrichment batch (default: 20)")
    parser.add_argument("--naming-batch-size", type=int, default=10, help="Number of clusters per LLM naming batch (default: 10)")
    parser.add_argument("--input", type=str, default=str(config.DATA_FILE), help="Input JSON file")
    args = parser.parse_args()

    # --no-llm overrides both enrichment and naming flags
    if args.no_llm:
        args.no_llm_enrichment = True
        args.no_llm_naming = True
        global _llm_disabled
        _llm_disabled = True

    start = time.time()
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    config.print_config()

    llm_status = "DISABLED (flag)" if args.no_llm else ("AVAILABLE" if config.has_llm_access() else "NOT AVAILABLE (no API key)")
    log.info("=" * 60)
    log.info("APP SEGMENTATION PIPELINE")
    log.info("  LLM: %s", llm_status)
    log.info("=" * 60)

    # Stage 0: Data prep
    df = load_data(Path(args.input))
    df = preprocess(df)

    # Stage 1: LLM enrichment (with batching)
    if not args.no_llm_enrichment:
        enrichments = llm_enrich_apps(df, batch_size=args.enrichment_batch_size)
        # Count how many got enrichment
        enriched_count = sum(1 for e in enrichments if e is not None)
        if enriched_count == 0:
            log.info("No enrichment data obtained. Using text-only embeddings.")
        else:
            log.info("Enriched %d/%d apps. Updating canonical texts.", enriched_count, len(df))
            # Invalidate embeddings cache since canonical text changed
            emb_cache = config.CACHE_DIR / "embeddings.npy"
            if emb_cache.exists():
                log.info("Removing stale embeddings cache (enrichment changed canonical text)")
                emb_cache.unlink()
    else:
        enrichments = [None] * len(df)
        log.info("LLM enrichment skipped (--no-llm-enrichment flag).")

    df["canonical_text"] = df.apply(
        lambda r: build_canonical_text(r, enrichments[r.name] if r.name < len(enrichments) else None),
        axis=1,
    )

    # Stage 2: Embedding (with fallback)
    embeddings = compute_embeddings(df)

    # Stage 3: Clustering
    labels = cluster_pipeline(embeddings)

    # Stage 4: Naming (with fallback)
    if args.no_llm_naming:
        cluster_info = name_clusters_rule_based(df, labels)
    else:
        cluster_info = name_clusters(df, labels, use_llm=True, naming_batch_size=args.naming_batch_size)

    # Stage 5-6: Metrics + Export
    metrics = export_results(df, labels, cluster_info)
    log.info("=" * 40)
    log.info("QUALITY METRICS")
    log.info("=" * 40)
    for k, v in metrics.items():
        log.info("  %s: %s", k, v if not isinstance(v, dict) else v)

    create_visualization(df, labels, cluster_info)

    elapsed = time.time() - start
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE in %.1f seconds (%.1f min)", elapsed, elapsed / 60)
    log.info("Results: %s", config.RESULT_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()