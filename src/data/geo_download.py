"""Download and prepare real clinical CAR-T scRNA-seq data from GEO.

Supports:
  - GSE151511: CD19 CAR-T longitudinal scRNA-seq (Deng et al.)
  - GSE197268: Large CAR-T cohort with outcomes (Haradhvala et al., Nature Med 2022)
  - GSE117556: CAR-T in CLL (Fraietta et al., Nature Med 2018)

Usage:
    python -m src.data.geo_download --dataset GSE197268
    python -m src.data.geo_download --all
"""

import os
import gzip
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

DATASETS = {
    "GSE197268": {
        "name": "Haradhvala2022",
        "paper": "Haradhvala et al., Nature Medicine 2022",
        "description": "Large-scale CAR-T determinants of response",
        "data_type": "10x_h5",
        "has_response_labels": True,
        "notes": (
            "This dataset contains scRNA-seq from CAR-T infusion products "
            "with clinical response annotations (CR/PR vs NR/PD). "
            "Download the processed count matrices from GEO supplementary files."
        ),
    },
    "GSE151511": {
        "name": "Deng2020",
        "paper": "Deng et al., 2020",
        "description": "CD19 CAR-T longitudinal scRNA-seq",
        "data_type": "10x_mtx",
        "has_response_labels": True,
        "notes": (
            "Pre- and post-infusion CAR-T cells with paired clinical data. "
            "Download the raw 10x matrix files from GEO."
        ),
    },
    "GSE117556": {
        "name": "Fraietta2018",
        "paper": "Fraietta et al., Nature Medicine 2018",
        "description": "CD19 CAR-T in CLL",
        "data_type": "counts_table",
        "has_response_labels": True,
        "notes": (
            "Leukapheresis and infusion product characterization. "
            "Bulk + single-cell data available."
        ),
    },
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def get_data_dir(dataset_id: str) -> Path:
    """Return (and create) the local data directory for a dataset."""
    base = Path(__file__).resolve().parents[2] / "data" / "raw" / dataset_id
    base.mkdir(parents=True, exist_ok=True)
    return base


def download_geo_metadata(dataset_id: str, output_dir: Optional[Path] = None) -> pd.DataFrame:
    """Download sample metadata (series matrix) from GEO using GEOparse.

    Returns a DataFrame with sample-level clinical annotations.
    """
    output_dir = output_dir or get_data_dir(dataset_id)
    meta_path = output_dir / "clinical_metadata.csv"

    if meta_path.exists():
        print(f"  Metadata already cached: {meta_path}")
        return pd.read_csv(meta_path, index_col=0)

    try:
        import GEOparse
    except ImportError:
        raise ImportError(
            "GEOparse is required to download GEO metadata.\n"
            "Install it with:  pip install GEOparse"
        )

    print(f"  Downloading metadata for {dataset_id} from GEO...")
    gse = GEOparse.get_GEO(geo=dataset_id, destdir=str(output_dir), silent=True)

    # Extract sample-level metadata
    rows = []
    for gsm_name, gsm in gse.gsms.items():
        row = {"sample_id": gsm_name, "title": gsm.metadata.get("title", [""])[0]}
        # Flatten characteristics
        for ch in gsm.metadata.get("characteristics_ch1", []):
            if ":" in ch:
                key, val = ch.split(":", 1)
                row[key.strip().lower().replace(" ", "_")] = val.strip()
        rows.append(row)

    meta = pd.DataFrame(rows).set_index("sample_id")
    meta.to_csv(meta_path)
    print(f"  Saved metadata ({len(meta)} samples) → {meta_path}")
    return meta


def download_geo_supplementary(dataset_id: str, output_dir: Optional[Path] = None) -> Path:
    """Download supplementary (count matrix) files from GEO.

    Uses the NCBI FTP to grab the _RAW.tar or individual .h5 / .mtx.gz files.
    """
    output_dir = output_dir or get_data_dir(dataset_id)
    marker = output_dir / ".download_complete"

    if marker.exists():
        print(f"  Data already downloaded: {output_dir}")
        return output_dir

    try:
        import GEOparse
    except ImportError:
        raise ImportError(
            "GEOparse is required. Install with:  pip install GEOparse"
        )

    print(f"  Downloading supplementary files for {dataset_id}...")
    gse = GEOparse.get_GEO(geo=dataset_id, destdir=str(output_dir), silent=True)

    # Try to download supplementary files
    for url in gse.metadata.get("supplementary_file", []):
        fname = url.split("/")[-1]
        dest = output_dir / fname
        if not dest.exists():
            print(f"    Downloading {fname}...")
            _download_url(url, dest)

    # Also try per-sample supplementary files
    for gsm_name, gsm in gse.gsms.items():
        sample_dir = output_dir / gsm_name
        sample_dir.mkdir(exist_ok=True)
        for url in gsm.metadata.get("supplementary_file", []):
            fname = url.split("/")[-1]
            dest = sample_dir / fname
            if not dest.exists():
                print(f"    Downloading {gsm_name}/{fname}...")
                _download_url(url, dest)

    # Extract any tar files
    for tar_file in output_dir.glob("*.tar"):
        print(f"    Extracting {tar_file.name}...")
        with tarfile.open(tar_file) as tf:
            tf.extractall(output_dir)

    # Decompress .gz files (keep originals)
    for gz_file in output_dir.rglob("*.gz"):
        out = gz_file.with_suffix("")
        if not out.exists():
            with gzip.open(gz_file, "rb") as f_in, open(out, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    marker.touch()
    print(f"  Download complete → {output_dir}")
    return output_dir


def _download_url(url: str, dest: Path):
    """Download a file from URL to dest path."""
    import urllib.request
    urllib.request.urlretrieve(url, str(dest))


# ---------------------------------------------------------------------------
# Loading real data into AnnData
# ---------------------------------------------------------------------------

def load_10x_h5(path: str):
    """Load a 10x Genomics .h5 file into AnnData."""
    import scanpy as sc
    return sc.read_10x_h5(path)


def load_10x_mtx(path: str):
    """Load a 10x Genomics matrix directory (matrix.mtx, genes.tsv, barcodes.tsv)."""
    import scanpy as sc
    return sc.read_10x_mtx(path)


def load_counts_csv(path: str):
    """Load a CSV/TSV count matrix (genes × cells) into AnnData."""
    import anndata as ad
    df = pd.read_csv(path, index_col=0, sep=None, engine="python")
    # Convention: rows=genes, cols=cells → transpose to cells×genes
    if df.shape[0] < df.shape[1]:
        df = df.T
    return ad.AnnData(X=df.values.astype(np.float32),
                      obs=pd.DataFrame(index=df.index),
                      var=pd.DataFrame(index=df.columns))


def load_real_dataset(dataset_id: str, sample_id: Optional[str] = None):
    """Load a real GEO dataset into AnnData.

    Parameters
    ----------
    dataset_id : str
        GEO accession (e.g. 'GSE197268').
    sample_id : str, optional
        Specific GSM sample to load. If None, loads all available.

    Returns
    -------
    anndata.AnnData
        Raw count matrix ready for preprocessing.
    """
    import anndata as ad

    data_dir = get_data_dir(dataset_id)
    info = DATASETS.get(dataset_id, {})
    data_type = info.get("data_type", "10x_mtx")

    adatas = []

    if sample_id:
        # Load specific sample
        sample_dir = data_dir / sample_id
        adata = _load_sample(sample_dir, data_type)
        adata.obs["sample_id"] = sample_id
        return adata

    # Load all samples
    for sample_dir in sorted(data_dir.iterdir()):
        if sample_dir.is_dir() and sample_dir.name.startswith("GSM"):
            try:
                adata = _load_sample(sample_dir, data_type)
                adata.obs["sample_id"] = sample_dir.name
                adatas.append(adata)
                print(f"  Loaded {sample_dir.name}: {adata.shape[0]} cells × {adata.shape[1]} genes")
            except Exception as e:
                print(f"  Warning: could not load {sample_dir.name}: {e}")

    # Also check for .h5 files directly in data_dir
    for h5_file in sorted(data_dir.glob("*.h5")):
        try:
            adata = load_10x_h5(str(h5_file))
            adata.obs["sample_id"] = h5_file.stem
            adatas.append(adata)
            print(f"  Loaded {h5_file.name}: {adata.shape[0]} cells × {adata.shape[1]} genes")
        except Exception as e:
            print(f"  Warning: could not load {h5_file.name}: {e}")

    if not adatas:
        raise FileNotFoundError(
            f"No data files found in {data_dir}.\n"
            f"Run download first:  python -m src.data.geo_download --dataset {dataset_id}"
        )

    # Concatenate all samples
    combined = ad.concat(adatas, join="outer", label="sample_id",
                         keys=[a.obs["sample_id"].iloc[0] for a in adatas])
    print(f"  Combined: {combined.shape[0]} cells × {combined.shape[1]} genes from {len(adatas)} samples")
    return combined


def _load_sample(sample_dir: Path, data_type: str):
    """Load a single sample from its directory."""
    if data_type == "10x_h5":
        h5_files = list(sample_dir.glob("*.h5"))
        if h5_files:
            return load_10x_h5(str(h5_files[0]))
    if data_type == "10x_mtx":
        mtx_files = list(sample_dir.glob("*matrix.mtx*"))
        if mtx_files:
            return load_10x_mtx(str(sample_dir))
    # Fallback: try CSV/TSV
    csv_files = list(sample_dir.glob("*.csv")) + list(sample_dir.glob("*.tsv"))
    if csv_files:
        return load_counts_csv(str(csv_files[0]))
    raise FileNotFoundError(f"No recognized data files in {sample_dir}")


# ---------------------------------------------------------------------------
# Clinical metadata parsing
# ---------------------------------------------------------------------------

def parse_response_labels(metadata: pd.DataFrame, dataset_id: str) -> dict:
    """Extract responder/non-responder labels from clinical metadata.

    Returns dict mapping sample_id → 'responder' or 'non_responder'.
    """
    labels = {}

    # Dataset-specific parsing logic
    if dataset_id == "GSE197268":
        # Haradhvala 2022: look for 'response' or 'best_response' column
        response_col = _find_column(metadata, ["response", "best_response", "clinical_response"])
        if response_col:
            for sid, row in metadata.iterrows():
                val = str(row[response_col]).upper().strip()
                if val in ("CR", "COMPLETE RESPONSE", "PR", "PARTIAL RESPONSE"):
                    labels[sid] = "responder"
                elif val in ("NR", "NO RESPONSE", "PD", "PROGRESSIVE DISEASE", "SD", "STABLE DISEASE"):
                    labels[sid] = "non_responder"
                else:
                    labels[sid] = "unknown"

    elif dataset_id == "GSE151511":
        # Deng 2020: response often in title or characteristics
        response_col = _find_column(metadata, ["response", "outcome", "clinical_response"])
        if response_col:
            for sid, row in metadata.iterrows():
                val = str(row[response_col]).upper().strip()
                if any(r in val for r in ["RESPOND", "CR", "COMPLETE", "PARTIAL", "PR"]):
                    labels[sid] = "responder"
                else:
                    labels[sid] = "non_responder"
        else:
            # Parse from title
            for sid, row in metadata.iterrows():
                title = str(row.get("title", "")).upper()
                if any(r in title for r in ["RESPONDER", "CR", "PR"]):
                    labels[sid] = "responder"
                elif any(r in title for r in ["NON-RESPONDER", "NR", "PD"]):
                    labels[sid] = "non_responder"
                else:
                    labels[sid] = "unknown"

    elif dataset_id == "GSE117556":
        # Fraietta 2018: CLL, response annotations in characteristics
        response_col = _find_column(metadata, ["response", "outcome", "remission"])
        if response_col:
            for sid, row in metadata.iterrows():
                val = str(row[response_col]).upper().strip()
                if any(r in val for r in ["CR", "COMPLETE", "RESPOND", "REMISSION"]):
                    labels[sid] = "responder"
                else:
                    labels[sid] = "non_responder"

    if not labels:
        print("  Warning: Could not auto-detect response labels.")
        print("  You may need to manually annotate from the paper's supplementary tables.")
        # Return all as unknown
        for sid in metadata.index:
            labels[sid] = "unknown"

    # Summary
    n_r = sum(1 for v in labels.values() if v == "responder")
    n_nr = sum(1 for v in labels.values() if v == "non_responder")
    n_unk = sum(1 for v in labels.values() if v == "unknown")
    print(f"  Response labels: {n_r} responders, {n_nr} non-responders, {n_unk} unknown")

    return labels


def _find_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Find a column name matching any of the candidates (case-insensitive)."""
    for col in df.columns:
        for cand in candidates:
            if cand.lower() in col.lower():
                return col
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_instructions():
    """Print step-by-step instructions for getting real data."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          HOW TO GET REAL CLINICAL CAR-T DATA                    ║
╚══════════════════════════════════════════════════════════════════╝

All data is FREE and publicly available from NCBI GEO.

OPTION 1: Automatic download (recommended)
─────────────────────────────────────────────
  pip install GEOparse
  python -m src.data.geo_download --dataset GSE197268

  This will:
    1. Download clinical metadata (patient response, demographics)
    2. Download count matrices (single-cell gene expression)
    3. Parse response labels (responder vs non-responder)
    4. Save everything to data/raw/GSE197268/

OPTION 2: Manual download from GEO website
──────────────────────────────────────────────
  1. Go to: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE197268
  2. Scroll to "Supplementary file" section
  3. Download the count matrix files (.h5 or .tar of .mtx.gz)
  4. Place in: data/raw/GSE197268/
  5. Download "Series Matrix File(s)" for clinical metadata
  6. Place in: data/raw/GSE197268/

OPTION 3: Use GEO2R (browser-based, no coding)
────────────────────────────────────────────────
  1. Go to the GEO page for the dataset
  2. Click "Analyze with GEO2R"
  3. Explore sample groups and metadata interactively

RECOMMENDED DATASETS (in order of priority):
─────────────────────────────────────────────
  1. GSE197268 — Haradhvala et al., Nature Med 2022
     Best dataset: large cohort, clinical outcomes, 10x scRNA-seq
     ~50+ patients, CR/PR/NR labels, infusion product scRNA-seq

  2. GSE151511 — Deng et al., 2020
     Longitudinal: pre/post infusion paired samples
     Smaller but has timepoint data matching our timecourse analysis

  3. GSE117556 — Fraietta et al., Nature Med 2018
     Original CAR-T response study, CLL patients
     Mixed bulk + single-cell, well-cited

FILE SIZES (approximate):
─────────────────────────
  GSE197268: ~2-5 GB (compressed)
  GSE151511: ~500 MB - 1 GB
  GSE117556: ~200-500 MB
""")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download CAR-T clinical data from GEO")
    parser.add_argument("--dataset", type=str, help="GEO accession (e.g. GSE197268)")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--info", action="store_true", help="Print download instructions")
    parser.add_argument("--metadata-only", action="store_true", help="Only download metadata")
    args = parser.parse_args()

    if args.info or (not args.dataset and not args.all):
        print_instructions()
    elif args.all:
        for ds_id in DATASETS:
            print(f"\n{'='*50}")
            print(f"Dataset: {ds_id} — {DATASETS[ds_id]['name']}")
            print(f"{'='*50}")
            meta = download_geo_metadata(ds_id)
            labels = parse_response_labels(meta, ds_id)
            if not args.metadata_only:
                download_geo_supplementary(ds_id)
    else:
        ds_id = args.dataset
        print(f"Dataset: {ds_id}")
        meta = download_geo_metadata(ds_id)
        labels = parse_response_labels(meta, ds_id)
        if not args.metadata_only:
            download_geo_supplementary(ds_id)
