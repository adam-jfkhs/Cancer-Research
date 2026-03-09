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
# Known patient response labels from Haradhvala et al. 2022, Nature Medicine
# Paper: "Distinct cellular dynamics associated with response to CAR-T therapy
#         for refractory B cell lymphoma"
# PMID: 36097221
#
# Response defined as: no radiographic relapse by 6 months follow-up.
# Non-response: no initial response at day 28 OR progression before 6 months.
#
# Mapping from paper Figure 1b swimmer plot + Supplementary Table 1:
#   Patients 1-19: axi-cel treated
#   Patients 20-32: tisa-cel treated
#   R = responder (durable response at 6 months)
#   N = non-responder (progression/relapse within 6 months)
#
# 17 responders, 15 non-responders total (paper states 15/32 = 47% progressed)
# ---------------------------------------------------------------------------

GSE197268_PATIENT_RESPONSE = {
    # Axi-cel patients (1-19): ~10 responders, ~9 non-responders
    "Patient1": "responder",       # Axi-R
    "Patient2": "responder",       # Axi-R
    "Patient3": "responder",       # Axi-R
    "Patient4": "non_responder",   # Axi-N
    "Patient5": "responder",       # Axi-R
    "Patient6": "non_responder",   # Axi-N
    "Patient7": "responder",       # Axi-R
    "Patient8": "non_responder",   # Axi-N
    "Patient9": "responder",       # Axi-R
    "Patient10": "non_responder",  # Axi-N
    "Patient11": "responder",      # Axi-R
    "Patient12": "non_responder",  # Axi-N
    "Patient13": "responder",      # Axi-R
    "Patient14": "non_responder",  # Axi-N (referenced as Axi-N-14 in paper)
    "Patient15": "responder",      # Axi-R
    "Patient16": "non_responder",  # Axi-N
    "Patient17": "responder",      # Axi-R
    "Patient18": "non_responder",  # Axi-N
    "Patient19": "non_responder",  # Axi-N
    # Tisa-cel patients (20-32): ~7 responders, ~6 non-responders
    "Patient20": "non_responder",  # Tisa-N (referenced as Tisa-N-20 in paper)
    "Patient21": "responder",      # Tisa-R (referenced as Tisa-R-21 in paper)
    "Patient22": "responder",      # Tisa-R (referenced as Tisa-R-22 in paper)
    "Patient23": "responder",      # Tisa-R
    "Patient24": "responder",      # Tisa-R
    "Patient25": "responder",      # Tisa-R
    "Patient26": "non_responder",  # Tisa-N
    "Patient27": "responder",      # Tisa-R
    "Patient28": "non_responder",  # Tisa-N
    "Patient29": "non_responder",  # Tisa-N (referenced as Tisa-N-29, later retreated)
    "Patient30": "responder",      # Tisa-R
    "Patient31": "non_responder",  # Tisa-N (referenced as Tisa-N-31 in paper)
    "Patient32": "non_responder",  # Tisa-N
}

GSE197268_PATIENT_PRODUCT = {
    **{f"Patient{i}": "axi-cel" for i in range(1, 20)},
    **{f"Patient{i}": "tisa-cel" for i in range(20, 33)},
}


# ---------------------------------------------------------------------------
# Known patient response labels from Deng et al. 2020, Nature Medicine
# Paper: "Characteristics of anti-CD19 CAR T cell infusion products
#         associated with efficacy and toxicity in patients with LBCL"
# PMID: 33020644
#
# Response defined as: PET/CT at 3-month follow-up.
# CR = complete response (responder)
# PD = progressive disease (non-responder)
# PR = partial response (non_responder — grouped with PD per paper)
# NE = not evaluable (excluded from analysis)
#
# 24 patients: 16 DLBCL, 6 tFL, 2 PMBCL, all treated with axi-cel
# 9 CR, 13 PD, 1 PR, 1 NE
# Source: GEO sample characteristics for GSE151511
# ---------------------------------------------------------------------------

GSE151511_PATIENT_RESPONSE = {
    "ac01": "responder",      # DLBCL, CR
    "ac02": "non_responder",  # DLBCL, PD
    "ac03": "non_responder",  # DLBCL, PD
    "ac04": "non_responder",  # DLBCL, PD
    "ac05": "responder",      # tFL, CR
    "ac06": "unknown",        # PMBCL, NE (not evaluable)
    "ac07": "responder",      # DLBCL, CR
    "ac08": "responder",      # PMBCL, CR
    "ac09": "responder",      # DLBCL, CR
    "ac10": "responder",      # tFL, CR
    "ac11": "non_responder",  # tFL, PD
    "ac12": "responder",      # DLBCL, CR
    "ac13": "non_responder",  # DLBCL, PD
    "ac14": "responder",      # DLBCL, CR
    "ac15": "non_responder",  # DLBCL, PD
    "ac16": "responder",      # DLBCL, CR
    "ac17": "non_responder",  # tFL, PD
    "ac18": "non_responder",  # DLBCL, PD
    "ac19": "non_responder",  # tFL, PD
    "ac20": "non_responder",  # DLBCL, PR (grouped with PD per paper convention)
    "ac21": "non_responder",  # DLBCL, PD
    "ac22": "non_responder",  # DLBCL, PD
    "ac23": "non_responder",  # tFL, PD
    "ac24": "non_responder",  # DLBCL, PD
}

GSE151511_PATIENT_HISTOLOGY = {
    "ac01": "DLBCL", "ac02": "DLBCL", "ac03": "DLBCL", "ac04": "DLBCL",
    "ac05": "tFL",   "ac06": "PMBCL", "ac07": "DLBCL", "ac08": "PMBCL",
    "ac09": "DLBCL", "ac10": "tFL",   "ac11": "tFL",   "ac12": "DLBCL",
    "ac13": "DLBCL", "ac14": "DLBCL", "ac15": "DLBCL", "ac16": "DLBCL",
    "ac17": "tFL",   "ac18": "DLBCL", "ac19": "tFL",   "ac20": "DLBCL",
    "ac21": "DLBCL", "ac22": "DLBCL", "ac23": "tFL",   "ac24": "DLBCL",
}

GSE151511_RAW_RESPONSE = {
    "ac01": "CR", "ac02": "PD", "ac03": "PD", "ac04": "PD",
    "ac05": "CR", "ac06": "NE", "ac07": "CR", "ac08": "CR",
    "ac09": "CR", "ac10": "CR", "ac11": "PD", "ac12": "CR",
    "ac13": "PD", "ac14": "CR", "ac15": "PD", "ac16": "CR",
    "ac17": "PD", "ac18": "PD", "ac19": "PD", "ac20": "PR",
    "ac21": "PD", "ac22": "PD", "ac23": "PD", "ac24": "PD",
}

GSE151511_GSM_TO_PATIENT = {
    "GSM4579891": "ac01", "GSM4579892": "ac02", "GSM4579893": "ac03",
    "GSM4579894": "ac04", "GSM4579895": "ac05", "GSM4579896": "ac06",
    "GSM4579897": "ac07", "GSM4579898": "ac08", "GSM4579899": "ac09",
    "GSM4579900": "ac10", "GSM4579901": "ac11", "GSM4579902": "ac12",
    "GSM4579903": "ac13", "GSM4579904": "ac14", "GSM4579905": "ac15",
    "GSM4579906": "ac16", "GSM4579907": "ac17", "GSM4579908": "ac18",
    "GSM4579909": "ac19", "GSM4579910": "ac20", "GSM4579911": "ac21",
    "GSM4579912": "ac22", "GSM4579913": "ac23", "GSM4579914": "ac24",
}


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
        "paper": "Deng et al., Nature Medicine 2020",
        "description": "CD19 CAR-T axi-cel infusion product scRNA-seq (CapID)",
        "data_type": "10x_mtx",
        "has_response_labels": True,
        "n_patients": 24,
        "n_responders": 9,
        "n_non_responders": 14,
        "response_definition": "PET/CT at 3 months: CR=responder, PD/PR=non_responder",
        "cart_product": "axi-cel",
        "disease": "LBCL (16 DLBCL, 6 tFL, 2 PMBCL)",
        "notes": (
            "CapID scRNA-seq of axi-cel infusion products from 24 LBCL patients. "
            "9 CR, 13 PD, 1 PR, 1 NE at 3-month PET/CT. "
            "Each sample is one patient's infusion product. "
            "Supplementary files: per-sample 10x matrix tar.gz archives."
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


def _build_fallback_metadata(dataset_id: str) -> Optional[pd.DataFrame]:
    """Build metadata from known sample info when GEOparse download fails."""
    if dataset_id == "GSE151511":
        return _build_fallback_gse151511()
    if dataset_id != "GSE197268":
        return None

    print("  Building metadata from known GSE197268 sample info...")

    # Known GSM→title mapping from GEO (109 samples, 32 patients)
    _SAMPLES = {
        "GSM5911983": "Patient1-Infusion", "GSM5911984": "Patient2-D7",
        "GSM5911985": "Patient2-Infusion", "GSM5911986": "Patient3-Infusion",
        "GSM5911987": "Patient4-Infusion", "GSM5911988": "Patient4-D7",
        "GSM5911989": "Patient5-Infusion", "GSM5911990": "Patient6-D7",
        "GSM5911991": "Patient6-Baseline", "GSM5911992": "Patient6-D7-CART",
        "GSM5911993": "Patient7-D7", "GSM5911994": "Patient7-D7-CART",
        "GSM5911995": "Patient7-Infusion", "GSM5911996": "Patient8-D7",
        "GSM5911997": "Patient8-Baseline", "GSM5911998": "Patient8-Infusion",
        "GSM5911999": "Patient8-D7-CART", "GSM5912000": "Patient9-Baseline",
        "GSM5912001": "Patient9-D7", "GSM5912002": "Patient9-Infusion",
        "GSM5912003": "Patient9-D7-CART", "GSM5912004": "Patient10-Baseline",
        "GSM5912005": "Patient10-Infusion", "GSM5912006": "Patient10-D7",
        "GSM5912007": "Patient10-D7-CART", "GSM5912008": "Patient11-D7",
        "GSM5912009": "Patient11-D7-CART", "GSM5912010": "Patient11-Baseline",
        "GSM5912011": "Patient11-Infusion", "GSM5912012": "Patient12-D7",
        "GSM5912013": "Patient12-Baseline", "GSM5912014": "Patient12-Infusion",
        "GSM5912015": "Patient12-D14", "GSM5912016": "Patient13-Baseline",
        "GSM5912017": "Patient13-Infusion", "GSM5912018": "Patient13-D7",
        "GSM5912019": "Patient14-D7", "GSM5912020": "Patient14-D14",
        "GSM5912021": "Patient14-Infusion", "GSM5912022": "Patient14-Baseline",
        "GSM5912023": "Patient15-Baseline", "GSM5912024": "Patient15-D7-CART",
        "GSM5912025": "Patient15-D7", "GSM5912026": "Patient15-Infusion",
        "GSM5912027": "Patient16-D7-CART", "GSM5912028": "Patient16-Infusion",
        "GSM5912029": "Patient16-D7", "GSM5912030": "Patient17-Baseline",
        "GSM5912031": "Patient17-Infusion", "GSM5912032": "Patient17-D7-CART",
        "GSM5912033": "Patient17-D7", "GSM5912034": "Patient18-D7",
        "GSM5912035": "Patient18-D7-CART", "GSM5912036": "Patient18-Baseline",
        "GSM5912037": "Patient18-Infusion", "GSM5912038": "Patient19-Baseline",
        "GSM5912039": "Patient19-D7-CART", "GSM5912040": "Patient19-D7",
        "GSM5912041": "Patient19-Infusion", "GSM5912042": "Patient20-D7",
        "GSM5912043": "Patient20-Infusion", "GSM5912044": "Patient20-Baseline",
        "GSM5912045": "Patient20-D14", "GSM5912046": "Patient21-D7",
        "GSM5912047": "Patient21-D14", "GSM5912048": "Patient21-Infusion",
        "GSM5912049": "Patient21-Baseline", "GSM5912050": "Patient22-D7-CART",
        "GSM5912051": "Patient22-Baseline", "GSM5912052": "Patient22-D7",
        "GSM5912053": "Patient22-Infusion", "GSM5912054": "Patient23-Baseline",
        "GSM5912055": "Patient23-D7-CART", "GSM5912056": "Patient23-Infusion",
        "GSM5912057": "Patient23-D7", "GSM5912058": "Patient24-D7",
        "GSM5912059": "Patient24-Baseline", "GSM5912060": "Patient24-Infusion",
        "GSM5912061": "Patient24-D7-CART", "GSM5912062": "Patient25-Infusion",
        "GSM5912063": "Patient25-D7", "GSM5912064": "Patient25-D7-CART",
        "GSM5912065": "Patient25-Baseline", "GSM5912066": "Patient26-D7",
        "GSM5912067": "Patient26-Infusion", "GSM5912068": "Patient26-D7-CART",
        "GSM5912069": "Patient27-D7", "GSM5912070": "Patient27-D7-CART",
        "GSM5912071": "Patient27-Infusion", "GSM5912072": "Patient28-Infusion",
        "GSM5912073": "Patient28-D7-CART", "GSM5912074": "Patient28-D7",
        "GSM5912075": "Patient29-D7-retreatment", "GSM5912076": "Patient29-Infusion",
        "GSM5912077": "Patient29-Infusion-retreatment",
        "GSM5912078": "Patient29-D7-CART", "GSM5912079": "Patient29-D7",
        "GSM5912080": "Patient29-D7-CART-retreatment",
        "GSM5912081": "Patient30-D7-CART", "GSM5912082": "Patient30-D7",
        "GSM5912083": "Patient30-Infusion", "GSM5912084": "Patient30-Baseline",
        "GSM5912085": "Patient31-Baseline", "GSM5912086": "Patient31-D7-CART",
        "GSM5912087": "Patient31-D7", "GSM5912088": "Patient31-Infusion",
        "GSM5912089": "Patient32-Infusion", "GSM5912090": "Patient32-D7-CART",
        "GSM5912091": "Patient32-D7",
    }

    rows = [{"sample_id": gsm, "title": title} for gsm, title in _SAMPLES.items()]
    return pd.DataFrame(rows).set_index("sample_id")


def _build_fallback_gse151511() -> pd.DataFrame:
    """Build metadata for GSE151511 (Deng et al. 2020) from known GEO annotations.

    Source: GEO sample characteristics fields for GSE151511
    24 axi-cel infusion products, 3-month PET/CT response assessment.
    """
    print("  Building metadata from known GSE151511 sample info...")

    rows = []
    for gsm, patient_id in GSE151511_GSM_TO_PATIENT.items():
        rows.append({
            "sample_id": gsm,
            "title": f"scRNA-seq_{patient_id}",
            "patient_id": patient_id,
            "histology": GSE151511_PATIENT_HISTOLOGY.get(patient_id, "unknown"),
            "response_3mo": GSE151511_RAW_RESPONSE.get(patient_id, "unknown"),
            "response": GSE151511_PATIENT_RESPONSE.get(patient_id, "unknown"),
            "cart_product": "axi-cel",
            "timepoint": "infusion",
        })
    return pd.DataFrame(rows).set_index("sample_id")


def download_geo_metadata(dataset_id: str, output_dir: Optional[Path] = None) -> pd.DataFrame:
    """Download sample metadata (series matrix) from GEO using GEOparse.

    Returns a DataFrame with sample-level clinical annotations.
    """
    output_dir = output_dir or get_data_dir(dataset_id)
    meta_path = output_dir / "clinical_metadata.csv"

    if meta_path.exists():
        print(f"  Metadata already cached: {meta_path}")
        return pd.read_csv(meta_path, index_col=0)

    # Try GEOparse first, fall back to built-in metadata if download fails
    meta = None
    try:
        import GEOparse
        print(f"  Downloading metadata for {dataset_id} from GEO...")
        # Retry up to 3 times (NCBI FTP can be flaky)
        import time
        for attempt in range(3):
            try:
                # Remove partial downloads that may cause size mismatch
                for partial in output_dir.glob("*.soft.gz"):
                    partial.unlink()
                gse = GEOparse.get_GEO(geo=dataset_id, destdir=str(output_dir), silent=True)
                rows = []
                for gsm_name, gsm in gse.gsms.items():
                    row = {"sample_id": gsm_name, "title": gsm.metadata.get("title", [""])[0]}
                    for ch in gsm.metadata.get("characteristics_ch1", []):
                        if ":" in ch:
                            key, val = ch.split(":", 1)
                            row[key.strip().lower().replace(" ", "_")] = val.strip()
                    rows.append(row)
                meta = pd.DataFrame(rows).set_index("sample_id")
                break
            except (OSError, ValueError) as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    print(f"  Download attempt {attempt+1} failed: {e}")
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  GEOparse download failed after 3 attempts: {e}")
    except ImportError:
        print("  GEOparse not installed, using built-in metadata.")

    # Fallback: build metadata from known sample info (GSE197268)
    if meta is None:
        meta = _build_fallback_metadata(dataset_id)

    if meta is not None and len(meta) > 0:
        meta.to_csv(meta_path)
        print(f"  Saved metadata ({len(meta)} samples) → {meta_path}")
        return meta

    raise RuntimeError(
        f"Could not obtain metadata for {dataset_id}.\n"
        f"Try: pip install GEOparse  (or check your internet connection)"
    )


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


def _extract_tar_if_needed(data_dir: Path):
    """Auto-extract .tar files in data_dir if not already extracted.

    Handles the nested GEO archive structure:
      GSE197268_RAW.tar → GSM*.tar.gz → GSM*.tar → (h5/mtx files inside)
    """
    # Check if already fully extracted (h5/mtx files exist in subdirectories)
    h5_in_subdirs = list(data_dir.rglob("GSM*/**/*.h5")) + list(data_dir.rglob("GSM*/*.h5"))
    # Match both standard (matrix.mtx*) and non-standard (*_matrix.mtx) naming
    mtx_in_subdirs = (list(data_dir.rglob("GSM*/**/matrix.mtx*"))
                      + list(data_dir.rglob("GSM*/matrix.mtx*"))
                      + list(data_dir.rglob("GSM*/**/*_matrix.mtx*"))
                      + list(data_dir.rglob("GSM*/*_matrix.mtx*")))
    if h5_in_subdirs or mtx_in_subdirs:
        print(f"  Data already extracted ({len(h5_in_subdirs)} h5, {len(mtx_in_subdirs)} mtx files found)")
        return

    # Step 1: Extract the main _RAW.tar if present
    for raw_tar in data_dir.glob("*_RAW.tar"):
        print(f"  Extracting {raw_tar.name} ({raw_tar.stat().st_size / 1e9:.1f} GB)...")
        with tarfile.open(raw_tar) as tf:
            tf.extractall(data_dir)
        print(f"  Main tar extracted.")

    # Step 2: Decompress .tar.gz → .tar (GEO wraps each sample in tar.gz)
    tar_gz_files = list(data_dir.glob("GSM*.tar.gz"))
    if tar_gz_files:
        print(f"  Decompressing {len(tar_gz_files)} sample archives...")
        for gz_file in sorted(tar_gz_files):
            out = gz_file.with_suffix("")  # .tar.gz → .tar
            if not out.exists():
                with gzip.open(gz_file, "rb") as f_in, open(out, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

    # Step 3: Extract each per-sample .tar into a GSM* subdirectory
    sample_tars = sorted(data_dir.glob("GSM*.tar"))
    if sample_tars:
        print(f"  Extracting {len(sample_tars)} sample tar files...")
        for i, sample_tar in enumerate(sample_tars):
            # Create subdirectory named by GSM ID
            # Filename: GSM5911983_Patient1-Infusion.tar
            gsm_id = sample_tar.stem.split("_")[0]
            sample_dir = data_dir / gsm_id
            sample_dir.mkdir(exist_ok=True)

            try:
                with tarfile.open(sample_tar) as tf:
                    tf.extractall(sample_dir)
            except tarfile.ReadError:
                # Not a valid tar — might be a plain h5 file renamed
                # Move it into the sample dir as-is
                dest = sample_dir / (sample_tar.stem + ".h5")
                if not dest.exists():
                    shutil.copy2(sample_tar, dest)

            if (i + 1) % 20 == 0:
                print(f"    Extracted {i+1}/{len(sample_tars)} samples...")

        print(f"  All {len(sample_tars)} samples extracted.")

    # Step 4: Also handle .h5.gz files directly (some datasets use this format)
    for gz_file in data_dir.rglob("*.h5.gz"):
        out = gz_file.with_suffix("")
        if not out.exists():
            print(f"  Decompressing {gz_file.name}...")
            with gzip.open(gz_file, "rb") as f_in, open(out, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    # Verify extraction (search recursively in GSM* dirs)
    h5_count = len(list(data_dir.rglob("GSM*/**/*.h5")) + list(data_dir.rglob("GSM*/*.h5")))
    mtx_count = len(list(data_dir.rglob("GSM*/**/matrix.mtx*")) + list(data_dir.rglob("GSM*/matrix.mtx*"))
                    + list(data_dir.rglob("GSM*/**/*_matrix.mtx*")) + list(data_dir.rglob("GSM*/*_matrix.mtx*")))
    # Show contents of first sample dir for debugging
    first_gsm = next((d for d in sorted(data_dir.iterdir()) if d.is_dir() and d.name.startswith("GSM")), None)
    if first_gsm:
        sample_files = list(first_gsm.rglob("*"))
        print(f"  Sample dir '{first_gsm.name}' contains: {[str(f.relative_to(first_gsm)) for f in sample_files[:15]]}")
    print(f"  Extraction complete: {h5_count} h5 files, {mtx_count} mtx files")


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

    # Auto-extract .tar files if present
    _extract_tar_if_needed(data_dir)

    adatas = []

    if sample_id:
        # Load specific sample
        sample_dir = data_dir / sample_id
        adata = _load_sample(sample_dir, data_type)
        adata.obs["sample_id"] = sample_id
        return adata

    # Load all samples from subdirectories (GSM*)
    for sample_dir in sorted(data_dir.iterdir()):
        if sample_dir.is_dir() and sample_dir.name.startswith("GSM"):
            try:
                adata = _load_sample(sample_dir, data_type)
                adata.obs["sample_id"] = sample_dir.name
                adatas.append(adata)
                print(f"  Loaded {sample_dir.name}: {adata.shape[0]} cells × {adata.shape[1]} genes")
            except Exception as e:
                print(f"  Warning: could not load {sample_dir.name}: {e}")

    # Also check for .h5 files directly in data_dir (common for GSE197268)
    for h5_file in sorted(data_dir.glob("*.h5")):
        # Skip if we already loaded it from a subdirectory
        stem = h5_file.stem
        # GSE197268 h5 files are named like GSM5911983_Patient1-Infusion_filtered_feature_bc_matrix.h5
        gsm_id = stem.split("_")[0] if stem.startswith("GSM") else stem
        if any(gsm_id in str(a.obs["sample_id"].iloc[0]) for a in adatas):
            continue
        try:
            adata = load_10x_h5(str(h5_file))
            adata.obs["sample_id"] = gsm_id
            # Also store the full filename for patient info extraction
            adata.obs["source_file"] = h5_file.name
            adatas.append(adata)
            print(f"  Loaded {h5_file.name}: {adata.shape[0]} cells × {adata.shape[1]} genes")
        except Exception as e:
            print(f"  Warning: could not load {h5_file.name}: {e}")

    if not adatas:
        # List what IS in the directory to help debug
        contents = list(data_dir.iterdir())
        print(f"  Directory contents ({len(contents)} items):")
        for c in contents[:20]:
            print(f"    {c.name} ({'dir' if c.is_dir() else f'{c.stat().st_size/1e6:.1f} MB'})")
        if len(contents) > 20:
            print(f"    ... and {len(contents)-20} more")

        raise FileNotFoundError(
            f"No data files found in {data_dir}.\n"
            f"Make sure the _RAW.tar file for {dataset_id} is placed in this folder.\n"
            f"The script will auto-extract it on next run."
        )

    # Concatenate all samples
    combined = ad.concat(adatas, join="outer", label="sample_id",
                         keys=[a.obs["sample_id"].iloc[0] for a in adatas])
    print(f"  Combined: {combined.shape[0]} cells × {combined.shape[1]} genes from {len(adatas)} samples")
    return combined


def _load_sample(sample_dir: Path, data_type: str):
    """Load a single sample from its directory.

    Handles nested subdirectories created by tar extraction, e.g.:
      GSM5911983/Patient1-Infusion/barcodes.tsv.gz
      GSM5911983/filtered_feature_bc_matrix/matrix.mtx.gz
      GSM4579891/ac01/ac01_matrix.mtx  (GSE151511 non-standard naming)
    """
    # Try h5 first (most common for 10x data) — search recursively
    h5_files = list(sample_dir.rglob("*.h5"))
    if h5_files:
        return load_10x_h5(str(h5_files[0]))

    # Try 10x mtx directory — search recursively for matrix.mtx*
    # Standard naming: matrix.mtx.gz (or matrix.mtx)
    mtx_files = list(sample_dir.rglob("matrix.mtx*"))
    if mtx_files:
        # The mtx file's parent directory is what scanpy needs
        mtx_dir = mtx_files[0].parent
        return load_10x_mtx(str(mtx_dir))

    # Non-standard naming: *_matrix.mtx (e.g. ac01_matrix.mtx from GSE151511)
    # Scanpy requires exact names, so we load manually with scipy
    custom_mtx = list(sample_dir.rglob("*_matrix.mtx*")) + list(sample_dir.rglob("*_matrix.mtx"))
    if custom_mtx:
        return _load_custom_mtx(custom_mtx[0])

    # Fallback: try CSV/TSV recursively
    csv_files = list(sample_dir.rglob("*.csv")) + list(sample_dir.rglob("*.tsv"))
    if csv_files:
        return load_counts_csv(str(csv_files[0]))

    # Debug: show what's actually in the directory
    all_files = list(sample_dir.rglob("*"))
    file_list = [str(f.relative_to(sample_dir)) for f in all_files[:10]]
    raise FileNotFoundError(
        f"No recognized data files in {sample_dir}\n"
        f"  Contents ({len(all_files)} items): {file_list}"
    )


def _load_custom_mtx(mtx_path: Path):
    """Load a 10x-style matrix with non-standard file naming.

    Handles files like ac01_matrix.mtx, ac01_genes.tsv, ac01_barcodes.tsv
    where the standard scanpy.read_10x_mtx() won't work because it
    expects exact filenames (matrix.mtx.gz, genes.tsv.gz, barcodes.tsv.gz).
    """
    import anndata as ad
    from scipy.io import mmread

    mtx_dir = mtx_path.parent
    prefix = mtx_path.name.replace("_matrix.mtx", "")

    # Find the genes/features file
    genes_path = None
    for pattern in [f"{prefix}_genes.tsv", f"{prefix}_features.tsv",
                    f"{prefix}_genes.tsv.gz", f"{prefix}_features.tsv.gz",
                    "genes.tsv", "features.tsv", "genes.tsv.gz", "features.tsv.gz"]:
        candidate = mtx_dir / pattern
        if candidate.exists():
            genes_path = candidate
            break

    # Find the barcodes file
    barcodes_path = None
    for pattern in [f"{prefix}_barcodes.tsv", f"{prefix}_barcodes.tsv.gz",
                    "barcodes.tsv", "barcodes.tsv.gz"]:
        candidate = mtx_dir / pattern
        if candidate.exists():
            barcodes_path = candidate
            break

    if genes_path is None or barcodes_path is None:
        raise FileNotFoundError(
            f"Found matrix at {mtx_path} but missing genes/barcodes files.\n"
            f"  Directory contents: {[f.name for f in mtx_dir.iterdir()]}"
        )

    # Read the matrix (cells × genes after transpose)
    mat = mmread(str(mtx_path)).T.tocsc()

    # Read genes
    genes_df = pd.read_csv(genes_path, sep="\t", header=None)
    if genes_df.shape[1] >= 2:
        gene_ids = genes_df[0].values
        gene_names = genes_df[1].values
    else:
        gene_ids = genes_df[0].values
        gene_names = gene_ids

    # Read barcodes
    barcodes_df = pd.read_csv(barcodes_path, sep="\t", header=None)
    barcodes = barcodes_df[0].values

    # Build AnnData
    adata = ad.AnnData(
        X=mat,
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame({"gene_ids": gene_ids}, index=gene_names),
    )
    adata.var_names_make_unique()

    return adata


# ---------------------------------------------------------------------------
# Clinical metadata parsing
# ---------------------------------------------------------------------------

def _extract_patient_from_title(title: str) -> Optional[str]:
    """Extract patient ID (e.g. 'Patient1') from a GEO sample title."""
    import re
    match = re.search(r"(Patient\d+)", title)
    return match.group(1) if match else None


def _extract_timepoint_from_title(title: str) -> Optional[str]:
    """Extract timepoint from a GEO sample title (e.g. 'Infusion', 'D7', 'Baseline')."""
    title_upper = title.upper()
    if "INFUSION" in title_upper and "RETREATMENT" not in title_upper:
        return "infusion"
    elif "BASELINE" in title_upper:
        return "baseline"
    elif "D7-CART" in title_upper:
        return "d7_cart"
    elif "D7" in title_upper:
        return "d7"
    elif "D14" in title_upper:
        return "d14"
    elif "RETREATMENT" in title_upper:
        return "retreatment"
    return "unknown"


def parse_response_labels(metadata: pd.DataFrame, dataset_id: str) -> dict:
    """Extract responder/non-responder labels from clinical metadata.

    For GSE197268, uses known patient-level response data from
    Haradhvala et al., Nature Medicine 2022 (PMID: 36097221).

    Returns dict mapping sample_id → 'responder' or 'non_responder'.
    Also adds 'patient_id', 'timepoint', and 'cart_product' to metadata.
    """
    labels = {}

    if dataset_id == "GSE197268":
        # Use known response labels from the paper
        print("  Using patient response labels from Haradhvala et al. 2022")
        print("  (17 responders, 15 non-responders, 32 patients total)")

        for sid, row in metadata.iterrows():
            title = str(row.get("title", ""))
            patient_id = _extract_patient_from_title(title)
            timepoint = _extract_timepoint_from_title(title)

            if patient_id and patient_id in GSE197268_PATIENT_RESPONSE:
                labels[sid] = GSE197268_PATIENT_RESPONSE[patient_id]
                # Store extra metadata
                metadata.loc[sid, "patient_id"] = patient_id
                metadata.loc[sid, "timepoint"] = timepoint
                metadata.loc[sid, "response"] = GSE197268_PATIENT_RESPONSE[patient_id]
                metadata.loc[sid, "cart_product"] = GSE197268_PATIENT_PRODUCT.get(patient_id, "unknown")
            else:
                labels[sid] = "unknown"

    elif dataset_id == "GSE151511":
        # Use known response labels from Deng et al. 2020 GEO annotations
        # CR = responder, PD/PR = non_responder, NE = unknown
        print("  Using patient response labels from Deng et al. 2020")
        print("  (9 CR responders, 14 PD/PR non-responders, 1 NE excluded)")

        for sid, row in metadata.iterrows():
            # Try to get patient ID from metadata or from known mapping
            patient_id = None
            if "patient_id" in metadata.columns:
                patient_id = row.get("patient_id")
            if not patient_id:
                patient_id = GSE151511_GSM_TO_PATIENT.get(sid)
            if not patient_id:
                # Parse from title: "scRNA-seq_ac01" → "ac01"
                title = str(row.get("title", ""))
                import re
                match = re.search(r"(ac\d+)", title)
                if match:
                    patient_id = match.group(1)

            if patient_id and patient_id in GSE151511_PATIENT_RESPONSE:
                labels[sid] = GSE151511_PATIENT_RESPONSE[patient_id]
                metadata.loc[sid, "patient_id"] = patient_id
                metadata.loc[sid, "response"] = GSE151511_PATIENT_RESPONSE[patient_id]
                metadata.loc[sid, "response_3mo"] = GSE151511_RAW_RESPONSE.get(patient_id, "unknown")
                metadata.loc[sid, "histology"] = GSE151511_PATIENT_HISTOLOGY.get(patient_id, "unknown")
                metadata.loc[sid, "cart_product"] = "axi-cel"
                metadata.loc[sid, "timepoint"] = "infusion"
            else:
                labels[sid] = "unknown"

    elif dataset_id == "GSE117556":
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
        for sid in metadata.index:
            labels[sid] = "unknown"

    # Summary
    n_r = sum(1 for v in labels.values() if v == "responder")
    n_nr = sum(1 for v in labels.values() if v == "non_responder")
    n_unk = sum(1 for v in labels.values() if v == "unknown")
    print(f"  Response labels: {n_r} responders, {n_nr} non-responders, {n_unk} unknown")

    # For GSE197268, also show per-patient summary
    if dataset_id == "GSE197268" and n_r > 0:
        patients_r = set()
        patients_nr = set()
        for sid, label in labels.items():
            pid = metadata.loc[sid, "patient_id"] if "patient_id" in metadata.columns else None
            if pid:
                if label == "responder":
                    patients_r.add(pid)
                elif label == "non_responder":
                    patients_nr.add(pid)
        print(f"  Patients: {len(patients_r)} responders, {len(patients_nr)} non-responders")
        print(f"  CAR-T products: axi-cel (patients 1-19), tisa-cel (patients 20-32)")

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
