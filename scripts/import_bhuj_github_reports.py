"""
Selectively import Bhuj 2001 report PDFs from a public GitHub repository.

Source:
https://github.com/yashky25-cloud/bhuj-2001

The script:
- checks the repository metadata
- detects the default branch automatically
- verifies which allowlisted PDF files actually exist
- downloads only verified allowlisted PDFs
- saves reports separately from image datasets
- writes provenance metadata to CSV
- never overwrites non-empty files

Uses only the Python standard library.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------
# Repository configuration
# ---------------------------------------------------------------------

REPO_OWNER = "yashky25-cloud"
REPO_NAME = "bhuj-2001"

REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}"
REPO_API = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"

SOURCE_COLLECTION = "GitHub_yashky25-cloud_bhuj-2001"

DOWNLOAD_DELAY_SEC = 0.75

USER_AGENT = (
    "Mozilla/5.0 "
    "(compatible; seismic-damage-research/0.1; "
    "+internal research/testing)"
)


# ---------------------------------------------------------------------
# Local project paths
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = (
    REPO_ROOT
    / "data"
    / "reports"
    / "bhuj_2001_github"
)

ANNOTATIONS_DIR = REPO_ROOT / "data" / "annotations"

CSV_PATH = (
    ANNOTATIONS_DIR
    / "bhuj_2001_github_sources.csv"
)


# ---------------------------------------------------------------------
# Explicit allowlist
# Only files in this list can be downloaded.
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ReportSpec:
    filename: str
    usefulness: str
    notes: str
    local_name: str | None = None


ALLOWLIST: tuple[ReportSpec, ...] = (

    ReportSpec(
        filename="report-eefit-bhuj-india-20190814 (1).pdf",
        local_name="report-eefit-bhuj-india-20190814.pdf",
        usefulness="high",
        notes=(
            "EEFIT field report with observations on building "
            "performance and earthquake damage."
        ),
    ),

    ReportSpec(
        filename="india_bhuj_eeri_preliminary_report.pdf",
        usefulness="high",
        notes=(
            "EERI preliminary reconnaissance report with observations "
            "on building failures and earthquake damage."
        ),
    ),

    ReportSpec(
        filename="2749_EQRBhuj.pdf",
        usefulness="high",
        notes="Bhuj earthquake reconnaissance-style reference report.",
    ),

    ReportSpec(
        filename="2713_Other200104GujaratEQTR0101.pdf",
        usefulness="high",
        notes=(
            "Gujarat earthquake observations supporting damage "
            "and disaster context."
        ),
    ),

    ReportSpec(
        filename="Bhuj-quake2001.pdf",
        usefulness="high",
        notes="Bhuj 2001 earthquake overview and context report.",
    ),

    ReportSpec(
        filename="india.pdf",
        usefulness="medium",
        notes=(
            "India/Bhuj earthquake compilation useful for "
            "background and contextual reference."
        ),
    ),

    ReportSpec(
        filename="paper028.pdf",
        usefulness="medium",
        notes="Research paper retained for literature context.",
    ),

    ReportSpec(
        filename="Sairam2018-SiteEffectsintheAhmedabad.pdf",
        usefulness="medium",
        notes=(
            "Ahmedabad site-effects research useful for "
            "regional seismic context."
        ),
    ),

    ReportSpec(
        filename="IntensityBasedCasualtyModelsCaseStudy.pdf",
        usefulness="medium",
        notes=(
            "Intensity and casualty modeling case study; "
            "secondary contextual material."
        ),
    ),

    ReportSpec(
        filename="India_Bhuj_Recovery_Nov03.pdf",
        usefulness="low",
        notes="Recovery-focused background document.",
    ),

    ReportSpec(
        filename="13_2042.pdf",
        usefulness="low",
        notes="Archived source document with opaque filename.",
    ),
)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    """Make a filename safe for Windows."""

    name = name.replace("\\", "_").replace("/", "_")
    name = re.sub(r'[<>:"|?*]', "_", name)
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"_+", "_", name)

    return name


def fetch_json(url: str):
    """Fetch JSON from a public HTTPS endpoint."""

    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/vnd.github+json",
        },
    )

    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_bytes(url: str) -> bytes:
    """Download raw bytes."""

    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/octet-stream",
        },
    )

    with urlopen(request, timeout=120) as response:
        return response.read()


def get_repository_metadata() -> dict:
    """Get repository metadata including default branch."""

    return fetch_json(REPO_API)


def get_repository_tree(branch: str) -> list[dict]:
    """
    Get the repository tree recursively.

    Returns all tracked files that GitHub exposes through
    the Git Trees API.
    """

    encoded_branch = quote(branch, safe="")

    url = (
        f"{REPO_API}/git/trees/"
        f"{encoded_branch}?recursive=1"
    )

    data = fetch_json(url)

    tree = data.get("tree")

    if not isinstance(tree, list):
        raise ValueError("GitHub API did not return a valid repository tree.")

    return tree


def raw_url(branch: str, repo_path: str) -> str:
    """Build a raw GitHub download URL."""

    encoded_path = "/".join(
        quote(part, safe="")
        for part in repo_path.split("/")
    )

    return (
        f"https://raw.githubusercontent.com/"
        f"{REPO_OWNER}/{REPO_NAME}/"
        f"{quote(branch, safe='')}/"
        f"{encoded_path}"
    )


def find_allowlisted_files(
    tree: list[dict],
) -> dict[str, str]:
    """
    Match allowlisted filenames against actual repository files.

    Returns:
        {filename_lower: actual_repository_path}
    """

    pdf_files: dict[str, str] = {}

    for item in tree:

        if item.get("type") != "blob":
            continue

        path = item.get("path")

        if not isinstance(path, str):
            continue

        if not path.lower().endswith(".pdf"):
            continue

        filename = Path(path).name.lower()

        # Keep first occurrence of each filename.
        pdf_files.setdefault(filename, path)

    matches: dict[str, str] = {}

    for spec in ALLOWLIST:

        key = spec.filename.lower()

        if key in pdf_files:
            matches[key] = pdf_files[key]

    return matches


def build_rows(
    branch: str,
    matches: dict[str, str],
) -> list[dict[str, str]]:
    """Build provenance rows only for verified files."""

    rows: list[dict[str, str]] = []

    index = 1

    for spec in ALLOWLIST:

        key = spec.filename.lower()

        repo_path = matches.get(key)

        if repo_path is None:
            continue

        local_name = sanitize_filename(
            spec.local_name or Path(repo_path).name
        )

        rows.append(
            {
                "document_id": f"bhuj_github_{index:03d}",
                "filename": local_name,
                "source_collection": SOURCE_COLLECTION,
                "source_repository": REPO_URL,
                "source_branch": branch,
                "source_repo_path": repo_path,
                "source_raw_url": raw_url(branch, repo_path),
                "usefulness": spec.usefulness,
                "notes": spec.notes,
                "local_relative_path": (
                    f"data/reports/"
                    f"bhuj_2001_github/"
                    f"{local_name}"
                ),
            }
        )

        index += 1

    return rows


def write_provenance_csv(
    rows: list[dict[str, str]],
    path: Path,
) -> None:
    """Write source provenance CSV."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "document_id",
        "filename",
        "source_collection",
        "source_repository",
        "source_branch",
        "source_repo_path",
        "source_raw_url",
        "usefulness",
        "notes",
        "local_relative_path",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:

    print("Bhuj 2001 GitHub report importer")
    print()

    # Safety check: reports must never go into image directories.
    forbidden_dirs = [
        REPO_ROOT / "data" / "raw" / "images" / "bhuj_2001",
        REPO_ROOT / "data" / "raw" / "images" / "bhuj_2001_slideshare",
        REPO_ROOT / "data" / "raw" / "images" / "bhuj_2001_iitk_rc",
    ]

    for forbidden in forbidden_dirs:

        if REPORTS_DIR.resolve() == forbidden.resolve():

            print(
                "ERROR: Refusing to write reports "
                "into an image dataset directory.",
                file=sys.stderr,
            )

            return 1

    # -------------------------------------------------------------
    # Get repository metadata
    # -------------------------------------------------------------

    try:

        print(f"Checking repository:")
        print(f"  {REPO_URL}")

        metadata = get_repository_metadata()

        branch = metadata.get("default_branch")

        if not isinstance(branch, str) or not branch:

            raise ValueError(
                "Could not determine repository default branch."
            )

        print(f"Default branch: {branch}")

    except (
        HTTPError,
        URLError,
        TimeoutError,
        OSError,
        ValueError,
    ) as exc:

        print(
            f"ERROR: Failed to inspect repository: {exc}",
            file=sys.stderr,
        )

        return 1

    # -------------------------------------------------------------
    # Inspect repository tree
    # -------------------------------------------------------------

    try:

        print()
        print("Inspecting repository file tree...")

        tree = get_repository_tree(branch)

        matches = find_allowlisted_files(tree)

    except (
        HTTPError,
        URLError,
        TimeoutError,
        OSError,
        ValueError,
    ) as exc:

        print(
            f"ERROR: Failed to inspect repository files: {exc}",
            file=sys.stderr,
        )

        return 1

    # -------------------------------------------------------------
    # Show verification results
    # -------------------------------------------------------------

    print()
    print("Allowlist verification:")

    for spec in ALLOWLIST:

        key = spec.filename.lower()

        repo_path = matches.get(key)

        if repo_path:

            print(f"  FOUND    {repo_path}")

        else:

            print(f"  MISSING  {spec.filename}")

    rows = build_rows(branch, matches)

    if not rows:

        print(
            "ERROR: None of the allowlisted PDFs were found.",
            file=sys.stderr,
        )

        return 1

    # -------------------------------------------------------------
    # Prepare directories and CSV
    # -------------------------------------------------------------

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_provenance_csv(
        rows,
        CSV_PATH,
    )

    print()
    print(
        f"Verified documents: {len(rows)}"
    )

    print(
        f"Target directory: "
        f"{REPORTS_DIR.relative_to(REPO_ROOT)}"
    )

    print(
        f"Provenance CSV: "
        f"{CSV_PATH.relative_to(REPO_ROOT)}"
    )

    # -------------------------------------------------------------
    # Download verified reports
    # -------------------------------------------------------------

    downloaded = 0
    skipped = 0
    failed = 0

    print()
    print("Downloading verified reports...")

    for i, row in enumerate(rows):

        destination = REPORTS_DIR / row["filename"]

        # Never overwrite existing non-empty file.
        if (
            destination.exists()
            and destination.stat().st_size > 0
        ):

            print(
                f"  skip  {row['filename']} "
                "(already present)"
            )

            skipped += 1
            continue

        temp_path = destination.with_suffix(
            destination.suffix + ".part"
        )

        try:

            data = fetch_bytes(
                row["source_raw_url"]
            )

            if not data:

                raise ValueError(
                    "Downloaded response is empty."
                )

            # Basic PDF validation.
            if not data.startswith(b"%PDF"):

                raise ValueError(
                    "Downloaded file does not appear "
                    "to be a PDF."
                )

            temp_path.write_bytes(data)

            # Atomic rename after successful validation.
            temp_path.replace(destination)

            print(
                f"  saved {row['filename']} "
                f"({len(data):,} bytes) "
                f"[{row['usefulness']}]"
            )

            downloaded += 1

        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            ValueError,
        ) as exc:

            print(
                f"  FAIL  {row['filename']}: {exc}",
                file=sys.stderr,
            )

            if temp_path.exists():

                temp_path.unlink(
                    missing_ok=True
                )

            failed += 1

        if i < len(rows) - 1:

            time.sleep(DOWNLOAD_DELAY_SEC)

    # -------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------

    print()
    print(
        f"Done. "
        f"downloaded={downloaded} "
        f"skipped={skipped} "
        f"failed={failed} "
        f"total={len(rows)}"
    )

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())