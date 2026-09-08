"""Download Bhuj 2001 GEER photo gallery images for internal research/testing.

Fetches the public india_photo.htm page, extracts jpg/jpeg/png URLs hosted under
the same Bhuj_2001 directory, downloads them with a polite delay, and writes a
CSV provenance file. Does not bypass authentication or access controls.
"""

from __future__ import annotations

import csv
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

SOURCE_PAGE = (
    "https://geerassociation.org/components/com_geer_reports/"
    "geerfiles/Bhuj_2001/india_photo.htm"
)
ALLOWED_DIR_PREFIX = "/components/com_geer_reports/geerfiles/Bhuj_2001/"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DOWNLOAD_DELAY_SEC = 0.75
USER_AGENT = "seismic-damage-research/0.1 (+internal research; respectful crawler)"

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "images" / "bhuj_2001"
ANNOTATIONS_DIR = REPO_ROOT / "data" / "annotations"
CSV_PATH = ANNOTATIONS_DIR / "bhuj_2001_sources.csv"


class GalleryParser(HTMLParser):
    """Collect image src attributes and nearby caption text from table rows."""

    def __init__(self) -> None:
        super().__init__()
        self.entries: list[tuple[str, str]] = []
        self._in_td = False
        self._td_depth = 0
        self._current_img: str | None = None
        self._caption_parts: list[str] = []
        self._collect_caption = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        if tag == "td":
            self._in_td = True
            self._td_depth += 1
            if self._current_img is not None and not self._collect_caption:
                self._collect_caption = True
                self._caption_parts = []
        elif tag == "img" and self._in_td:
            src = attr_map.get("src", "").strip()
            if src and self._current_img is None:
                self._current_img = src
        elif tag == "br" and self._collect_caption:
            self._caption_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._td_depth > 0:
            self._td_depth -= 1
            if self._td_depth == 0:
                self._in_td = False
                if self._collect_caption and self._current_img is not None:
                    caption = _normalize_whitespace("".join(self._caption_parts))
                    self.entries.append((self._current_img, caption))
                    self._current_img = None
                    self._collect_caption = False
                    self._caption_parts = []
        elif tag == "tr":
            # Incomplete row: drop dangling image without a caption cell.
            self._current_img = None
            self._collect_caption = False
            self._caption_parts = []
            self._in_td = False
            self._td_depth = 0

    def handle_data(self, data: str) -> None:
        if self._collect_caption:
            self._caption_parts.append(data)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def fetch_html(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def resolve_image_url(src: str, base_url: str) -> str | None:
    absolute = urljoin(base_url, src)
    parsed = urlparse(absolute)
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.path.startswith(ALLOWED_DIR_PREFIX):
        return None
    suffix = Path(urlparse(absolute).path).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return None
    return absolute


def extract_gallery_entries(html: str, page_url: str) -> list[dict[str, str]]:
    parser = GalleryParser()
    parser.feed(html)

    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for index, (src, caption) in enumerate(parser.entries, start=1):
        image_url = resolve_image_url(src, page_url)
        if image_url is None or image_url in seen:
            continue
        seen.add(image_url)
        original_name = Path(urlparse(image_url).path).name
        sequential_name = f"{index:03d}_{original_name}"
        rows.append(
            {
                "image_id": f"bhuj_2001_{index:03d}",
                "filename": sequential_name,
                "source_page": page_url,
                "original_image_url": image_url,
                "caption": caption,
            }
        )
    return rows


def download_file(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


def write_sources_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "filename",
        "source_page",
        "original_image_url",
        "caption",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    print(f"Fetching gallery page:\n  {SOURCE_PAGE}")
    try:
        html = fetch_html(SOURCE_PAGE)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        print(f"Failed to fetch source page: {exc}", file=sys.stderr)
        return 1

    entries = extract_gallery_entries(html, SOURCE_PAGE)
    if not entries:
        print("No jpg/jpeg/png images found under the Bhuj_2001 directory.", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_sources_csv(entries, CSV_PATH)
    print(f"Wrote provenance CSV: {CSV_PATH.relative_to(REPO_ROOT)}")
    print(f"Found {len(entries)} image(s). Saving to {OUTPUT_DIR.relative_to(REPO_ROOT)}/")

    downloaded = 0
    skipped = 0
    failed = 0

    for i, entry in enumerate(entries):
        dest = OUTPUT_DIR / entry["filename"]
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  skip  {entry['filename']} (already present)")
            skipped += 1
            continue

        try:
            download_file(entry["original_image_url"], dest)
            print(f"  saved {entry['filename']}")
            downloaded += 1
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            print(f"  FAIL  {entry['filename']}: {exc}", file=sys.stderr)
            if dest.exists():
                dest.unlink(missing_ok=True)
            failed += 1

        if i < len(entries) - 1:
            time.sleep(DOWNLOAD_DELAY_SEC)

    print(
        f"Done. downloaded={downloaded} skipped={skipped} failed={failed} "
        f"total={len(entries)}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
