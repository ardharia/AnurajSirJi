"""Download publicly exposed SlideShare slide images for Bhuj 2001 research/testing.

Inspects the public slideshow HTML (including embedded __NEXT_DATA__), builds CDN
URLs from the page's own publicly exposed slide host/path/size metadata, and
downloads full-resolution slide images with provenance tracking.

Does not bypass authentication, DRM, paywalls, or access controls, and does not
call private/authenticated APIs.
"""

from __future__ import annotations

import csv
import html as html_lib
import json
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

SOURCE_PAGE = "https://www.slideshare.net/slideshow/earthquack-2001-bhujgujarat/45917265"
ALLOWED_IMAGE_HOST = "image.slidesharecdn.com"
DOWNLOAD_DELAY_SEC = 0.75
USER_AGENT = (
    "Mozilla/5.0 (compatible; seismic-damage-research/0.1; "
    "+internal research/testing; respectful crawler)"
)
ACCEPT_HEADER = "image/jpeg,image/png,image/webp,image/*;q=0.8,*/*;q=0.5"

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "images" / "bhuj_2001_slideshare"
ANNOTATIONS_DIR = REPO_ROOT / "data" / "annotations"
CSV_PATH = ANNOTATIONS_DIR / "bhuj_2001_slideshare_sources.csv"

IMAGE_MAGIC: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
]


def detect_image_extension(data: bytes) -> str | None:
    """Return a file extension if magic bytes identify a common image format."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    for magic, ext in IMAGE_MAGIC:
        if data.startswith(magic):
            return ext
    return None


def fetch_bytes(url: str, *, accept: str | None = None) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    request = Request(url, headers=headers)
    with urlopen(request, timeout=60) as response:
        return response.read()


def fetch_html(url: str) -> str:
    raw = fetch_bytes(url, accept="text/html,application/xhtml+xml;q=0.9,*/*;q=0.8")
    return raw.decode("utf-8", errors="replace")


def parse_next_data(page_html: str) -> dict:
    match = re.search(
        r'<script id="__NEXT_DATA__"[^>]*type="application/json"[^>]*>(.*?)</script>',
        page_html,
        re.S,
    )
    if match is None:
        match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', page_html, re.S)
    if match is None:
        raise ValueError("Public page HTML did not contain __NEXT_DATA__ JSON.")
    return json.loads(match.group(1))


class SlideCaptionParser(HTMLParser):
    """Collect publicly rendered slide preview alt text keyed by slide number."""

    def __init__(self) -> None:
        super().__init__()
        self.captions: dict[int, str] = {}
        self._widths: dict[int, int] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "img":
            return
        attr_map = {k: (v or "") for k, v in attrs}
        src = attr_map.get("src", "")
        if ALLOWED_IMAGE_HOST not in src:
            return
        match = re.search(r"-(\d+)-(\d+)\.(?:jpg|jpeg|png|webp)(?:\?.*)?$", src, re.I)
        if match is None:
            return
        slide_num = int(match.group(1))
        width = int(match.group(2))
        caption = _normalize_whitespace(html_lib.unescape(attr_map.get("alt", "")))
        prev_width = self._widths.get(slide_num, -1)
        if width >= prev_width and caption:
            self.captions[slide_num] = caption
            self._widths[slide_num] = width


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def choose_best_image_size(image_sizes: list[dict]) -> dict:
    if not image_sizes:
        raise ValueError("slideshow.slides.imageSizes missing from public page data.")
    return max(image_sizes, key=lambda item: int(item.get("width", 0)))


def build_slide_url(slides_meta: dict, slide_number: int, size: dict) -> str:
    host = str(slides_meta["host"]).rstrip("/")
    location = str(slides_meta["imageLocation"]).strip("/")
    title = str(slides_meta["title"])
    quality = int(size["quality"])
    width = int(size["width"])
    return f"{host}/{location}/{quality}/{title}-{slide_number}-{width}.jpg"


def extract_slide_entries(page_html: str, source_page: str) -> list[dict[str, str]]:
    next_data = parse_next_data(page_html)
    slideshow = next_data.get("props", {}).get("pageProps", {}).get("slideshow")
    if not isinstance(slideshow, dict):
        raise ValueError("Public page JSON did not include slideshow metadata.")

    if slideshow.get("isPrivate") is True:
        raise PermissionError(
            "Slideshow is marked private in public page metadata; refusing to download."
        )
    if slideshow.get("isViewable") is False:
        raise PermissionError(
            "Slideshow is not viewable according to public page metadata; refusing."
        )

    slides_meta = slideshow.get("slides")
    if not isinstance(slides_meta, dict):
        raise ValueError("Public page JSON did not include slideshow.slides metadata.")

    host = urlparse(str(slides_meta.get("host", ""))).netloc.lower()
    if host != ALLOWED_IMAGE_HOST:
        raise ValueError(f"Unexpected slide image host in public metadata: {host!r}")

    total_slides = int(slideshow.get("totalSlides") or 0)
    if total_slides <= 0:
        raise ValueError("Public page metadata did not expose a positive totalSlides count.")

    size = choose_best_image_size(list(slides_meta.get("imageSizes") or []))
    presentation_title = str(slideshow.get("title") or slides_meta.get("title") or "slide")

    caption_parser = SlideCaptionParser()
    caption_parser.feed(page_html)

    entries: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for slide_number in range(1, total_slides + 1):
        image_url = build_slide_url(slides_meta, slide_number, size)
        if image_url in seen_urls:
            continue
        seen_urls.add(image_url)

        original_name = Path(urlparse(image_url).path).name
        filename = f"{slide_number:03d}_{original_name}"
        caption = caption_parser.captions.get(slide_number) or (
            f"{presentation_title} — slide {slide_number} of {total_slides}"
        )
        entries.append(
            {
                "image_id": f"bhuj_2001_slideshare_{slide_number:03d}",
                "filename": filename,
                "source_page": source_page,
                "original_image_url": image_url,
                "caption": caption,
            }
        )

    return entries


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


def download_and_validate(url: str, destination: Path) -> Path:
    """Download one image, validate magic bytes, and adjust extension if needed."""
    data = fetch_bytes(url, accept=ACCEPT_HEADER)
    if not data:
        raise ValueError("Empty response body")

    detected_ext = detect_image_extension(data)
    if detected_ext is None:
        raise ValueError("Downloaded bytes are not a recognized image format")

    final_path = destination
    if destination.suffix.lower() != detected_ext:
        final_path = destination.with_suffix(detected_ext)

    final_path.write_bytes(data)
    return final_path


def main() -> int:
    print(f"Fetching public slideshow page:\n  {SOURCE_PAGE}")
    try:
        page_html = fetch_html(SOURCE_PAGE)
        entries = extract_slide_entries(page_html, SOURCE_PAGE)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, PermissionError) as exc:
        print(f"Failed to inspect/extract public slide URLs: {exc}", file=sys.stderr)
        return 1

    if not entries:
        print("No publicly exposed slide image URLs found.", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # CSV is written up front with planned filenames; updated below if extension changes.
    write_sources_csv(entries, CSV_PATH)
    print(f"Wrote provenance CSV: {CSV_PATH.relative_to(REPO_ROOT)}")
    print(
        f"Found {len(entries)} publicly exposed slide image(s). "
        f"Saving to {OUTPUT_DIR.relative_to(REPO_ROOT)}/"
    )

    downloaded = 0
    skipped = 0
    failed = 0

    for i, entry in enumerate(entries):
        planned = OUTPUT_DIR / entry["filename"]
        stem_prefix = planned.name.rsplit(".", 1)[0]
        existing = sorted(OUTPUT_DIR.glob(f"{stem_prefix}.*"))
        already = next(
            (
                path
                for path in existing
                if path.is_file()
                and path.stat().st_size > 0
                and detect_image_extension(path.read_bytes()[:32]) is not None
            ),
            None,
        )
        if already is not None:
            entry["filename"] = already.name
            print(f"  skip  {already.name} (already present)")
            skipped += 1
            continue

        try:
            saved = download_and_validate(entry["original_image_url"], planned)
            entry["filename"] = saved.name
            print(f"  saved {saved.name}")
            downloaded += 1
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            print(f"  FAIL  {entry['filename']}: {exc}", file=sys.stderr)
            for leftover in OUTPUT_DIR.glob(f"{stem_prefix}.*"):
                leftover.unlink(missing_ok=True)
            failed += 1

        if i < len(entries) - 1:
            time.sleep(DOWNLOAD_DELAY_SEC)

    write_sources_csv(entries, CSV_PATH)
    print(
        f"Done. downloaded={downloaded} skipped={skipped} failed={failed} "
        f"total={len(entries)}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
