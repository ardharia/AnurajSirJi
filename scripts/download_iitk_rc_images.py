"""Download IITK/NICEE Bhuj RC building damage images for internal research/testing.

The listed entry URL is a frameset. This script fetches it over normal HTTP, follows
public frame targets to the content document, extracts jpg/jpeg/png image URLs from
the Bhuj report directory, associates nearby captions, and downloads with provenance.

Does not bypass authentication, access controls, robots restrictions, or anti-bot
protections. Uses only the Python standard library.
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

ENTRY_PAGE = "https://www.iitk.ac.in/nicee/EQ_Reports/Bhuj/build_rc.htm"
ALLOWED_NETLOC = {"www.iitk.ac.in", "iitk.ac.in"}
ALLOWED_PATH_PREFIX = "/nicee/EQ_Reports/Bhuj/"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SOURCE_COLLECTION = "IITK_NICEE_Bhuj_RC"
DOWNLOAD_DELAY_SEC = 0.75
USER_AGENT = (
    "Mozilla/5.0 (compatible; seismic-damage-research/0.1; "
    "+internal research/testing; respectful crawler)"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "images" / "bhuj_2001_iitk_rc"
ANNOTATIONS_DIR = REPO_ROOT / "data" / "annotations"
CSV_PATH = ANNOTATIONS_DIR / "bhuj_2001_iitk_rc_sources.csv"
ROBOTS_URL = "https://www.iitk.ac.in/robots.txt"


class FrameSrcParser(HTMLParser):
    """Collect frame/iframe src attributes from a frameset/wrapper page."""

    def __init__(self) -> None:
        super().__init__()
        self.srcs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in {"frame", "iframe"}:
            return
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        src = attr_map.get("src", "").strip()
        if src:
            self.srcs.append(src)


class ContentEventParser(HTMLParser):
    """Emit ordered image and text events for caption association."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, str]] = []
        self._text_parts: list[str] = []
        self._ignore_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._ignore_depth += 1
            return
        if self._ignore_depth:
            return

        attr_map = {k.lower(): (v or "") for k, v in attrs}
        if tag == "img":
            self._flush_text()
            src = attr_map.get("src", "").strip()
            if src:
                self.events.append(("img", src))
            return
        if tag == "br":
            self._text_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignore_depth:
            self._ignore_depth -= 1
            return
        if self._ignore_depth:
            return
        if tag in {"p", "div", "td", "li", "h1", "h2", "h3", "h4", "tr"}:
            self._flush_text()

    def handle_data(self, data: str) -> None:
        if self._ignore_depth:
            return
        self._text_parts.append(data)

    def _flush_text(self) -> None:
        text = _normalize_whitespace("".join(self._text_parts))
        self._text_parts = []
        if text:
            self.events.append(("text", text))


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read()


def fetch_text(url: str) -> str:
    raw = fetch_bytes(url)
    for encoding in ("utf-8", "windows-1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def path_allowed_by_robots(path: str, robots_text: str) -> bool:
    """Respect simple User-agent: * Disallow rules from robots.txt."""
    active = False
    disallows: list[str] = []
    allows: list[str] = []
    for raw_line in robots_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("user-agent:"):
            agent = line.split(":", 1)[1].strip()
            active = agent == "*"
            continue
        if not active:
            continue
        if lower.startswith("disallow:"):
            value = line.split(":", 1)[1].strip()
            if value:
                disallows.append(value)
        elif lower.startswith("allow:"):
            value = line.split(":", 1)[1].strip()
            if value:
                allows.append(value)

    def _matches(rule: str, candidate: str) -> bool:
        # Support trailing "$" exact-suffix rules used on this host.
        if rule.endswith("$"):
            return candidate.endswith(rule[:-1])
        return candidate.startswith(rule)

    matched_allow = max((len(r) for r in allows if _matches(r, path)), default=0)
    matched_disallow = max((len(r) for r in disallows if _matches(r, path)), default=0)
    if matched_disallow == 0:
        return True
    return matched_allow > matched_disallow


def ensure_robots_allows(urls: list[str]) -> None:
    try:
        robots_text = fetch_text(ROBOTS_URL)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        print(f"Warning: could not fetch robots.txt ({exc}); proceeding cautiously.")
        return

    for url in urls:
        path = urlparse(url).path or "/"
        if not path_allowed_by_robots(path, robots_text):
            raise PermissionError(f"robots.txt disallows access to path: {path}")


def is_allowed_image_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc.lower() not in ALLOWED_NETLOC:
        return False
    path = parsed.path or ""
    if not path.startswith(ALLOWED_PATH_PREFIX):
        return False
    suffix = Path(path).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return False
    # Skip tiny UI/nav naming patterns if they appear under the report path.
    name = Path(path).name.lower()
    if any(token in name for token in ("logo", "icon", "button", "spacer", "arrow", "nav")):
        return False
    return True


def resolve_content_pages(entry_url: str, entry_html: str) -> list[str]:
    """Return content page URLs. Follow frames when the entry page is a frameset."""
    lower = entry_html.lower()
    if "<frameset" not in lower and "<frame" not in lower:
        return [entry_url]

    parser = FrameSrcParser()
    parser.feed(entry_html)
    candidates: list[str] = []
    for src in parser.srcs:
        absolute = urljoin(entry_url, src)
        parsed = urlparse(absolute)
        if parsed.netloc.lower() not in ALLOWED_NETLOC:
            continue
        if not (parsed.path or "").startswith(ALLOWED_PATH_PREFIX):
            continue
        # Prefer content documents under Bhuj/, skip shared header/nav frames.
        name = Path(parsed.path).name.lower()
        if name in {"head.htm", "earthquake_reports_title.htm", "bhuj_t.htm"}:
            continue
        candidates.append(absolute)

    # De-duplicate while preserving order.
    unique: list[str] = []
    seen: set[str] = set()
    for url in candidates:
        if url not in seen:
            seen.add(url)
            unique.append(url)

    if not unique:
        raise ValueError(
            "Frameset page did not expose a public Bhuj content frame under "
            f"{ALLOWED_PATH_PREFIX}"
        )
    return unique


def extract_entries_from_content(page_url: str, page_html: str) -> list[dict[str, str]]:
    parser = ContentEventParser()
    parser.feed(page_html)
    parser._flush_text()

    pending_imgs: list[str] = []
    pairs: list[tuple[str, str]] = []

    def flush_pending(caption: str) -> None:
        nonlocal pending_imgs
        for src in pending_imgs:
            pairs.append((src, caption))
        pending_imgs = []

    for kind, value in parser.events:
        if kind == "img":
            pending_imgs.append(value)
        elif kind == "text":
            if pending_imgs:
                flush_pending(value)
            # Ignore standalone headings/text with no preceding images.

    # Images at end of page without a following caption.
    flush_pending("")

    entries: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for src, caption in pairs:
        absolute = urljoin(page_url, src)
        if not is_allowed_image_url(absolute):
            continue
        if absolute in seen_urls:
            continue
        seen_urls.add(absolute)
        original_name = Path(urlparse(absolute).path).name
        entries.append(
            {
                "source_page": page_url,
                "original_image_url": absolute,
                "caption": caption,
                "original_name": original_name,
            }
        )
    return entries


def assign_filenames(raw_entries: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, item in enumerate(raw_entries, start=1):
        filename = f"iitk_rc_{index:03d}_{item['original_name']}"
        rows.append(
            {
                "image_id": f"iitk_rc_{index:03d}",
                "filename": filename,
                "source_page": item["source_page"],
                "original_image_url": item["original_image_url"],
                "caption": item["caption"],
                "source_collection": SOURCE_COLLECTION,
            }
        )
    return rows


def write_sources_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "filename",
        "source_page",
        "original_image_url",
        "caption",
        "source_collection",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def download_file(url: str, destination: Path) -> None:
    data = fetch_bytes(url)
    if not data:
        raise ValueError("Empty response body")
    destination.write_bytes(data)


def main() -> int:
    print(f"Fetching entry page:\n  {ENTRY_PAGE}")
    try:
        ensure_robots_allows([ENTRY_PAGE, ALLOWED_PATH_PREFIX])
        entry_html = fetch_text(ENTRY_PAGE)
        content_pages = resolve_content_pages(ENTRY_PAGE, entry_html)
        print("Resolved content page(s):")
        for page in content_pages:
            print(f"  {page}")

        ensure_robots_allows(content_pages)
        raw_entries: list[dict[str, str]] = []
        seen: set[str] = set()
        for page_url in content_pages:
            page_html = fetch_text(page_url)
            for entry in extract_entries_from_content(page_url, page_html):
                url = entry["original_image_url"]
                if url in seen:
                    continue
                seen.add(url)
                raw_entries.append(entry)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, PermissionError) as exc:
        print(f"Failed to inspect/extract public image URLs: {exc}", file=sys.stderr)
        return 1

    entries = assign_filenames(raw_entries)
    if not entries:
        print("No eligible jpg/jpeg/png images found on the Bhuj RC report page.", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_sources_csv(entries, CSV_PATH)
    print(f"Wrote provenance CSV: {CSV_PATH.relative_to(REPO_ROOT)}")
    print(
        f"Found {len(entries)} image(s). Saving to {OUTPUT_DIR.relative_to(REPO_ROOT)}/"
    )

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
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
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
