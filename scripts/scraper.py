"""
Auslan Signbank scraper for the Auslan NLP Sign Retrieval project.

Scrapes sign entries from https://auslan.org.au/ and merges them into
the project's sign dictionary. Respects robots.txt, rate-limits all
requests, and identifies itself as a research/educational bot.

License note: Auslan Signbank content is CC BY-NC-ND 4.0.
This scraper is intended for non-commercial research/education only.

Usage:
    python scripts/scraper.py                        # Full scrape
    python scripts/scraper.py --limit 100            # First 100 signs only
    python scripts/scraper.py --dry-run              # Fetch URLs, no file writes
    python scripts/scraper.py --delay 2.0            # Set request delay (seconds)
    python scripts/scraper.py --output data/gloss/signbank_scraped.json
    python scripts/scraper.py --no-merge             # Skip merge into main dict
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
import urllib.parse
import urllib.robotparser
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional heavy dependencies — fail clearly at import time if missing
# ---------------------------------------------------------------------------
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    sys.exit("ERROR: 'requests' is not installed. Run: pip install requests>=2.31.0")

try:
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit(
        "ERROR: 'beautifulsoup4' is not installed. "
        "Run: pip install beautifulsoup4>=4.12.0"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://auslan.org.au"
ROBOTS_URL = "https://auslan.org.au/robots.txt"

# Signbank uses an A-Z search interface — each letter returns a paginated list
# of sign detail pages at /dictionary/words/WORD-N.html
SEARCH_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SEARCH_URL_TEMPLATE = (
    "https://auslan.org.au/dictionary/search/?query={letter}&category="
)

USER_AGENT = (
    "AuslanNLPResearchBot/1.0 "
    "(Non-commercial academic research; "
    "Auslan NLP Sign Retrieval project; "
    "contact: research-bot@localhost)"
)

REQUEST_TIMEOUT = 10  # seconds
MAX_RETRIES = 3
BACKOFF_FACTOR = 2.0
SAVE_INTERVAL = 50  # save progress every N signs

# Paths relative to the project root (one level up from scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "data", "gloss", "signbank_scraped.json")
MAIN_DICT_PATH = os.path.join(PROJECT_ROOT, "data", "gloss", "auslan_dictionary.json")
BACKUP_DICT_PATH = os.path.join(
    PROJECT_ROOT, "data", "gloss", "auslan_dictionary_backup.json"
)
# Download scraped videos to D: drive by default to save C: drive space.
# Override with SCRAPED_VIDEO_DIR env var if needed.
VIDEO_DIR = os.environ.get("SCRAPED_VIDEO_DIR", r"D:\nlp\auslan-videos")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------


def build_session() -> requests.Session:
    """Return a requests Session with retry logic and the research User-Agent."""
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


# ---------------------------------------------------------------------------
# robots.txt check
# ---------------------------------------------------------------------------


def check_robots(session: requests.Session) -> urllib.robotparser.RobotFileParser:
    """
    Fetch and parse robots.txt from Auslan Signbank.
    Returns the parser so callers can use .can_fetch().
    """
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(ROBOTS_URL)
    try:
        response = session.get(ROBOTS_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        rp.parse(response.text.splitlines())
        logger.info("robots.txt fetched and parsed successfully.")
    except Exception as exc:
        logger.warning("Could not fetch robots.txt (%s). Proceeding cautiously.", exc)
    return rp


def is_allowed(rp: urllib.robotparser.RobotFileParser, url: str) -> bool:
    """Return True if USER_AGENT is permitted to fetch url."""
    return rp.can_fetch(USER_AGENT, url)


# ---------------------------------------------------------------------------
# Sign list discovery
# ---------------------------------------------------------------------------


def _is_search_url(url: str) -> bool:
    """Return True if url points to the Signbank search endpoint."""
    parsed = urllib.parse.urlparse(url)
    return parsed.path.rstrip("/") == "/dictionary/search"


def fetch_sign_urls(
    session: requests.Session,
    rp: urllib.robotparser.RobotFileParser,
    delay: float,
    limit: Optional[int] = None,
) -> List[str]:
    """
    Crawl the Auslan Signbank A-Z search pages and return a deduplicated
    list of individual sign-page URLs.

    Signbank organises signs under:
        /dictionary/words/<WORD>-<N>.html
    Discovery works by iterating through /dictionary/search/?query=A..Z
    and following pagination within each letter.

    Only /dictionary/search/ pages are visited during discovery to avoid
    accidentally crawling into /dictionary/gloss/ detail pages.
    """
    sign_urls: List[str] = []
    visited_pages: set = set()

    for letter in SEARCH_LETTERS:
        if limit and len(sign_urls) >= limit:
            break

        search_url = SEARCH_URL_TEMPLATE.format(letter=letter)
        pages_to_visit = [search_url]

        while pages_to_visit:
            if limit and len(sign_urls) >= limit:
                break

            page_url = pages_to_visit.pop(0)
            if page_url in visited_pages:
                continue

            # Guard: only fetch actual search-results pages
            if not _is_search_url(page_url):
                logger.debug("Skipping non-search URL during discovery: %s", page_url)
                continue

            if not is_allowed(rp, page_url):
                logger.warning("robots.txt disallows: %s", page_url)
                continue

            logger.info("Fetching search page: %s", page_url)
            try:
                resp = session.get(page_url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
            except Exception as exc:
                logger.error("Failed to fetch %s: %s", page_url, exc)
                time.sleep(delay)
                continue

            visited_pages.add(page_url)
            soup = BeautifulSoup(resp.text, "html.parser")

            # Collect sign-entry links from this page
            _collect_sign_links(soup, sign_urls)

            # Discover pagination links (page 2, 3, etc. within this letter)
            for pag_url in _extract_pagination_urls(soup, page_url):
                if pag_url not in visited_pages:
                    pages_to_visit.append(pag_url)

            time.sleep(delay)

        logger.info(
            "Letter %s complete — %d total sign URLs so far.", letter, len(sign_urls)
        )

    # Deduplicate while preserving order (also normalise to lowercase path)
    seen: set = set()
    unique_urls: List[str] = []
    for url in sign_urls:
        normalised = url.lower()
        if normalised not in seen:
            seen.add(normalised)
            unique_urls.append(url)

    if limit:
        unique_urls = unique_urls[:limit]

    logger.info("Discovered %d unique sign URLs.", len(unique_urls))
    return unique_urls


def _make_absolute(href: str, base: str) -> str:
    """Convert a relative href to an absolute URL."""
    return urllib.parse.urljoin(base, href)


def _collect_sign_links(soup: BeautifulSoup, result: List[str]) -> None:
    """
    Scan a BeautifulSoup document for sign-entry anchor tags and append
    their absolute URLs to result.

    Auslan Signbank sign pages follow the pattern:
        /dictionary/words/<WORD>-<N>.html
    """
    for tag in soup.find_all("a", href=True):
        href: str = tag["href"]
        if "/dictionary/words/" in href and href.endswith(".html"):
            abs_url = _make_absolute(href, BASE_URL)
            result.append(abs_url)


def _extract_pagination_urls(soup: BeautifulSoup, current_url: str) -> List[str]:
    """
    Return a list of pagination URLs found on a search results page.
    Signbank uses numbered page links and next/prev arrows.

    Only returns URLs that point to /dictionary/search/ to prevent the
    crawler from wandering into /dictionary/gloss/ detail pages.
    """
    urls: List[str] = []
    for tag in soup.find_all("a", href=True):
        href: str = tag["href"]
        abs_url = _make_absolute(href, current_url)

        # Hard requirement: pagination links must stay on the search endpoint
        if not _is_search_url(abs_url):
            continue

        text = tag.get_text(strip=True).lower()
        # Accept links with pagination indicators or page= query params
        if any(indicator in text for indicator in ("next", "»", "›", ">")) or (
            "page=" in href
        ):
            if abs_url not in urls:
                urls.append(abs_url)
    return urls


# ---------------------------------------------------------------------------
# Individual sign page parsing
# ---------------------------------------------------------------------------


def parse_sign_page(url: str, html: str) -> Optional[Dict]:
    """
    Parse an individual Auslan Signbank sign page and return a dict
    matching the project's dictionary schema, or None if unusable.

    The parser is intentionally resilient: each field is extracted
    independently so that missing elements do not abort the whole record.
    """
    soup = BeautifulSoup(html, "html.parser")

    # --- gloss / id_gloss ---
    gloss = _extract_gloss(soup, url)
    if not gloss:
        logger.debug("No gloss found at %s — skipping.", url)
        return None

    id_gloss = _extract_id_gloss(url, gloss)

    # --- keywords ---
    keywords = _extract_keywords(soup, gloss)
    if not keywords:
        keywords = [gloss.lower()]

    # --- video URL ---
    video_url = _extract_video_url(soup, url)
    if not video_url:
        logger.debug("No video URL found at %s — skipping.", url)
        return None

    # --- optional fields ---
    description = _extract_description(soup)
    handshape = _extract_field(soup, "handshape")
    location = _extract_field(soup, "location")
    category = _infer_category(soup, keywords)

    # Primary key for the dictionary is the lowercase first keyword
    primary_key = keywords[0].lower().strip()

    return {
        primary_key: {
            "gloss": gloss.upper(),
            "id_gloss": id_gloss,
            "category": category,
            "video_url": video_url,
            "video_local": None,
            "description": description,
            "keywords": keywords,
            "handshape": handshape,
            "location": location,
            "difficulty": None,
            "regional_variants": [],
            "frequency": None,
            "source": "signbank",
        }
    }


def _extract_gloss(soup: BeautifulSoup, url: str) -> Optional[str]:
    """Extract the primary gloss label from the page."""
    # Strategy 1: <h1> or <h2> with class containing 'gloss' or 'keyword'
    for tag in soup.find_all(["h1", "h2", "h3"]):
        cls = " ".join(tag.get("class", []))
        if any(k in cls.lower() for k in ("gloss", "keyword", "sign-title", "entry")):
            text = tag.get_text(strip=True)
            if text:
                return text

    # Strategy 2: <span> or <div> with id/class 'gloss'
    for selector in ("gloss", "keyword", "sign-name", "headword"):
        tag = soup.find(attrs={"id": selector}) or soup.find(attrs={"class": selector})
        if tag:
            text = tag.get_text(strip=True)
            if text:
                return text

    # Strategy 3: derive from URL — e.g. /dictionary/gloss/HELP-1.html -> HELP
    path = urllib.parse.urlparse(url).path
    basename = os.path.basename(path)  # e.g. "HELP-1.html"
    stem = os.path.splitext(basename)[0]  # e.g. "HELP-1"
    if stem:
        # Remove trailing numeric suffix (-1, -2a, etc.)
        word = re.sub(r"-\d+[a-z]?$", "", stem, flags=re.IGNORECASE)
        if word:
            return word.upper()

    # Strategy 4: first <h1> on the page regardless of class
    h1 = soup.find("h1")
    if h1:
        text = h1.get_text(strip=True)
        if text:
            return text

    return None


def _extract_id_gloss(url: str, gloss: str) -> str:
    """Derive an id_gloss string from the URL or gloss."""
    path = urllib.parse.urlparse(url).path
    basename = os.path.splitext(os.path.basename(path))[0]  # e.g. "HELP-1"
    if basename:
        return basename.lower()
    return gloss.lower().replace(" ", "-") + "-1"


def _extract_keywords(soup: BeautifulSoup, gloss: str) -> List[str]:
    """
    Extract the list of English keywords/synonyms for this sign.
    Returns deduplicated list, primary keyword first.

    On Signbank pages the keywords appear in a `.keywords` element like:
        <div class="keywords">Keywords: hello, hi</div>
    or as comma-separated text after a "Keywords:" label.
    """
    keywords = []

    # Strategy 1: element with id or class 'keywords' (Auslan Signbank pattern)
    # The actual site uses <div id="keywords"> containing:
    #   <p><strong>Keywords:</strong> <b>primary</b>, other1, other2</p>
    kw_el = soup.find(attrs={"id": "keywords"}) or soup.find(
        attrs={"class": "keywords"}
    )
    if kw_el:
        text = kw_el.get_text(strip=True)
        # Strip the "Keywords:" prefix if present
        if ":" in text:
            text = text.split(":", 1)[1]
        for kw in text.split(","):
            kw = kw.strip().lower()
            if kw and kw not in keywords:
                keywords.append(kw)

    # Strategy 2: a <ul>/<ol> or <div> labelled 'keyword-list' or 'translations'
    if not keywords:
        for selector in ("keyword-list", "translations"):
            container = soup.find(attrs={"id": selector}) or soup.find(
                attrs={"class": selector}
            )
            if container:
                for item in container.find_all(["li", "span", "a"]):
                    text = item.get_text(strip=True).lower()
                    if text and text not in keywords:
                        keywords.append(text)

    # Strategy 3: table rows labelled 'Keywords' or 'Translations'
    if not keywords:
        for row in soup.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if len(cells) >= 2:
                header = cells[0].get_text(strip=True).lower()
                if header in ("keywords", "keyword", "translations", "english"):
                    value = cells[1].get_text(strip=True)
                    for kw in value.split(","):
                        kw = kw.strip().lower()
                        if kw and kw not in keywords:
                            keywords.append(kw)

    # Always include the gloss itself as a fallback keyword
    gloss_lower = gloss.lower()
    if gloss_lower not in keywords:
        keywords.insert(0, gloss_lower)

    return keywords


def _extract_video_url(soup: BeautifulSoup, page_url: str) -> Optional[str]:
    """
    Find the video URL on the sign page.
    Checks <video>, <source>, and common data-attributes.
    Returns an absolute URL or None.
    """
    # Strategy 1: <source> inside <video>
    for source in soup.find_all("source"):
        src = source.get("src", "")
        if src and _looks_like_video(src):
            return _make_absolute(src, page_url)

    # Strategy 2: <video src="...">
    for video in soup.find_all("video"):
        src = video.get("src", "")
        if src and _looks_like_video(src):
            return _make_absolute(src, page_url)
        # data-src fallback (lazy-loaded)
        src = video.get("data-src", "")
        if src and _looks_like_video(src):
            return _make_absolute(src, page_url)

    # Strategy 3: <a href="..."> pointing to a video file
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if _looks_like_video(href):
            return _make_absolute(href, page_url)

    # Strategy 4: any element with a data-video or data-src attribute
    for tag in soup.find_all(attrs={"data-video": True}):
        src = tag["data-video"]
        if src:
            return _make_absolute(src, page_url)

    return None


def _looks_like_video(url: str) -> bool:
    """Return True if the URL path ends with a known video extension."""
    path = urllib.parse.urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in (".mp4", ".webm", ".ogg", ".mov"))


def _extract_description(soup: BeautifulSoup) -> Optional[str]:
    """Extract the textual definition/description of the sign."""
    # Strategy 1: Auslan Signbank uses .definition-panel containing definitions
    panels = soup.find_all(attrs={"class": "definition-panel"})
    if panels:
        parts = []
        for panel in panels:
            heading = panel.find(attrs={"class": "panel-heading"})
            heading_text = heading.get_text(strip=True) if heading else ""
            entries = panel.find_all(attrs={"class": "definition-entry"})
            for entry in entries:
                text = entry.get_text(strip=True)
                # Strip the leading counter number (e.g. "1.")
                text = re.sub(r"^\d+\.\s*", "", text)
                if text:
                    prefix = f"({heading_text}) " if heading_text else ""
                    parts.append(prefix + text)
            # If no .definition-entry children, grab the panel text directly
            if not entries:
                text = panel.get_text(strip=True)
                if heading_text and text.startswith(heading_text):
                    text = text[len(heading_text) :].strip()
                text = re.sub(r"^\d+\.\s*", "", text)
                if text:
                    parts.append(text)
        if parts:
            return "; ".join(parts[:3])  # Keep it concise — max 3 definitions

    # Strategy 2: element with id or class matching common names
    for selector in ("definition", "description", "meaning", "sign-description"):
        tag = soup.find(attrs={"id": selector}) or soup.find(attrs={"class": selector})
        if tag:
            text = tag.get_text(strip=True)
            if text:
                return text

    # Strategy 3: look for a <dt>/<dd> pair
    for dt in soup.find_all("dt"):
        label = dt.get_text(strip=True).lower()
        if label in ("definition", "description", "meaning"):
            dd = dt.find_next_sibling("dd")
            if dd:
                text = dd.get_text(strip=True)
                if text:
                    return text

    return None


def _extract_field(soup: BeautifulSoup, field_name: str) -> Optional[str]:
    """
    Generic extractor for named metadata fields (handshape, location, etc.).
    Looks for table rows, dt/dd pairs, and labelled spans/divs.
    """
    # Table row search
    for row in soup.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) >= 2:
            header = cells[0].get_text(strip=True).lower()
            if field_name in header:
                value = cells[1].get_text(strip=True)
                if value:
                    return value

    # dt/dd search
    for dt in soup.find_all("dt"):
        if field_name in dt.get_text(strip=True).lower():
            dd = dt.find_next_sibling("dd")
            if dd:
                text = dd.get_text(strip=True)
                if text:
                    return text

    # Labelled span/div
    for tag in soup.find_all(attrs={"class": True}):
        cls = " ".join(tag.get("class", []))
        if field_name in cls.lower():
            text = tag.get_text(strip=True)
            if text:
                return text

    return None


# Category keyword heuristics — maps broad topic keywords onto category labels
_CATEGORY_HINTS: List[Tuple[List[str], str]] = [
    (["greeting", "hello", "goodbye", "welcome"], "greeting"),
    (["emotion", "feeling", "happy", "sad", "angry", "fear", "love"], "emotions"),
    (["food", "eat", "drink", "cook", "meal", "hunger"], "basic_needs"),
    (["time", "day", "week", "month", "year", "yesterday", "tomorrow"], "time"),
    (["colour", "color", "red", "blue", "green", "yellow"], "descriptive"),
    (["number", "count", "quantity", "how many"], "numbers"),
    (["family", "mother", "father", "sister", "brother", "parent"], "family"),
    (["health", "sick", "doctor", "hospital", "medicine", "pain"], "health"),
    (["work", "job", "school", "learn", "study", "teach"], "education_work"),
    (["place", "location", "country", "city", "home", "house"], "places"),
    (["travel", "transport", "car", "bus", "train", "plane"], "transport"),
    (["sport", "exercise", "gym", "fitness", "run", "swim"], "fitness_core"),
    (["animal", "dog", "cat", "bird", "fish"], "animals"),
    (["nature", "weather", "rain", "sun", "tree", "flower"], "nature"),
    (["money", "pay", "cost", "price", "buy", "sell"], "commerce"),
    (["communication", "phone", "call", "email", "letter"], "communication"),
    (["question", "who", "what", "where", "when", "why", "how"], "questions"),
]


def _infer_category(soup: BeautifulSoup, keywords: List[str]) -> str:
    """
    Infer a category string by matching page topic metadata or keywords
    against known category hints. Falls back to 'general'.
    """
    # Try to find an explicit topic/category label on the page
    for selector in ("topic", "category", "domain", "field"):
        tag = soup.find(attrs={"id": selector}) or soup.find(attrs={"class": selector})
        if tag:
            text = tag.get_text(strip=True).lower()
            if text:
                # Try to map to a known category
                for hints, cat in _CATEGORY_HINTS:
                    if any(h in text for h in hints):
                        return cat
                return text  # Return the raw topic label if no mapping found

    # Keyword-based heuristic
    kw_text = " ".join(keywords).lower()
    for hints, cat in _CATEGORY_HINTS:
        if any(h in kw_text for h in hints):
            return cat

    return "general"


# ---------------------------------------------------------------------------
# Video downloading
# ---------------------------------------------------------------------------


def download_video(
    session: requests.Session,
    video_url: str,
    primary_key: str,
    delay: float,
) -> Optional[str]:
    """
    Download a sign video to media/videos/<primary_key>.mp4.

    Returns the local relative path (e.g. "media/videos/hello.mp4") on success,
    or None on failure. Skips download if the file already exists.
    """
    os.makedirs(VIDEO_DIR, exist_ok=True)

    # Sanitise filename: keep only alphanumeric, hyphens, underscores
    safe_name = re.sub(r"[^\w\-]", "_", primary_key).strip("_").lower()
    if not safe_name:
        safe_name = "unknown"
    local_filename = f"{safe_name}.mp4"
    local_path = os.path.join(VIDEO_DIR, local_filename)
    relative_path = f"media/videos/{local_filename}"

    # Skip if already downloaded
    if os.path.isfile(local_path) and os.path.getsize(local_path) > 1000:
        logger.debug("Video already exists: %s", local_path)
        return relative_path

    try:
        resp = session.get(video_url, timeout=30, stream=True)
        resp.raise_for_status()

        with open(local_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)

        size_kb = os.path.getsize(local_path) / 1024
        logger.info("Downloaded video: %s (%.0f KB)", local_filename, size_kb)
        time.sleep(delay)
        return relative_path

    except Exception as exc:
        logger.error("Failed to download video %s: %s", video_url, exc)
        # Clean up partial file
        if os.path.exists(local_path):
            os.remove(local_path)
        return None


# ---------------------------------------------------------------------------
# Scraper orchestration
# ---------------------------------------------------------------------------


def scrape(
    delay: float = 1.5,
    limit: Optional[int] = None,
    dry_run: bool = False,
    output_path: str = DEFAULT_OUTPUT,
    no_merge: bool = False,
    download_videos: bool = True,
) -> Dict:
    """
    Main scraping routine.

    Returns the dict of scraped entries (even in dry-run mode so callers
    can inspect results without file I/O).
    """
    session = build_session()
    rp = check_robots(session)

    # Verify the root URL is scrapable before committing to a full run
    if not is_allowed(rp, BASE_URL + "/dictionary/"):
        logger.error(
            "robots.txt disallows access to /dictionary/. "
            "Aborting out of respect for site policy."
        )
        return {}

    # Phase 1: discover sign URLs
    sign_urls = fetch_sign_urls(session, rp, delay, limit)
    if not sign_urls:
        logger.warning("No sign URLs found. Check the index URL and HTML structure.")
        return {}

    if dry_run:
        logger.info(
            "[DRY RUN] Would scrape %d sign pages. No files written.", len(sign_urls)
        )
        for url in sign_urls[:20]:
            logger.info("  %s", url)
        if len(sign_urls) > 20:
            logger.info("  ... and %d more.", len(sign_urls) - 20)
        return {}

    # Phase 2: scrape individual sign pages
    scraped: Dict = {}
    errors: List[str] = []

    # Load any previously saved partial progress
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as fh:
                scraped = json.load(fh)
            logger.info(
                "Resuming: loaded %d previously scraped entries from %s",
                len(scraped),
                output_path,
            )
        except Exception as exc:
            logger.warning("Could not load existing output file: %s", exc)

    for i, url in enumerate(sign_urls, start=1):
        # Skip if we already have this entry (resume support)
        id_gloss = os.path.splitext(os.path.basename(urllib.parse.urlparse(url).path))[
            0
        ].lower()
        if any(entry.get("id_gloss") == id_gloss for entry in scraped.values()):
            logger.debug("Already scraped %s — skipping.", url)
            continue

        if not is_allowed(rp, url):
            logger.warning("robots.txt disallows %s — skipping.", url)
            continue

        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except Exception as exc:
            logger.error("Request failed for %s: %s", url, exc)
            errors.append(url)
            time.sleep(delay)
            continue

        entry = parse_sign_page(url, resp.text)
        if entry:
            # Download the video; keep the remote URL so clones work without videos
            if download_videos:
                for key, data in entry.items():
                    remote_url = data.get("video_url", "")
                    if remote_url and remote_url.startswith("http"):
                        data["video_url_remote"] = remote_url
                        local_path = download_video(session, remote_url, key, delay)
                        if local_path:
                            data["video_url"] = local_path
                            data["video_local"] = local_path
                        else:
                            logger.warning(
                                "Video download failed for %s — keeping remote URL.",
                                key,
                            )
            else:
                for key, data in entry.items():
                    remote_url = data.get("video_url", "")
                    if remote_url and remote_url.startswith("http"):
                        data["video_url_remote"] = remote_url
            scraped.update(entry)
        else:
            logger.debug("No usable data extracted from %s", url)

        # Progress report
        if i % 10 == 0 or i == len(sign_urls):
            logger.info(
                "Progress: %d / %d pages processed, %d entries collected, %d errors.",
                i,
                len(sign_urls),
                len(scraped),
                len(errors),
            )

        # Incremental save
        if i % SAVE_INTERVAL == 0:
            _save_json(scraped, output_path)
            logger.info(
                "Incremental save: %d entries written to %s.", len(scraped), output_path
            )

        time.sleep(delay)

    # Final save
    _save_json(scraped, output_path)
    logger.info(
        "Scrape complete. %d entries saved to %s. %d URLs failed.",
        len(scraped),
        output_path,
        len(errors),
    )

    if errors:
        logger.warning("Failed URLs (%d):", len(errors))
        for err_url in errors:
            logger.warning("  %s", err_url)

    # Phase 3: merge into main dictionary
    if not no_merge:
        merge(scraped, MAIN_DICT_PATH, BACKUP_DICT_PATH)

    return scraped


# ---------------------------------------------------------------------------
# Dictionary merge
# ---------------------------------------------------------------------------


def merge(
    new_entries: Dict,
    main_path: str = MAIN_DICT_PATH,
    backup_path: str = BACKUP_DICT_PATH,
) -> None:
    """
    Merge new_entries into the main Auslan dictionary.

    Existing entries take priority — new entries are only added when
    the key does not already exist in the main dictionary.
    Backs up the original before writing.
    """
    if not new_entries:
        logger.info("No new entries to merge.")
        return

    # Load existing dictionary
    existing: Dict = {}
    if os.path.exists(main_path):
        try:
            with open(main_path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
            logger.info("Loaded %d existing entries from %s.", len(existing), main_path)
        except Exception as exc:
            logger.error("Failed to load main dictionary: %s", exc)
            return

        # Backup
        try:
            shutil.copy2(main_path, backup_path)
            logger.info("Backup written to %s.", backup_path)
        except Exception as exc:
            logger.warning("Could not write backup: %s", exc)
    else:
        logger.info("Main dictionary not found — will create it at %s.", main_path)

    # Merge: existing entries win
    added = 0
    for key, entry in new_entries.items():
        if key not in existing:
            existing[key] = entry
            added += 1

    logger.info(
        "Merge complete: %d new entries added. Total dictionary size: %d.",
        added,
        len(existing),
    )

    _save_json(existing, main_path)
    logger.info("Merged dictionary written to %s.", main_path)


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def _save_json(data: Dict, path: str) -> None:
    """Write data as formatted JSON, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape Auslan Signbank and expand the project's sign dictionary. "
            "Content is CC BY-NC-ND 4.0 — non-commercial research use only."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only scrape the first N sign URLs (useful for testing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch index pages and list sign URLs; do not scrape or write files.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        metavar="SECONDS",
        help="Seconds to wait between requests (default: 1.5).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        metavar="PATH",
        help="Output path for scraped JSON (default: data/gloss/signbank_scraped.json).",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip merging scraped data into the main dictionary.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip downloading sign videos to media/videos/.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.delay < 1.0:
        logger.warning(
            "Delay %.1fs is below the 1-second minimum recommended for "
            "respectful scraping. Consider using --delay 1.0 or higher.",
            args.delay,
        )

    logger.info(
        "Starting Auslan Signbank scraper | delay=%.1fs | limit=%s | dry_run=%s",
        args.delay,
        args.limit if args.limit else "none",
        args.dry_run,
    )

    scrape(
        delay=args.delay,
        limit=args.limit,
        dry_run=args.dry_run,
        output_path=args.output,
        no_merge=args.no_merge,
        download_videos=not args.no_download,
    )


if __name__ == "__main__":
    main()
