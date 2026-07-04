import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque

def fetch_text(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Collect links FIRST — the crawler needs the nav links to find sub-pages.
    base_host = urlparse(url).netloc
    links = []
    for a in soup.find_all("a", href=True):
        absolute = urljoin(url, a["href"])
        if absolute.startswith("http") and urlparse(absolute).netloc == base_host:
            links.append(absolute)
    links = list(dict.fromkeys(links))

    # THEN strip boilerplate so the text is just content (kills repeated nav).
    for tag in soup(["nav", "header", "footer", "script", "style", "aside"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)

    return text, links

def crawl(seed_url, max_pages=12):
    """BFS over same-host doc pages within the seed's section. Returns [(url, text), ...]."""
    seed_prefix = urlparse(seed_url).path.rsplit("/", 1)[0]   # parent dir, e.g. /en/rest/issues
    seen, queue, pages = set(), deque([seed_url]), []

    while queue and len(pages) < max_pages:
        url = queue.popleft().split("#")[0].rstrip("/")       # drop fragment + trailing slash
        if url in seen:
            continue
        seen.add(url)
        try:
            text, links = fetch_text(url)
        except requests.RequestException:
            continue                                          # error-safe: skip dead pages
        pages.append((url, text))

        for link in links:
            clean = link.split("#")[0].rstrip("/")
            if clean not in seen and urlparse(clean).path.startswith(seed_prefix):
                queue.append(clean)                           # only follow links in the same section

    return pages
