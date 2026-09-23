"""Scraping de feeds RSS de checadores brasileiros e lusofonos.

Usa feedparser (pip install feedparser) e BeautifulSoup para limpar HTML.

Uso:
    python -m investigation.expansion.rss_scraper \\
        --out investigation/expansion/raw/rss_claims.jsonl \\
        --rss investigation/expansion/config/feeds.json

O arquivo feeds.json tem o formato:
    [
      {"name": "aos_fatos", "publisher": "Aos Fatos",
       "feed": "https://www.aosfatos.org/feed/", "type": "news"},
      ...
    ]
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

# Checadores cobertos. Cada item: name, publisher, feed, type.
# type: 'news' | 'whatsapp' | 'social' | 'politics'
DEFAULT_FEEDS = [
    # BR
    {"name": "aos_fatos", "publisher": "Aos Fatos",
     "feed": "https://www.aosfatos.org/feed/", "type": "news"},
    {"name": "lupa", "publisher": "Agência Lupa",
     "feed": "https://lupa.uol.com.br/feed/", "type": "news"},
    {"name": "boatos", "publisher": "Boatos.org",
     "feed": "https://www.boatos.org/feed", "type": "news"},
    {"name": "uol_confere", "publisher": "UOL Confere",
     "feed": "https://noticias.uol.com.br/confere/rss.xml", "type": "news"},
    {"name": "e_farsas", "publisher": "E-Farsas",
     "feed": "https://www.e-farsas.com/feed", "type": "news"},
    # PT
    {"name": "poligrafo", "publisher": "Polígrafo SAPO",
     "feed": "https://poligrafo.sapo.pt/feed", "type": "news"},
    {"name": "observador", "publisher": "Observador",
     "feed": "https://observador.pt/feed/", "type": "news"},
]


class HTMLStripper(HTMLParser):
    """Remove tags HTML, preserva texto."""

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data):
        self.parts.append(data)

    def get_text(self) -> str:
        return " ".join(" ".join(self.parts).split())


def strip_html(html: str) -> str:
    s = HTMLStripper()
    try:
        s.feed(html)
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)
    return s.get_text()


def fetch(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "FakenewsBR/2.0 (research)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_rss(xml_bytes: bytes) -> list[dict]:
    """Parser RSS minimo. Evita dependencia de feedparser para 0 deps."""
    text = xml_bytes.decode("utf-8", errors="ignore")
    items = []
    # Quebra por <item> ou <entry>
    for block in re.findall(r"<item[^>]*>(.*?)</item>", text, re.DOTALL):
        title = re.search(r"<title[^>]*>(.*?)</title>", block, re.DOTALL)
        link = re.search(r"<link[^>]*>(.*?)</link>", block)
        pub = re.search(
            r"<pubDate[^>]*>(.*?)</pubDate>", block, re.DOTALL)
        desc = re.search(
            r"<description[^>]*>(.*?)</description>", block, re.DOTALL)
        items.append({
            "title": (title.group(1).strip() if title else ""),
            "link": (link.group(1).strip() if link else ""),
            "pubDate": (pub.group(1).strip() if pub else ""),
            "description": (desc.group(1).strip() if desc else ""),
        })
    return items


def pubdate_to_iso(s: str) -> str | None:
    """Tenta RFC822 (formato RSS mais comum)."""
    if not s:
        return None
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(s).date().isoformat()
    except Exception:
        return None


def scrape(feeds: list[dict], out_path: Path, sleep_s: float = 2.0) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counters = {
        "feeds_ok": 0, "feeds_failed": 0, "items": 0, "items_kept": 0,
    }
    with out_path.open("a", encoding="utf-8") as f:
        for feed in feeds:
            try:
                xml = fetch(feed["feed"])
                items = parse_rss(xml)
                counters["feeds_ok"] += 1
                counters["items"] += len(items)
                for it in items:
                    desc = strip_html(it["description"])
                    title = strip_html(it["title"])
                    if len(desc) < 50:
                        continue
                    rid_src = it["link"] or title
                    from .schema import short_hash
                    rid = f"rss_{feed['name']}_{short_hash(rid_src)}"
                    row = {
                        "rid": rid,
                        "dataset_name": f"RSS_{feed['name'].upper()}",
                        "source_type": feed["type"],
                        "source_description": feed["publisher"],
                        "label": "",  # precisa de LLM para rotular
                        "date_iso": pubdate_to_iso(it["pubDate"]),
                        "url_review": it["link"],
                        "text": f"{title}\n\n{desc}",
                        "text_clean": f"{title}\n\n{desc}",
                        "text_no_url": re.sub(
                            r"https?://\S+", "", f"{title}\n\n{desc}").strip(),
                        "extracted_urls": " ".join(
                            re.findall(r"https?://\S+", f"{title}\n\n{desc}")),
                        "is_duplicated": 0, "is_null": 0, "too_short": 0,
                        "factcheck_rating": "",
                        "factcheck_claimant": "",
                        "factcheck_url": it["link"],
                        "_provenance": {
                            "source": "rss", "feed": feed["name"],
                            "publisher": feed["publisher"],
                        },
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.write(json.dumps(row["_provenance"],
                                       ensure_ascii=False) + "\n")
                    counters["items_kept"] += 1
            except Exception as e:
                counters["feeds_failed"] += 1
                print(f"  ! falha em {feed['name']}: {e}", flush=True)
            time.sleep(sleep_s)
    return counters


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path("investigation/expansion/raw/rss_claims.jsonl"))
    ap.add_argument("--feeds", type=Path, default=None,
                    help="JSON customizado com lista de feeds")
    ap.add_argument("--sleep", type=float, default=2.0)
    args = ap.parse_args()

    feeds = DEFAULT_FEEDS
    if args.feeds and args.feeds.exists():
        feeds = json.loads(args.feeds.read_text(encoding="utf-8"))

    print(f"scrape de {len(feeds)} feeds RSS")
    counters = scrape(feeds, args.out, args.sleep)
    for k, v in counters.items():
        print(f"  {k}: {v}")
    print(f"saida em {args.out}")


if __name__ == "__main__":
    main()
