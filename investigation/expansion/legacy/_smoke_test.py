"""Smoke test: valida que os scripts basicos funcionam sem rede."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from investigation.expansion.rss_scraper import parse_rss, pubdate_to_iso, strip_html

xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
<title>Teste</title>
<item>
<title>Falso: terra plana</title>
<link>https://www.aosfatos.org/fake/terra-plana</link>
<pubDate>Mon, 15 Mar 2021 10:00:00 +0000</pubDate>
<description>Circulou em redes sociais que a Terra e plana. Isso e falso.</description>
</item>
<item>
<title>Verdadeiro: vacinas salvam vidas</title>
<link>https://www.aosfatos.org/real/vacinas</link>
<pubDate>Tue, 20 Jul 2021 12:00:00 +0000</pubDate>
<description>Estudos comprovam eficacia das vacinas contra COVID-19.</description>
</item>
</channel>
</rss>"""

items = parse_rss(xml)
print(f"{len(items)} items parseados")
for it in items:
    print(f"  titulo: {strip_html(it['title'])[:60]}")
    print(f"  pubDate: {it['pubDate']!r} -> {pubdate_to_iso(it['pubDate'])}")
    print(f"  desc: {strip_html(it['description'])[:80]}")
    print()

# Schema tests
from investigation.expansion.schema import (
    rating_to_label, is_trusted_publisher, compute_metrics, short_hash)

print("rating_to_label tests:")
cases = [
    ("Falso", "fake"), ("Verdadeiro", "true"), ("Enganoso", "hard"),
    ("COMPROVADO", "true"), ("fake", "fake"), ("Distorcido", "hard"),
    ("Fora de contexto", "hard"), ("xpto", "other"),
]
ok = 0
for inp, expected in cases:
    got = rating_to_label(inp)
    status = "OK" if got == expected else "FAIL"
    if got == expected:
        ok += 1
    print(f"  {inp!r:25s} -> {got!r:8s} expected={expected!r} [{status}]")
print(f"  {ok}/{len(cases)} ok")

print()
print("compute_metrics:")
print(compute_metrics("Ola Mundo! Como vai? AAAAAAAAA..."))
print()
print("short_hash dedupe:", short_hash("a") != short_hash("b"))
