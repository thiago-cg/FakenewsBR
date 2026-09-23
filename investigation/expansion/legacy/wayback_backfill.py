"""Backfill de datas via Wayback Machine CDX API.

Para cada registro em <input> sem date_iso ou com data < cutoff, busca
o snapshot mais proximo via CDX e atualiza date_iso.

Uso:
    python -m investigation.expansion.wayback_backfill \\
        --in investigation/expansion/raw/gfc_claims.jsonl \\
        --out investigation/expansion/raw/gfc_claims_dated.jsonl \\
        --cutoff 2018-01-01

Nao requer API key. Wayback Machine e' gratuito para uso nao-comercial.
Rate limit sugerido: 1 req/s.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

CDX_URL = "https://web.archive.org/cdx/server/cdx/search/cdx"


def cdx_lookup(url: str) -> str | None:
    """Busca o snapshot mais proximo de uma URL no CDX.

    Retorna a data mais antiga (YYYYMMDD) encontrada, ou None.
    Escolhemos a mais antiga para evitar cair em versoes
    redirecionadas/editadas que perderam o conteudo original.
    """
    params = {
        "url": url,
        "limit": "5",
        "output": "json",
        "fl": "timestamp,original,statuscode,length",
        "filter": "statuscode:200",
    }
    full = f"{CDX_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full, headers={"User-Agent": "FakenewsBR/2.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        rows = json.loads(r.read().decode("utf-8"))
    if not rows or len(rows) < 2:
        return None
    # rows[0] = header
    dates = [row[0][:8] for row in rows[1:] if row[0]]
    return min(dates) if dates else None  # snapshot mais antigo


def backfill(in_path: Path, out_path: Path, cutoff: str,
             sleep_s: float = 1.1) -> dict:
    """Atualiza date_iso nos registros faltantes ou antigos.

    cutoff: data ISO; registros com date_iso < cutoff ou NaN sao backfilled.
    """
    cutoff_dt = datetime.fromisoformat(cutoff)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counters = {
        "total": 0, "kept": 0, "updated": 0, "no_snapshot": 0, "errors": 0,
    }
    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        rows: list[dict] = []
        # Le em pares (row, provenance)
        while True:
            line = fin.readline()
            if not line:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            prov_line = fin.readline()
            if prov_line:
                rows.append(json.loads(prov_line))

        # Processa apenas linhas de dados (pares)
        i = 0
        while i < len(rows):
            row = rows[i]
            i += 1
            counters["total"] += 1
            try:
                url = row.get("url_review", "")
                date_iso = row.get("date_iso") or ""
                need_backfill = (
                    not date_iso
                    or datetime.fromisoformat(date_iso[:10]) < cutoff_dt
                )
                if need_backfill and url:
                    snap = cdx_lookup(url)
                    if snap:
                        # YYYYMMDD -> YYYY-MM-DD
                        row["date_iso"] = (
                            f"{snap[:4]}-{snap[4:6]}-{snap[6:8]}")
                        counters["updated"] += 1
                    else:
                        counters["no_snapshot"] += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                # Mantem a provenance
                if i < len(rows) and "_provenance" in str(rows[i]):
                    fout.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
                    i += 1
                counters["kept"] += 1
            except Exception as e:
                counters["errors"] += 1
                print(f"  ! erro em linha {counters['total']}: {e}",
                      flush=True)
            time.sleep(sleep_s)
    return counters


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cutoff", default="2010-01-01",
                    help="registros com date_iso < cutoff sao backfilled. "
                         "Default 2010-01-01 para cobrir redes sociais iniciais.")
    ap.add_argument("--sleep", type=float, default=1.1,
                    help="intervalo entre requisicoes (rate limit)")
    args = ap.parse_args()

    print(f"backfill via Wayback CDX")
    print(f"  in={args.in_path}")
    print(f"  out={args.out_path}")
    print(f"  cutoff={args.cutoff}")
    counters = backfill(args.in_path, args.out_path, args.cutoff, args.sleep)
    for k, v in counters.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
