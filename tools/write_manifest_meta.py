#!/usr/bin/env python
"""Write the sidecar manifest.meta.json for a hand-assembled manifest.jsonl.

validate_manifest() requires <dir>/manifest.meta.json and checks
manifest_sha256 == sha256(manifest.jsonl) and num_shards. stream_preprocess emits
this automatically, but a manifest built by concatenating shards (e.g.
build_cpt_manifest.sh) has none, so we compute it here.

  python tools/write_manifest_meta.py <manifest.jsonl> [--version 0.1.0-cpt-pool]
"""
import json, os, sys, hashlib, datetime, argparse
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--version", default="0.1.0-cpt-pool")
    args = ap.parse_args()

    h = hashlib.sha256()
    nb = nr = nt = ns = 0
    tok = defaultdict(int)
    cnt = defaultdict(int)
    with open(args.manifest, "rb") as f:
        for raw in f:
            h.update(raw)
            o = json.loads(raw)
            nb += o.get("bytes", 0)
            nr += o.get("num_records", 0)
            nt += o.get("estimated_tokens", 0)
            ns += 1
            tok[o["source"]] += o.get("estimated_tokens", 0)
            cnt[o["source"]] += 1

    meta = {
        "manifest_version": args.version,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "manifest_sha256": h.hexdigest(),
        "num_shards": ns,
        "total_bytes": nb,
        "total_records": nr,
        "total_estimated_tokens": nt,
    }
    out = os.path.join(os.path.dirname(args.manifest), "manifest.meta.json")
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out}: num_shards={ns} records={nr:,} est_tok={nt/1e9:.2f}B "
          f"sha256={meta['manifest_sha256'][:12]}..")
    for s in sorted(tok, key=lambda x: -tok[x]):
        print(f"  {s:24s} {cnt[s]:6d} shards  {tok[s]/1e9:7.2f}B est")
    print(f"  {'TOTAL':24s} {ns:6d} shards  {nt/1e9:7.2f}B est")


if __name__ == "__main__":
    main()
