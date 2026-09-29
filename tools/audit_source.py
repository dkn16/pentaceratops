#!/usr/bin/env python3
"""Read-only check that imported reference source files remain unchanged."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] / "docs/source_manifest.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    failures = []
    for entry in manifest["files"]:
        path = Path(entry["source"])
        if not path.is_file():
            failures.append({"source": str(path), "status": "missing"})
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["source_sha256"]:
            failures.append({"source": str(path), "status": "changed"})
    print(json.dumps({"files_checked": len(manifest["files"]), "differences": failures}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
