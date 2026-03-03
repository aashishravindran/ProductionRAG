"""
Validate that all required source PDFs exist and are readable.

Usage:
    python -m data_sourcing.validate_sources
"""

import os
import sys
from pathlib import Path

SOURCES = {
    "github_profile.pdf": "data_sourcing/data/github_profile.pdf",
    "linkedin_profile.pdf": "data_sourcing/data/linkedin_profile.pdf",
}

PDF_MAGIC = b"%PDF-"


def validate_pdf(path: str) -> tuple[bool, str]:
    """Returns (is_valid, reason)."""
    p = Path(path)
    if not p.exists():
        return False, "File not found"
    size = p.stat().st_size
    if size < 1024:
        return False, f"File too small ({size} bytes) — may be empty or corrupt"
    with open(p, "rb") as f:
        header = f.read(5)
    if header != PDF_MAGIC:
        return False, f"Not a valid PDF (bad magic bytes: {header!r})"
    return True, "OK"


def validate_all_sources() -> dict[str, dict]:
    results = {}
    for name, path in SOURCES.items():
        valid, reason = validate_pdf(path)
        size_kb = 0
        if Path(path).exists():
            size_kb = round(Path(path).stat().st_size / 1024, 1)
        results[name] = {
            "path": path,
            "exists": Path(path).exists(),
            "valid": valid,
            "size_kb": size_kb,
            "reason": reason,
        }
    return results


def main() -> None:
    results = validate_all_sources()

    print("\nData Source Validation Report")
    print("=" * 60)
    all_valid = True
    for name, info in results.items():
        status = "PASS" if info["valid"] else "FAIL"
        print(f"\n  [{status}] {name}")
        print(f"         Path:   {info['path']}")
        if info["valid"]:
            print(f"         Size:   {info['size_kb']} KB")
        else:
            print(f"         Reason: {info['reason']}")
            if name == "linkedin_profile.pdf" and not info["exists"]:
                print(
                    "         Action: Export your LinkedIn profile as PDF and place it at:\n"
                    f"                 {info['path']}"
                )
            all_valid = False
    print("\n" + "=" * 60)
    if all_valid:
        print("All sources valid. Ready for ingestion.\n")
    else:
        print("Some sources are missing or invalid. See above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
