from __future__ import annotations

import argparse
import sys

from src.ingestion.inference import infer_figure_summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Debug visual inference using the same request path as ingest figure summary."
    )
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--caption", default="", help="Optional figure caption")
    parser.add_argument(
        "--footnote",
        action="append",
        default=[],
        help="Optional footnote. Repeat this flag to provide multiple footnotes.",
    )
    parser.add_argument(
        "--llm-backend",
        choices=("api", "ollama"),
        default="api",
        help="Visual inference backend",
    )
    args = parser.parse_args()

    try:
        summary = infer_figure_summary(
            asset_path=args.image_path,
            caption=args.caption,
            footnotes=args.footnote,
            llm_backend=args.llm_backend,
        )
    except Exception as exc:
        print(f"visual inference failed: {exc}", file=sys.stderr)
        return 1

    print(summary)
    print(f"\nchars={len(summary)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
