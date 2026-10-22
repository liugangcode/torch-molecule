"""Refresh the shared README and documentation figures from the dataset hub."""

import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path
from urllib.request import urlopen


HUB_URL = "https://huggingface.co/datasets/liuganghuggingface/molecular-datasets-index/resolve/main/hub.json"
OUTPUT = Path(__file__).parent / "source" / "_static"


def statistics_svg(hub, compact=False):
    summary = hub["summary"]
    domains = Counter(dataset["domain"] for dataset in hub["datasets"])
    published = sum(bool(dataset["links"].get("huggingface")) for dataset in hub["datasets"])
    metrics = [
        (f'{summary["catalog_entries"]:,}', "datasets"),
        (f"{published:,}", "on Hugging Face"),
        (f'{summary["rows"] / 1_000_000:.1f}M', "source rows"),
        (f'{summary["datasets_with_processed_tables"]:,}', "with processed tables"),
        (f'{summary["processed_tables"]:,}', "processed tables"),
        (f'{summary["distinct_tasks"]:,}', "distinct tasks"),
    ]
    width, columns, header = (360, 2, 88) if compact else (960, 3, 72)
    padding = 24 if compact else 32
    rows = len(metrics) // columns
    footer = header + rows * 92 + 12
    height = footer + (116 if compact else 80)
    cell_width = (width - 2 * padding) / columns
    description = (
        f'Snapshot {hub["snapshot"]}: {summary["catalog_entries"]:,} datasets, '
        f'{published:,} published on Hugging Face, {summary["rows"]:,} source rows, '
        f'{summary["datasets_with_processed_tables"]:,} with processed tables, '
        f'{summary["processed_tables"]:,} processed tables, and {summary["distinct_tasks"]:,} distinct tasks. '
        "Source rows may include repeated structures. Each dataset retains its own license."
    )
    metadata = {"snapshot": hub["snapshot"], "datasets": summary["catalog_entries"],
                "published_datasets": published}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        f'<metadata id="dataset-statistics">{escape(json.dumps(metadata))}</metadata>',
        '<title id="title">Molecular Dataset Hub statistics</title>',
        f'<desc id="description">{escape(description)}</desc>',
        """<style>
          :root { --bg: #faf9f5; --ink: #141413; --muted: #6b6a64; --line: #e6e3d8; --clay: #c96442; }
          @media (prefers-color-scheme: dark) {
            :root { --bg: #1d1c1a; --ink: #f2f0e8; --muted: #a29f95; --line: #33312d; --clay: #e08a6a; }
          }
          text { fill: var(--ink); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
          .display { font-family: Georgia, serif; }
          .muted { fill: var(--muted); }
          .accent { fill: var(--clay); }
        </style>""",
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="14" '
        'fill="var(--bg)" stroke="var(--line)"/>',
        f'<text x="{padding}" y="38" class="display" font-size="26">Molecular Dataset Hub</text>',
    ]
    if compact:
        parts.append(f'<text x="{padding}" y="64" class="muted" font-size="16">Snapshot {escape(hub["snapshot"])}</text>')
    else:
        parts.append(f'<text x="{width - 32}" y="36" text-anchor="end" class="muted" '
                     f'font-size="15">Snapshot {escape(hub["snapshot"])}</text>')
    for index, (value, label) in enumerate(metrics):
        x = padding + (index % columns) * cell_width
        y = header + (index // columns) * 92 + 38
        if compact and label == "with processed tables":
            label = "with tables"
        parts.extend([
            f'<text x="{x:g}" y="{y}" class="display" font-size="40">{value}</text>',
            f'<text x="{x:g}" y="{y + 27}" class="muted" font-size="18">{label}</text>',
        ])
    parts.append(f'<path d="M{padding} {footer} H{width - padding}" stroke="var(--line)"/>')
    if compact:
        parts.extend([
            f'<text x="{padding}" y="{footer + 30}" font-size="18">{domains["small_molecules"]:,} small-molecule datasets</text>',
            f'<text x="{padding}" y="{footer + 58}" font-size="18">{domains["polymers"]:,} polymer datasets</text>',
            f'<text x="{padding}" y="{footer + 90}" class="accent" font-size="18">Explore the interactive map ↗</text>',
        ])
    else:
        parts.extend([
            f'<text x="{padding}" y="{footer + 30}" font-size="18">'
            f'{domains["small_molecules"]:,} small-molecule · {domains["polymers"]:,} polymer datasets</text>',
            f'<text x="{padding}" y="{footer + 56}" class="accent" font-size="16">Explore the interactive dataset map ↗</text>',
        ])
    parts.append("</svg>\n")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Read a saved hub.json instead of the published release.")
    args = parser.parse_args()
    if args.input:
        hub = json.loads(args.input.read_text())
    else:
        with urlopen(HUB_URL, timeout=30) as response:
            hub = json.load(response)
    for compact, suffix in [(False, ""), (True, "-compact")]:
        path = OUTPUT / f"dataset-statistics{suffix}.svg"
        path.write_text(statistics_svg(hub, compact), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
