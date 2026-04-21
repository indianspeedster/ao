"""Convert bench_2d_3d_grouped_gemm.py tabulate output to a markdown table.

Usage:
    python tools/bench_table.py bench.log > bench_table.md
"""
import re
import sys


def main():
    if len(sys.argv) != 2:
        sys.exit(0)
    lines = open(sys.argv[1]).read().splitlines()
    try:
        i = next(
            k for k, line in enumerate(lines)
            if re.match(r"\s*E\s+M\s+N\s+K\s+bf16", line)
        )
    except StopIteration:
        sys.exit(0)
    header = lines[i].split()
    rows = []
    for line in lines[i + 2:]:
        if not line.strip() or line.strip().startswith("#"):
            break
        cells = line.split()
        if len(cells) == len(header):
            rows.append(cells)
    out = []
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
