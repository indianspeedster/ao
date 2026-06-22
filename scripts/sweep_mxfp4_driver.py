"""Drive per-shape MXFP4-opt sweeps across 8 GPUs (8 shapes concurrent).

Collects best config per shape, prints a table + a pastable _BEST_CFGS dict
keyed by (E, N, K).
"""
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

NGPU = 8
WORKER = os.path.join(os.path.dirname(__file__), "sweep_mxfp4_worker.py")
PY = sys.executable

LLAMA4 = list(itertools.product([1, 2, 4, 8], [16640], [2048, 5120, 8192], [2048, 5120, 8192]))
DSV3 = [(4, 32768, 2048, 7168), (8, 32768, 2048, 7168),
        (4, 128000, 2048, 7168), (8, 128000, 2048, 7168)]

shapes = ([(e, m, n, k) for e, m, n, k in LLAMA4] +
          [(e, m, n, k) for e, m, n, k in DSV3])
if len(sys.argv) > 1 and sys.argv[1] == "dsv3":
    shapes = [(e, m, n, k) for e, m, n, k in DSV3]

_slots = list(range(NGPU))


def run_shape(idx_shape):
    idx, (E, M, N, K) = idx_shape
    gpu = idx % NGPU
    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu))
    try:
        out = subprocess.run(
            [PY, WORKER, str(E), str(M), str(N), str(K)],
            env=env, capture_output=True, text=True, timeout=1800,
        )
        for line in out.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return {"E": E, "M": M, "N": N, "K": K, "error": out.stderr[-300:]}
    except Exception as ex:  # noqa: BLE001
        return {"E": E, "M": M, "N": N, "K": K, "error": str(ex)}


def main():
    results = [None] * len(shapes)
    with ThreadPoolExecutor(max_workers=NGPU) as ex:
        for r in ex.map(run_shape, list(enumerate(shapes))):
            results[shapes.index((r["E"], r["M"], r["N"], r["K"]))] = r
            tag = (f"{r.get('speedup','ERR')}x  {r['E']},{r['M']},{r['N']},{r['K']}"
                   if "cfg" in r else f"ERR {r['E']},{r['M']},{r['N']},{r['K']}: {r.get('error','')[:120]}")
            print(tag, flush=True)

    import math
    print("\n==== best config per shape ====")
    geo = 0.0
    n = 0
    print("_BEST_CFGS = {")
    for r in results:
        if "cfg" not in r:
            continue
        c = r["cfg"]
        print(f"    ({r['E']}, {r['N']}, {r['K']}): dict("
              f"BLOCK_M={c['BLOCK_M']}, BLOCK_N={c['BLOCK_N']}, BLOCK_K={c['BLOCK_K']}, "
              f"GROUP_M={c['GROUP_M']}, num_warps={c['num_warps']}, num_stages=2, "
              f"waves_per_eu={c['waves_per_eu']}, matrix_instr_nonkdim={c['matrix_instr_nonkdim']}),"
              f"  # {r['speedup']}x  ({r['best_us']}us vs mxfp8 {r['mxfp8_us']}us)")
        geo += math.log(r["speedup"]); n += 1
    print("}")
    if n:
        print(f"\nGeomean best-config speedup vs MXFP8 ({n} shapes): {math.exp(geo/n):.3f}x")


if __name__ == "__main__":
    main()
