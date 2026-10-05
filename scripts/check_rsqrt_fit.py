"""Does a low-degree polynomial stand a chance on v^-1/2? Range is everything."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from baby_mamba.polynomial import fit_general
from fhedlm.polynorm import rsqrt_report

f = lambda v: 1.0 / np.sqrt(v)
print(f"{'degree':>6} {'interval':>22} {'decades':>8} {'max_rel_err':>12}")
for lo, hi in [(0.04, 0.05), (0.01, 0.05), (0.001, 0.05), (1e-4, 1.0), (1e-6, 1.0)]:
    for deg in (3, 5):
        c = fit_general(f, deg, lo, hi, method="chebyshev")
        r = rsqrt_report(c, lo, hi)
        print(f"{deg:>6} {f'[{lo:g}, {hi:g}]':>22} {r['range_decades']:>8.2f} "
              f"{r['max_rel_error']:>12.3e}")
