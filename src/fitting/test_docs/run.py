import bootstrap  # noqa: F401  (sys.path)
import sys, os, subprocess, json, shutil, logging
logging.disable(logging.CRITICAL)
from compute import compute, verify, TESTS
import figs, latex
from paths import DOCS_OUT as OUT

def run(k):
    R = compute(k); R["verify"] = verify(R)
    d = os.path.join(OUT, TESTS[k]["short"]); os.makedirs(os.path.join(d, "figs"), exist_ok=True)
    stem = f"test{k}_formulas"
    fn = dict(ss="figs/selfsimilar.png", m="figs/profiles_m.png", x="figs/profiles_x.png", fronts="figs/fronts.png")
    figs.selfsimilar(R, os.path.join(d, fn["ss"])); figs.dimensional(R, os.path.join(d, fn["m"]), os.path.join(d, fn["x"])); figs.fronts(R, os.path.join(d, fn["fronts"]))
    open(os.path.join(d, stem + ".tex"), "w").write(latex.document(R, fn))
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem + ".tex"], cwd=d, capture_output=True, text=True)
    ok = os.path.exists(os.path.join(d, stem + ".pdf")) and r.returncode == 0
    print(f"Test {k}: pdflatex {'OK' if ok else 'FAILED'}")
    if not ok: print(r.stdout[-3000:])
    # machine-readable summary
    summ = dict(sub_fit={v: {kk: vv for kk, vv in p.items()} for v, p in R["sub_fit"].items()},
                shock_fit={v: {kk: vv for kk, vv in p.items()} for v, p in R["shock_fit"].items()},
                scales=R["sc"], verify=[{str(kk): vv for kk, vv in row.items()} for row in R["verify"]])
    json.dump(summ, open(os.path.join(d, f"test{k}_fit_parameters.json"), "w"), indent=1, default=float)
    return R

if __name__ == "__main__":
    for k in map(int, sys.argv[1:]): run(k)
