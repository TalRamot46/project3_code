# `src/imc/sio2/`: the Back SiO₂ problem

Everything specific to the SiO₂ foam problem is in this folder: the original standalone IMC script, the drive table it reads, and the same problem rebuilt on the generic solver in `src/imc/core/` (see `md/imc_code/imc_core.md`).

## 1. The problem

A heat wave driven into a low-density silica foam by a hohlraum-like radiation bath. It is the opening benchmark of the IMC plan because it is not optically thick: transport effects are visible.

| | |
|---|---|
| Geometry | 1-D slab, L = 1 mm, 100 uniform cells, unit transverse area |
| Material | SiO₂ foam, ρ = 0.05 g/cm³, gray, pure absorption (σ_P = σ_R) |
| EOS | u_m = 8.78 ρ^0.91 θ^1.10 MJ/cm³, with θ = T/(100 eV) |
| Opacity | σ = 9175 ρ^1.75 θ^−3.53 cm⁻¹ |
| Initial state | T_m = 300 K, no radiation |
| Left boundary | isotropic blackbody bath at T_D(t) from `back_pulse.txt` |
| Right boundary | vacuum |
| Output | the flux leaving the right face, normalized to σ_SB·(165 eV)⁴ |

**Units:** cm, ns, MJ and θ = T/(100 eV), so the radiation constant is a·(100 eV)⁴ ≈ 1.372×10⁻³ MJ/cm³ and c = 29.979 cm/ns.

## 2. The files

### `sio2_1mm_gray_imc_check_og.py`: the original, frozen

The standalone script this project started from. **Do not edit it.** It is the regression baseline that everything else is checked against.
- One fix was applied before it was frozen: the initial temperature array `np.full(1, THETA0)` became `np.full(N_CELLS, THETA0)`. Before the fix, the energy check read about 80 and only cell 0 emitted on the first step.
- Its docstring was also corrected (analog absorption, synthetic drive).
- `md/theory/imc_mc_derivation.md` Appendix A maps each of its lines to F&C's equations.

Run it with:

```bash
uv run python src/imc/sio2/sio2_1mm_gray_imc_check_og.py --prefix results/sio2_frozen
```

It writes `<prefix>.npz`, `<prefix>_Fout.png` and `<prefix>_summary.txt`.

### `back_pulse.txt`: the drive

Two columns: t [ns] and T_D [eV], 29 lines including the header. **It is synthetic, not measured:** a smooth stand-in for a ~3 ns laser-driven hohlraum pulse (fast rise, ~190 eV flat top, slow decay). Before its first entry the drive is padded back to t = 0 at 300 K. Both IMC versions and the S_N twin `src/sn/sn.py` read this same file.

### `problem.py`: the same problem on `imc.core`

The same physics, constants, defaults and outputs as the frozen script, built from the core modules: `MATERIAL` is a `PowerLawMaterial` in θ units, `make_bath()` builds the drive, and `run_one(...)` runs one seed. Its differences from the frozen script don't change the equation being solved:

1. **Bath birth times follow the drive within each step** instead of being uniform. `--bath-times uniform` restores the old choice.
2. **The census comb is stratified, not systematic.** It never triggers at the defaults.
3. **Packets carry their cell index**, so there is no 10⁻¹⁰ Δx nudge at faces.
4. **Extra diagnostics:** the per-cell energy ledger, the number of capped histories, and `--fleck-mode explicit` (milestone M2a).

Run it with:

```bash
uv run python -m imc.sio2.problem --prefix results/sio2_pkg
uv run python -m imc.sio2.problem --fleck-mode explicit --seeds 1729 --prefix results/sio2_explicit
```

Defaults, as in the frozen script: Δt = 1 ps, t_end = 3 ns, 4000 bath packets and a target of 2500 emission packets per step, `max_census` 120000, seeds 1729, 2718 and 3141, smoothing width 0.02 ns. The summary file also reports the peak's standard error, the maximum per-cell ledger residual and the number of capped histories.

## 3. Reference results (default settings)

| Run | Breakout (flux ≥ 0.01) | Peak (normalized) | Max energy error | Wall time |
|---|---|---|---|---|
| Frozen script | 1.47 ns | 0.954 at 1.91 ns | 5.3×10⁻¹⁴ | 59 s |
| `problem.py` | 1.47 ns | 0.957 ± 0.013 at 1.97 ns | 1.4×10⁻¹³ | 45 s |
| S_N twin (`src/sn/sn.py`, 16 ordinates) | 1.4675 ns | 0.9506 at 1.95 ns | 5×10⁻¹⁴ | 4 s |

**How the curves compare.** These are 0.05 ns block means for t > 1.5 ns, with errors from the spread between seeds.
- `problem.py` vs. the frozen script: χ²/dof = 1.4 over 30 blocks. With 3 seeds the error estimate has only 2 degrees of freedom, so this is consistent.
- Both IMC versions vs. S_N: mean bias −0.04% and rms difference 0.3%, which is the size of the statistical error.

**Explicit mode** (f = 1) also runs stably here and gives the same answer (breakout 1.47 ns, peak 0.960 for one seed). That is expected: the explicit scheme is unstable only when β = du_r/du_m > 1, and in this foam β ≲ 0.05 because radiation stores far less heat than the material.

## 4. Tests

- `tests/imc/test_basics.py` checks that `problem.py`'s material, packet allocation and bath energy are identical to the frozen script's own functions.
- `tests/imc/test_sio2_regression.py` (marked slow, about 1 min) runs the default 3-seed calculation and requires:
  - breakout 1.47 ± 0.011 ns;
  - the peak within 2.5 standard errors of 0.9538;
  - energy error < 10⁻¹², per-cell ledger < 10⁻¹⁰, no capped histories;
  - agreement with S_N to < 0.5% mean bias and < 1% rms.

## 5. Notes

- **Stale files in the repo root.** `sio2_1mm_gray_imc_check.npz`, `_Fout.png` and `_summary.txt` there were written before the theta fix. Their energy error of about 80 is that bug, not a property of the current code.
- **Numba cache clash.** `sn.py` and the frozen script use `cache=True`. If a module's cache was written when it was imported as a package module (e.g. `sn.sn` from the tests) and the file is later run as a script, a stale cache can fail with "No module named 'sn.sn'". Deleting that folder's `__pycache__` fixes it.
- **Typical system size.** The advisor asked to confirm L, ρ and the drive for a typical system with the colleague named in the meeting notes (plan Addendum A3), before this becomes the M4 benchmark preset.
