# Test 3's cold start against the density exponent ω

Question: the cold-start runs of Test 3 (gold, ω = +0.5) fall below the solver's entropy because
matter and radiation are not in equilibrium during the first ~0.1 ns
([test3_on_the_solver.md](test3_on_the_solver.md) § 2.2). Does a flatter initial density
(ω → 0.05) make the matter equilibrate faster, so that IMC converges to the solver?

**Answer: no.** The deficit is the same for every ω from 0.5 to 0.05, in IMC and in 2T diffusion.
The surface matter temperature follows the same curve in time for all six ω.

## 1. The runs

Gold, T_s = 1 HeV, paper bath, 30 K target, cold start from t = 0, to 1 ns. Only ω changes;
each run is compared with the `AblationSolver` built for its own ω.

| | |
|---|---|
| ω | 0.5 (the Test 3 runs of `test3_on_the_solver.py`, read at their 1 ns snapshot), 0.4, 0.3, 0.2, 0.1, 0.05 |
| IMC | 3 seeds per ω, packets as in Test 3 |
| 2T diffusion | χ = 1, one run per ω |
| Mesh | `log_mass`; surface part as Test 3's `MESH` (first cell 4·10⁻⁷ g/cm², 5% growth to 0.0216 mg/cm², until 1.5 mg/cm²); beyond it 0.3% growth up to 1/1060 of the domain |
| Domain | 1.1 × the solver's shocked mass at 1 ns |
| Cost | 25,000 steps (ω = 0.05) to 211,000 (ω = 0.4); 1.7 million for ω = 0.5. IMC 2–15 min per run, 2T 0.4–2.5 min |
| Energy error | 2T ≤ 3·10⁻¹³, IMC ≤ 1.3·10⁻¹⁰ |

In the solver the heat wave does not depend on ω: the ablated mass (1.019 mg/cm² at 1 ns) and
the ablation pressure (2.73 Mbar at 1 ns) are identical for all six. Only the shock depends on ω:

| ω | shocked mass at 1 ns [mg/cm²] | shocked / ablated mass | exponent a in m_shock ∝ t^a |
|---|---|---|---|
| 0.50 | 202.8 | 199 | 0.517 |
| 0.40 | 96.4 | 95 | 0.582 |
| 0.30 | 51.2 | 50 | 0.639 |
| 0.20 | 29.6 | 29 | 0.690 |
| 0.10 | 18.4 | 18 | 0.735 |
| 0.05 | 14.9 | 15 | 0.756 |

## 2. Results at 1 ns

Entropy p/ρ^γ over the solver's. "Shocked at t" is the mass element the solver's shock passed at
time t. Region: from 2 × the heat front to 98.5% of the nearer shock.

**IMC (mean of 3 seeds)**

| ω | deepest deficit | shocked at 0.05 ns | 0.1 ns | 0.2 ns | 0.5 ns | 0.8 ns | shock mass / solver's |
|---|---|---|---|---|---|---|---|
| 0.50 | 0.782 | 0.836 | 0.873 | 0.908 | 0.931 | 0.958 | 0.977 |
| 0.40 | 0.779 | 0.835 | 0.869 | 0.905 | 0.936 | 0.948 | 0.975 |
| 0.30 | 0.772 | 0.830 | 0.869 | 0.904 | 0.932 | 0.955 | 0.972 |
| 0.20 | 0.781 | 0.832 | 0.859 | 0.888 | 0.936 | 0.944 | 0.968 |
| 0.10 | 0.806 | – | 0.849 | 0.891 | 0.925 | 0.928 | 0.962 |
| 0.05 | 0.803 | – | 0.853 | 0.881 | 0.911 | 0.935 | 0.957 |

**2T diffusion, χ = 1**

| ω | deepest deficit | shocked at 0.05 ns | 0.1 ns | 0.2 ns | 0.5 ns | 0.8 ns | shock mass / solver's |
|---|---|---|---|---|---|---|---|
| 0.50 | 0.827 | 0.883 | 0.914 | 0.940 | 0.963 | 0.970 | 0.984 |
| 0.40 | 0.823 | 0.880 | 0.910 | 0.936 | 0.959 | 0.970 | 0.983 |
| 0.30 | 0.816 | 0.877 | 0.907 | 0.933 | 0.957 | 0.965 | 0.980 |
| 0.20 | 0.832 | 0.869 | 0.898 | 0.927 | 0.952 | 0.961 | 0.977 |
| 0.10 | 0.853 | – | 0.887 | 0.915 | 0.942 | 0.954 | 0.971 |
| 0.05 | 0.861 | – | 0.877 | 0.908 | 0.936 | 0.947 | 0.968 |

"–": the element lies inside 2 × the heat front and is not in the compared region. For ω ≤ 0.1
the "deepest deficit" is therefore taken over a region that starts later (elements shocked after
~0.07 ns), which is why it reads 2–3% shallower; at equal shocking time these ω are not better.

What the tables show:

1. At equal shocking time the deficit changes by at most 3% over the whole ω range, and the
   change is in the wrong direction for the hypothesis: lower ω is slightly *further* from the
   solver (2T, element shocked at 0.5 ns: 0.963 at ω = 0.5, 0.936 at ω = 0.05).
2. The shock lags a little more at low ω (IMC 0.977 → 0.957, 2T 0.984 → 0.968).
3. IMC is 4–5% deeper than 2T at every ω, as in Test 3.

## 3. Why ω does not matter

Matter and radiation exchange energy in the ablated layer: the hot, expanded gold between the
surface and the heat front. Its density is set by the ablation pressure and the temperature
(ρ ~ p/T), not by the initial density of the target. Written in mass coordinates, the heat-wave
equations do not contain the initial density at all, which is why the solver's heat solution is
the same for every ω. The equilibration time 1/(β c κ ρ) is evaluated in that layer, so it is
the same too.

The runs confirm it directly (`layer.png`). The surface matter temperature is the same function
of time for all ω, in both codes:

| t | 1 ps | 10 ps | 100 ps | 1 ns |
|---|---|---|---|---|
| IMC, ω = 0.5 … 0.05 [HeV] | 0.635 … 0.646 | 0.795 … 0.801 | 0.933 … 0.940 | 0.987 … 0.994 |
| 2T, ω = 0.5 … 0.05 [HeV] | 0.637 … 0.650 | 0.790 … 0.794 | 0.928 … 0.929 | 0.982 |

The heat front over the solver's is also the same curve for ω ≥ 0.3 (1.10 at 10 ps, 0.98 at
100 ps) and slightly further ahead early on at low ω (1.17 at 10 ps for ω = 0.05).

What ω does change is how the shocked mass is distributed over shocking time. With m_shock ∝ t^a,
the fraction of the 1 ns shocked mass that was shocked during the first 0.1 ns is 0.1^a: 30% at
ω = 0.5 and 18% at ω = 0.05. Lower ω therefore records the early, non-equilibrium phase in a
smaller share of its mass, but the later-shocked mass is slightly worse (point 1), and the
curves against m/m_shock in `convergence.png` end up nearly on top of each other.

## 4. Not explained

- The pressure on the dense side of the heat front, over the solver's p₀t^τ, differs with ω
  during the first 0.1 ns: 0.88 at 10 ps for ω = 0.5 but 1.17 (IMC) and 1.23 (2T) for ω = 0.05.
  The entropy deficit is nevertheless the same. At low ω the shock is only 15 heat-front masses
  ahead and the early heat front is ahead of the solver's, so this sampled pressure is probably
  not the pressure the shock feels; this was not checked.
- Why the later-shocked elements and the shock position are slightly worse at low ω.
- ω = 0.5 is compared at the 1 ns snapshot of its 2 ns runs (t = 0.9985–1.0006 ns).

## 5. Figures

`results/phase_c/figures/test3_omega_scan/`

- `entropy_by_omega.png`: entropy of IMC and 2T against the solver's, one panel per ω.
- `convergence.png`: entropy over the solver's for all ω on one axis, against m/m_shock and
  against the shocking time; deficit and shock mass against ω. (A ratio is used here because
  the absolute entropies of different ω differ by four orders of magnitude.)
- `layer.png`: front pressure and heat front over the solver's and the surface matter
  temperature against time; temperatures at 1 ns.
- `table.md`: the tables of § 2.

## 6. Code and reproduction

- `src/rad_hydro_sim/verification/test3_omega_scan.py`: cases, per-ω solver (cached), runs.
- `src/rad_hydro_sim/verification/test3_omega_scan_figures.py`: figures and table.
- `test3_on_the_solver.py`: `run` split so that `simulate_and_save(path, case, cfg)` is reusable.

```
python -m rad_hydro_sim.verification.test3_omega_scan --run-all --jobs 2     # ~1 h on 2 cores
python -m rad_hydro_sim.verification.test3_omega_scan --figures
```

Run data: `results/test3_omega_scan/` (untracked).
