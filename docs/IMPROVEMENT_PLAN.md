# Improvement Plan (review of commit 59143a0)

Four parallel read-only reviews (engine rules, Monte Carlo statistics, dashboard
and docs, tests/performance) were run against `59143a0`. This file records the
findings and the phased plan. Status is updated as phases land.

Conventions used below: EV is **per initial round, in base-bet units** (a
split/double adds exposure but not to the denominator). SE is one standard
error. House-edge references for 6D H17 DAS no-surrender, resplit to 4, no RSA
are approximate (~0.6-0.65%) and should be checked against a cited source
before being quoted.

## Headline

The engine had two strategy bugs with offsetting EV effects, so the aggregate
looked plausible. 10M rounds, effectively unlimited bankroll:

| Engine | EV / round | SD |
|---|---|---|
| As of 59143a0 | -0.577% +/- 0.037% | 1.160 |
| Bugs 1.1 + 1.2 (+1.3) fixed | -0.683% +/- 0.037% | 1.160 |

## Phase 1 - Engine correctness

1. **Split aces:** the first ace hand re-entered the inner action loop and could
   hit/double (only the inserted second hand was frozen).
   Repro: `[1,6,1,10,5,7,10,10]` -> `doubles=1, das=1, exposure=3.0`.
   (`bankroll_sim.py` `play_round`)
2. **Soft 18 without double available** (3+ cards, or unfunded double fallback)
   hit vs 2-6. Correct: stand vs 2-8, hit vs 9/10/A. (`_soft_strategy`)
3. **Soft 19 vs 6** should double (H17 chart) when allowed. (`_soft_strategy`)
4. **Float bankroll:** dollars compared with `bankroll < bet`; non-binary-exact
   bets (e.g. $0.30, $7.30) produce false ruin / refused splits. Track bankroll
   in integer half-bet units internally.
5. **Bankroll < bet at start:** sample paths left as `np.empty` garbage.
   Fill the path; dashboard should not allow bankroll < bet.

## Phase 2 - Tests

- Full strategy-grid regression (hard/soft/pair x upcard, double allowed/not).
- Fix vacuous tests: split-aces test (both aces drew 10s), split-cap test
  (never exercised `play_round`).
- Add: dealer BJ, BJ vs BJ, player bust (dealer draws nothing), resplit to 4 and
  cap, resplit gated by bankroll, bankroll == bet, survival monotone, Excel
  sheets present.
- Replace the +/-20 sigma EV test with a tighter slow test.

## Phase 3 - Statistical presentation

1. Report Monte Carlo error: Wilson 95% CI for ruin / survival, SE for EV.
2. Rename "House edge" -> realized EV per round (bankroll-constrained). Gating
   biases EV: -2.65% at 1 unit bankroll vs -0.70% at 100 units (1M rounds each).
3. TTL is right-censored at `max_hands`; median/p75 printed as 480 were just the
   cap. Use survival-curve (Kaplan-Meier, here = empirical survival since all
   censoring is at max_hands) quantiles; report "> max_hands" when not reached.
   TTL histogram: ruined paths only.
4. Ruin-by-hours rows beyond the session length (terminal/Excel) were mislabeled.
5. Dashboard median band was over plotted subset only; compute over all sims.
   Remove theoretical +/-8 units/round lines (off-chart / unreachable -$100).
6. Survival curve: include hand 0, dedupe checkpoints. Dealer bust rate label.

## Phase 4 - Dashboard bugs / UX

- `st.metric` delta strings like `"$-123"` render an up arrow; pass numbers.
- Negative seed crashes `default_rng`; `min_value=0`.
- Stale results when inputs change without pressing Run; show warning.
- Ruin-by-duration table: always include full session row.
- Plot trajectories as one trace with NaN separators.

## Phase 5 - Performance / memory

- Per-round Python lists (~0.6 GB at 9.6M rounds) -> running sums.
  Drop unused `SimResult.round_units`.
- `hand_value` without generator (~25% faster, identical results).
- Optional: multiprocessing with `SeedSequence.spawn` (~2x on 4 cores).

## Phase 6 - Maintainability / hygiene

- Remove dead code (`validate_pmf`, `sims_per_batch`, `stats_block` duplicate).
- Single shared ruin-by-hours helper used by terminal, Excel and app.
- `BankrollMonteCarlo.bas` is the legacy PMF model: move to `legacy/`.
- Untrack `.vscode/settings.json`; add `.vscode/` to `.gitignore`.
- README accuracy (terminal output list).
- Later: split module into engine/simulate/reporting; pinned requirements; CI.

Not planned (nits): split second-card draw order, cut-card effect (~0.01-0.02%),
theoretical mid-round reshuffle path.
