# Blackjack Bankroll Monte Carlo

Estimates how long a flat-betting blackjack bankroll lasts under realistic
casino rules. The simulator now uses a card-level blackjack engine instead of
a fixed outcome table, so splits, double downs, double-after-split, and the
extra bankroll exposure required by those decisions are modeled directly.

## Rules modeled

- 6-deck shoe
- Dealer hits soft 17 (H17)
- Double after split allowed (DAS)
- Blackjack pays 3:2
- No surrender, no insurance, no side bets, no counting
- Split up to 4 total hands
- Split aces receive one card only and cannot be re-split
- Embedded H17/DAS basic strategy
- Flat base bet with table-realistic bankroll gating

When the player cannot afford a recommended split or double, the engine falls
back to the best legal lower-exposure basic-strategy action.

## Tests

```powershell
python -m unittest discover -s tests
```

The suite includes a full basic-strategy grid check against a 4-8 deck H17 DAS
chart. A slow ~1M-round EV check is opt-in:
`BLACKJACK_SLOW_TESTS=1 python -m unittest discover -s tests`.

## Install

```powershell
pip install -r requirements.txt
```

## Run

Terminal simulation:

```powershell
python bankroll_sim.py
```

Streamlit dashboard:

```powershell
streamlit run app.py
```

You can configure:

- Dollars per hand
- Starting bankroll
- Minutes per hand
- Number of simulations
- Session length in hours; hands per simulation are derived from session length
  and minutes per hand
- Random seed
- Number of plotted sample trajectories

## Outputs

- Terminal: input echo, realized EV per round with 95% CI, SD, exposure
  statistics, round-level split/double/DAS probabilities, action counts per
  round, time-to-ruin quantiles (censoring-aware), ruin probabilities by
  duration with 95% Wilson intervals, all-player ending bankroll, and
  survivor-only bankroll summary.
- `bankroll_simulation.xlsx`, with sheets:
  - `Inputs` - parameters, rule set, seed, timestamp
  - `Engine_Stats` - measured EV/SD, exposure, round event rates, and action rates
  - `Summary` - time-to-ruin quantiles, ruin rate with CI, ruin by duration
  - `TTL_Distribution` - histogram of time-to-ruin in minutes for ruined
    sessions, plus one row counting sessions that survived
  - `Survival_Curve` - P(not yet ruined) and all-simulation bankroll
    10th/50th/90th percentiles at hand checkpoints
  - `Sample_Trajectories` - sample bankroll paths frozen at ruin

## Model notes

All EV figures are per initial round in base-bet units (a split or double adds
exposure, not rounds). The reported EV is the *realized, bankroll-constrained*
EV: near ruin the player cannot fund every split or double, which lowers EV
relative to the unconstrained house edge. Its 95% CI uses a standard error
clustered by session.

Time-to-ruin quantiles treat survivors as censored at the session length. A
quantile that is not reached within the session is reported as
"> max_hands" rather than the session cap.

Ruin probabilities carry 95% Wilson score intervals; at 1,000 simulations a
ruin rate near 50% is only known to about +/-3 percentage points.

Ruin is defined as bankroll dropping below one base bet before the next round.
Bankroll is tracked internally in base-bet units, so fractional bets such as
$0.30 do not trigger false ruin from floating-point rounding.
Within a round, the player only takes split and double actions that can be
funded by the bankroll currently available at the table.

## Legacy

`legacy/BankrollMonteCarlo.bas` is the earlier Excel/VBA model driven by a
fixed outcome-probability table. It does not use the card-level engine and is
kept for reference only.
