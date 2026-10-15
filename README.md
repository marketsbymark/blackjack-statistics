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
- $25 table minimum; every flat base bet must be at least $25
- Full flat bet required before every round; no smaller all-in or rounded-up bets
- Exact integer-cent bankroll accounting; bets use $0.02 increments so 3:2 payouts are exact

When the player cannot afford a recommended split or double, the engine falls
back to the best legal lower-exposure basic-strategy action.

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

- Terminal: input echo, measured EV/SD, house edge, exposure statistics,
  round-level split/double/DAS probabilities, action counts per round, TTL
  percentiles, ruin probabilities, all-player ending bankroll, and survivor-only
  bankroll summary.
- `bankroll_simulation.xlsx`, with sheets:
  - `Inputs` - parameters, rule set, seed, timestamp
  - `Engine_Stats` - measured EV/SD, exposure, round event rates, and action rates
  - `Summary` - time-to-live and ruin summary statistics
  - `TTL_Distribution` - histogram of time-to-ruin in minutes
  - `Survival_Curve` - P(not yet ruined) at hand checkpoints
  - `Sample_Trajectories` - sample bankroll paths frozen at ruin

## Model notes

Ruin means the bankroll is below the full configured flat bet, including at the
start or after the final round. At a $25 flat bet, $25.00 can play and $24.99
cannot. A $12.50 remainder is retained and the player is done. At a $50 flat
bet, a $37.50 balance also stops: the flat bet never shrinks. Starting below
the flat bet produces zero rounds and a constant bankroll trajectory.
Within a round, the player only takes split and double actions that can be
funded by the bankroll currently available at the table.


Reported EV and loss per active hour are sample estimates from the rounds
actually played, including bankroll constraints. They are not the theoretical
house edge or a guaranteed hourly loss. A short simulation can show a player
advantage through sampling variation. Split/double event rates are per initial
round; action rates can count multiple actions in the same round.

Session-capped duration statistics include survivors at the horizon; their actual
time to ruin is unknown. The time-to-ruin histogram includes stopped players
only, and its probabilities use all simulated players as the denominator.
Duration-specific ruin probabilities are reported only within the simulated
horizon. Session lengths use only complete rounds (floor, never rounding up).

The dashboard displays the parameters of its last run and warns when sidebar
inputs have changed. The old fixed-outcome VBA model is retired: its public
entry points display an explanation without changing workbook results. It could
not correctly fund split/double decisions and its state rounding and bankroll
ceiling made its “exact” results incompatible with the current card engine.
Use Python's Excel export for current results.

Run rule, accounting, reporting, and dashboard checks:

```sh
python -m unittest discover -s tests -v
```
