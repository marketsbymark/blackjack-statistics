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

Ruin is defined as bankroll dropping below one base bet before the next round.
Within a round, the player only takes split and double actions that can be
funded by the bankroll currently available at the table.
