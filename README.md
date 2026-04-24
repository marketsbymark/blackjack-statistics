# Blackjack Bankroll Monte Carlo

Estimates how long a flat-betting blackjack bankroll lasts under realistic
rules. Answers the question: *"Starting with $500 at $25/hand, 1 hand per
30 seconds, how long does the money last?"*

## Rules modeled

- 6-deck shoe
- Dealer hits on soft 17 (H17)
- Double after split allowed (DAS)
- Blackjack pays 3:2
- Perfect basic strategy, flat bet, no insurance / side bets / counting

Under these rules the house edge is ~0.64% and the per-hand standard deviation
is ~1.142 units.

## Install

```
pip install -r requirements.txt
```

## Run

```
python bankroll_sim.py
```

You will be prompted for:

- Dollars per hand (default 25)
- Starting bankroll (default 500)
- Minutes per hand (default 0.5, i.e. 1 hand / 30 s)
- Number of simulations (default 20000)
- Max hands per simulation (default 20000; ~167 h of play at 30 s/hand)
- Random seed (default 20260424)

Press Enter at any prompt to accept the default.

## Outputs

- Terminal: input echo, PMF sanity check, TTL percentile table (hands /
  minutes / hours), ruin probabilities at 1h / 2h / 4h / 8h of play, and
  mean ending bankroll among survivors.
- `bankroll_simulation.xlsx`, with sheets:
  - `Inputs` — parameters + rule set + seed + timestamp
  - `PMF` — outcome distribution and its EV / SD
  - `Summary` — key statistics
  - `TTL_Distribution` — histogram of time-to-ruin in minutes
  - `Survival_Curve` — P(not yet ruined) at 50 hand checkpoints
  - `Sample_Trajectories` — 50 bankroll paths over time (frozen at ruin)

## Model notes

Per-hand dollar outcomes are sampled from a fixed 8-point PMF calibrated to
published blackjack statistics. This is standard for bankroll / ruin analysis:
a full shoe-and-basic-strategy simulator is 100–1000x slower and does not
change ruin-time estimates once EV and SD are calibrated. The PMF is
validated at startup (EV and SD must match reference values within tolerance).

Ruin is defined as bankroll dropping below one bet (the player can no longer
place the minimum wager), not bankroll reaching $0.
