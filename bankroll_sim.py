"""Blackjack bankroll Monte Carlo simulator.

Rules modeled: 6-deck, H17, DAS, BJ pays 3:2, basic strategy, flat betting.
Per-hand outcomes are drawn from an empirical PMF calibrated so that
E[X] ~= -0.0064 (house edge ~0.64%) and SD[X] ~= 1.142 units.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Per-hand outcome PMF in units of the flat bet.
OUTCOMES = np.array([4.0, 2.0, 1.5, 1.0, 0.0, -1.0, -2.0, -4.0])
PROBS = np.array([0.0010, 0.0520, 0.0475, 0.3383, 0.0848, 0.4319, 0.0430, 0.0015])
OUTCOME_LABELS = [
    "Won double-after-split",
    "Won double / won split",
    "Natural blackjack (3:2)",
    "Ordinary win",
    "Push",
    "Ordinary loss",
    "Lost double / lost split",
    "Lost double-after-split",
]

EXPECTED_EV = -0.0064
EXPECTED_SD = 1.142


def validate_pmf() -> tuple[float, float]:
    total = PROBS.sum()
    assert abs(total - 1.0) < 1e-9, f"PMF probabilities sum to {total}, not 1.0"
    ev = float(np.sum(OUTCOMES * PROBS))
    var = float(np.sum(PROBS * (OUTCOMES - ev) ** 2))
    sd = float(np.sqrt(var))
    assert abs(ev - EXPECTED_EV) < 5e-4, f"EV drift: {ev:.5f} vs {EXPECTED_EV}"
    assert abs(sd - EXPECTED_SD) < 0.02, f"SD drift: {sd:.5f} vs {EXPECTED_SD}"
    return ev, sd


@dataclass
class SimParams:
    bet: float
    bankroll: float
    minutes_per_hand: float
    sims: int
    max_hands: int
    seed: int
    sims_per_batch: int = 2000
    n_trajectories: int = 50


@dataclass
class SimResult:
    ttl_hands: np.ndarray          # shape (sims,), int
    ruined: np.ndarray             # shape (sims,), bool
    ending_bankroll: np.ndarray    # shape (sims,), float
    survival_hands: np.ndarray     # checkpoint hand indices
    survival_prob: np.ndarray      # P(bankroll >= bet) at each checkpoint
    sample_paths: np.ndarray       # (n_samples, max_hands+1) dollars


def simulate(params: SimParams, ev: float, sd: float) -> SimResult:
    rng = np.random.default_rng(params.seed)

    ttl_hands = np.empty(params.sims, dtype=np.int64)
    ruined = np.empty(params.sims, dtype=bool)
    ending = np.empty(params.sims, dtype=np.float64)

    n_checkpoints = 50
    survival_hands = np.linspace(
        1, params.max_hands, n_checkpoints, dtype=np.int64
    )
    survival_hits = np.zeros(n_checkpoints, dtype=np.int64)

    n_samples = min(params.n_trajectories, params.sims)
    sample_paths = np.empty((n_samples, params.max_hands + 1), dtype=np.float64)
    samples_filled = 0

    done = 0
    while done < params.sims:
        b = min(params.sims_per_batch, params.sims - done)
        draws = rng.choice(OUTCOMES, size=(b, params.max_hands), p=PROBS)
        deltas = draws * params.bet
        traj = params.bankroll + np.cumsum(deltas, axis=1)

        # Ruin: first hand index where next bet cannot be placed.
        below = traj < params.bet
        batch_ruined = below.any(axis=1)
        first_below = np.where(
            batch_ruined, below.argmax(axis=1) + 1, params.max_hands
        )

        # Ending bankroll: value at ttl (ruin) or last hand if survived.
        end_idx = first_below - 1
        end_vals = traj[np.arange(b), end_idx]

        # Survival at checkpoint K: sim has NOT been ruined by hand K.
        # first_below is 1-indexed ruin hand. For survivors, we pretend ruin is at max_hands + 1 
        # so they pass the strictly-greater-than check for the final checkpoint.
        survival_check = np.where(batch_ruined, first_below, params.max_hands + 1)
        survival_hits += (survival_check[:, None] > survival_hands[None, :]).sum(axis=0)

        ttl_hands[done:done + b] = first_below
        ruined[done:done + b] = batch_ruined
        ending[done:done + b] = end_vals

        if samples_filled < n_samples:
            take = min(n_samples - samples_filled, b)
            # Freeze each sample path at its ruin hand (bankroll after ruin = end_val).
            sample_paths[samples_filled:samples_filled + take, 0] = params.bankroll
            sample_paths[samples_filled:samples_filled + take, 1:] = traj[:take]
            for j in range(take):
                fb = int(first_below[j])
                if fb <= params.max_hands:
                    # Indices 1..max_hands correspond to hands 1..max_hands.
                    sample_paths[samples_filled + j, fb + 1:] = traj[j, fb - 1]
            samples_filled += take

        done += b

    survival_prob = survival_hits / params.sims
    return SimResult(
        ttl_hands=ttl_hands,
        ruined=ruined,
        ending_bankroll=ending,
        survival_hands=survival_hands,
        survival_prob=survival_prob,
        sample_paths=sample_paths,
    )


def prompt_float(label: str, default: float) -> float:
    raw = input(f"{label} [{default}]: ").strip()
    if not raw:
        return float(default)
    return float(raw)


def prompt_int(label: str, default: int) -> int:
    raw = input(f"{label} [{default}]: ").strip()
    if not raw:
        return int(default)
    return int(raw)


def collect_inputs() -> SimParams:
    print("=" * 60)
    print(" Blackjack Bankroll Monte Carlo")
    print(" Rules: 6-deck, H17, DAS, BJ 3:2 (basic strategy, flat bet)")
    print("=" * 60)
    bet = prompt_float("Dollars per hand", 25)
    bankroll = prompt_float("Starting bankroll ($)", 500)
    minutes_per_hand = prompt_float("Minutes per hand", 0.5)
    sims = prompt_int("Number of simulations", 20000)
    max_hands = prompt_int("Max hands per simulation", 20000)
    seed = prompt_int("Random seed", 20260424)
    if bet <= 0 or bankroll <= 0 or minutes_per_hand <= 0:
        raise ValueError("bet, bankroll, and minutes_per_hand must be positive")
    if bankroll < bet:
        raise ValueError("bankroll must be at least one bet")
    return SimParams(
        bet=bet,
        bankroll=bankroll,
        minutes_per_hand=minutes_per_hand,
        sims=sims,
        max_hands=max_hands,
        seed=seed,
    )


def percentile_row(arr: np.ndarray, multiplier: float = 1.0) -> dict:
    return {
        "mean": float(np.mean(arr)) * multiplier,
        "p10": float(np.percentile(arr, 10)) * multiplier,
        "p25": float(np.percentile(arr, 25)) * multiplier,
        "median": float(np.median(arr)) * multiplier,
        "p75": float(np.percentile(arr, 75)) * multiplier,
        "p90": float(np.percentile(arr, 90)) * multiplier,
    }


def print_terminal_report(
    params: SimParams, ev: float, sd: float, result: SimResult
) -> None:
    ttl_hands = result.ttl_hands
    ttl_minutes = ttl_hands * params.minutes_per_hand
    ttl_hours = ttl_minutes / 60.0
    ruin_rate = float(result.ruined.mean())

    dollar_loss_per_hand = -ev * params.bet
    hands_per_hour = 60.0 / params.minutes_per_hand
    ev_dollar_loss_per_hour = dollar_loss_per_hand * hands_per_hour

    print()
    print("-" * 60)
    print(" Inputs")
    print("-" * 60)
    print(f"  Bet per hand        : ${params.bet:,.2f}")
    print(f"  Starting bankroll   : ${params.bankroll:,.2f}")
    print(f"  Minutes per hand    : {params.minutes_per_hand}")
    print(f"  Simulations         : {params.sims:,}")
    print(f"  Max hands per sim   : {params.max_hands:,}")
    print(f"  Seed                : {params.seed}")

    print()
    print("-" * 60)
    print(" PMF sanity check")
    print("-" * 60)
    print(f"  E[X] (units)        : {ev:+.5f}  (target {EXPECTED_EV:+.4f})")
    print(f"  SD[X] (units)       : {sd:.5f}  (target {EXPECTED_SD:.3f})")
    print(f"  House edge          : {-ev * 100:.3f}%")
    print(f"  EV $ loss per hand  : ${dollar_loss_per_hand:,.4f}")
    print(f"  EV $ loss per hour  : ${ev_dollar_loss_per_hour:,.2f}")

    print()
    print("-" * 60)
    print(" Time-to-Live (TTL)")
    print("-" * 60)
    stats_hands = percentile_row(ttl_hands)
    stats_min = percentile_row(ttl_minutes)
    stats_hr = percentile_row(ttl_hours)
    header = f"  {'stat':<8}{'hands':>12}{'minutes':>14}{'hours':>12}"
    print(header)
    for key in ("mean", "p10", "p25", "median", "p75", "p90"):
        print(
            f"  {key:<8}{stats_hands[key]:>12,.1f}"
            f"{stats_min[key]:>14,.1f}{stats_hr[key]:>12,.2f}"
        )

    print()
    print("-" * 60)
    print(" Ruin probability by play duration")
    print("-" * 60)
    for hours in (1, 2, 4, 8):
        hand_target = int(round(hours * hands_per_hour))
        if hand_target <= 0:
            continue
        # P(ruined by hand_target) = fraction of sims with ttl <= hand_target AND ruined.
        ruined_by = np.mean(result.ruined & (ttl_hands <= hand_target))
        print(
            f"  by {hours:>2}h ({hand_target:>6,} hands) : "
            f"P(ruin) = {ruined_by * 100:5.2f}%"
        )
    print(f"  overall ruin rate (<= {params.max_hands:,} hands): "
          f"{ruin_rate * 100:.2f}%")

    survivors = result.ending_bankroll[~result.ruined]
    if survivors.size:
        print(
            f"  mean ending bankroll among survivors: "
            f"${float(np.mean(survivors)):,.2f}"
        )


def write_excel(
    params: SimParams,
    ev: float,
    sd: float,
    result: SimResult,
    path: str,
) -> None:
    ttl_hands = result.ttl_hands
    ttl_minutes = ttl_hands * params.minutes_per_hand
    ttl_hours = ttl_minutes / 60.0
    hands_per_hour = 60.0 / params.minutes_per_hand

    inputs_df = pd.DataFrame(
        [
            ("Bet per hand ($)", params.bet),
            ("Starting bankroll ($)", params.bankroll),
            ("Minutes per hand", params.minutes_per_hand),
            ("Hands per hour", hands_per_hour),
            ("Simulations", params.sims),
            ("Max hands per simulation", params.max_hands),
            ("Random seed", params.seed),
            ("Rule set", "6-deck, H17, DAS, BJ 3:2, basic strategy"),
            ("Generated (UTC)", _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")),
        ],
        columns=["Parameter", "Value"],
    )

    pmf_df = pd.DataFrame(
        {
            "Outcome (units)": OUTCOMES,
            "Meaning": OUTCOME_LABELS,
            "Probability": PROBS,
        }
    )
    pmf_summary = pd.DataFrame(
        [
            ("Sum of probabilities", float(PROBS.sum())),
            ("E[X] (units)", ev),
            ("SD[X] (units)", sd),
            ("House edge", -ev),
        ],
        columns=["Metric", "Value"],
    )

    def stats_block(arr: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
        }

    s_hands = stats_block(ttl_hands)
    s_min = stats_block(ttl_minutes)
    s_hr = stats_block(ttl_hours)
    summary_rows = []
    for key in ("mean", "p10", "p25", "median", "p75", "p90"):
        summary_rows.append(
            (f"TTL {key} (hands)", s_hands[key])
        )
        summary_rows.append(
            (f"TTL {key} (minutes)", s_min[key])
        )
        summary_rows.append(
            (f"TTL {key} (hours)", s_hr[key])
        )
    summary_rows.append(("Ruin rate", float(result.ruined.mean())))
    survivors = result.ending_bankroll[~result.ruined]
    summary_rows.append(
        (
            "Mean ending bankroll (survivors)",
            float(np.mean(survivors)) if survivors.size else float("nan"),
        )
    )
    summary_rows.append(
        (
            "EV $ loss per hour",
            -ev * params.bet * hands_per_hour,
        )
    )
    for hours in (1, 2, 4, 8):
        hand_target = int(round(hours * hands_per_hour))
        ruined_by = float(
            np.mean(result.ruined & (ttl_hands <= hand_target))
        )
        summary_rows.append((f"P(ruin) by {hours}h", ruined_by))
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

    max_minutes = float(ttl_minutes.max())
    n_bins = 60
    bin_edges = np.linspace(0, max(max_minutes, params.minutes_per_hand), n_bins + 1)
    counts, _ = np.histogram(ttl_minutes, bins=bin_edges)
    ttl_dist_df = pd.DataFrame(
        {
            "bin_low_minutes": bin_edges[:-1],
            "bin_high_minutes": bin_edges[1:],
            "count": counts,
            "probability": counts / params.sims,
        }
    )

    survival_df = pd.DataFrame(
        {
            "hand_index": result.survival_hands,
            "minutes": result.survival_hands * params.minutes_per_hand,
            "P(bankroll >= bet)": result.survival_prob,
        }
    )

    n_samples = result.sample_paths.shape[0]
    trajectory_columns = {"hand_index": np.arange(result.sample_paths.shape[1])}
    for i in range(n_samples):
        trajectory_columns[f"sim_{i+1}"] = result.sample_paths[i]
    trajectories_df = pd.DataFrame(trajectory_columns)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        inputs_df.to_excel(writer, sheet_name="Inputs", index=False)
        pmf_df.to_excel(writer, sheet_name="PMF", index=False, startrow=0)
        pmf_summary.to_excel(
            writer, sheet_name="PMF", index=False, startrow=len(pmf_df) + 2
        )
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        ttl_dist_df.to_excel(writer, sheet_name="TTL_Distribution", index=False)
        survival_df.to_excel(writer, sheet_name="Survival_Curve", index=False)
        trajectories_df.to_excel(
            writer, sheet_name="Sample_Trajectories", index=False
        )


def main() -> None:
    ev, sd = validate_pmf()
    params = collect_inputs()
    print(f"\nRunning {params.sims:,} simulations (chunked "
          f"{params.sims_per_batch:,}/batch) ...")
    result = simulate(params, ev, sd)
    print_terminal_report(params, ev, sd, result)
    out_path = "bankroll_simulation.xlsx"
    write_excel(params, ev, sd, result, out_path)
    print(f"\nExcel workbook written to: {out_path}")


if __name__ == "__main__":
    main()
