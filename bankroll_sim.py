"""Card-level blackjack bankroll Monte Carlo simulator.

Rules modeled: 6-deck shoe, H17, DAS, blackjack pays 3:2, no surrender,
no insurance, flat betting, common Vegas split rules.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

RULE_SET = (
    "6-deck, H17, DAS, BJ 3:2, no surrender/insurance, "
    "split to 4 hands, split aces one card only"
)
DECKS = 6
RESHUFFLE_AT_CARDS = 78
MAX_SPLIT_HANDS = 4
MAX_ROUND_NET_UNITS = MAX_SPLIT_HANDS * 2.0
N_CHECKPOINTS = 50
DURATION_HOURS = (1, 2, 4, 8, 12, 24, 48)
EV_LABEL = "Realized EV per round (bankroll-constrained)"


@dataclass
class SimParams:
    bet: float
    bankroll: float
    minutes_per_hand: float
    sims: int
    max_hands: int
    seed: int
    n_trajectories: int = 50


@dataclass
class SimResult:
    ttl_hands: np.ndarray
    ruined: np.ndarray
    ending_bankroll: np.ndarray
    survival_hands: np.ndarray
    survival_prob: np.ndarray
    sample_paths: np.ndarray
    band_p10: np.ndarray
    band_p50: np.ndarray
    band_p90: np.ndarray
    ev_units: float
    ev_se_units: float
    sd_units: float
    avg_exposure_units: float
    max_exposure_units: float
    split_rate: float
    double_rate: float
    das_rate: float
    split_action_rate: float
    double_action_rate: float
    das_action_rate: float
    blackjack_rate: float
    push_rate: float
    player_bust_rate: float
    dealer_bust_rate: float
    multi_bet_rate: float
    hands_played_total: int
    splits_total: int
    doubles_total: int
    das_total: int
    blackjacks_total: int
    pushes_total: int
    player_busts_total: int
    dealer_busts_total: int
    multi_bet_rounds_total: int
    split_rounds_total: int
    double_rounds_total: int
    das_rounds_total: int
    push_rounds_total: int
    player_bust_rounds_total: int


@dataclass
class HandState:
    cards: list[int]
    wager_units: float = 1.0
    from_split: bool = False
    split_aces: bool = False
    doubled: bool = False
    stood: bool = False
    busted: bool = False


@dataclass
class RoundResult:
    net_units: float
    exposure_units: float
    splits: int
    doubles: int
    das: int
    blackjack: int
    pushes: int
    player_busts: int
    dealer_bust: int


class Shoe:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.cards: list[int] = []
        self.shuffle()

    def shuffle(self) -> None:
        one_deck = [1, 2, 3, 4, 5, 6, 7, 8, 9] * 4 + [10] * 16
        cards = one_deck * DECKS
        self.rng.shuffle(cards)
        self.cards = list(cards)

    def ensure_cards(self) -> None:
        if len(self.cards) < RESHUFFLE_AT_CARDS:
            self.shuffle()

    def draw(self) -> int:
        if not self.cards:
            self.shuffle()
        return self.cards.pop()


def hand_value(cards: list[int]) -> tuple[int, bool]:
    """Best total and whether an ace is being counted as 11 (soft)."""
    total = sum(cards)
    if 1 in cards and total + 10 <= 21:
        return total + 10, True
    return total, False


def is_blackjack(cards: list[int]) -> bool:
    return len(cards) == 2 and hand_value(cards)[0] == 21


def can_split(cards: list[int]) -> bool:
    return len(cards) == 2 and cards[0] == cards[1]


def dealer_should_hit(cards: list[int]) -> bool:
    total, soft = hand_value(cards)
    return total < 17 or (total == 17 and soft)


def _hard_strategy(total: int, dealer_up: int, allow_double: bool) -> str:
    if total >= 17:
        return "stand"
    if 13 <= total <= 16:
        return "stand" if 2 <= dealer_up <= 6 else "hit"
    if total == 12:
        return "stand" if 4 <= dealer_up <= 6 else "hit"
    if total == 11:
        return "double" if allow_double else "hit"
    if total == 10:
        return "double" if allow_double and 2 <= dealer_up <= 9 else "hit"
    if total == 9:
        return "double" if allow_double and 3 <= dealer_up <= 6 else "hit"
    return "hit"


def _soft_strategy(total: int, dealer_up: int, allow_double: bool) -> str:
    if total >= 20:
        return "stand"
    if total == 19:
        return "double" if allow_double and dealer_up == 6 else "stand"
    if total == 18:
        if allow_double and 2 <= dealer_up <= 6:
            return "double"
        return "stand" if 2 <= dealer_up <= 8 else "hit"
    if total == 17:
        return "double" if allow_double and 3 <= dealer_up <= 6 else "hit"
    if total in (15, 16):
        return "double" if allow_double and 4 <= dealer_up <= 6 else "hit"
    if total in (13, 14):
        return "double" if allow_double and 5 <= dealer_up <= 6 else "hit"
    return "hit"


def basic_strategy(hand: HandState, dealer_up: int, allow_double: bool, allow_split: bool) -> str:
    total, soft = hand_value(hand.cards)
    if allow_split and can_split(hand.cards):
        pair = hand.cards[0]
        if pair == 1:
            return "split"
        if pair == 10:
            return "stand"
        if pair == 9:
            return "split" if dealer_up in (2, 3, 4, 5, 6, 8, 9) else "stand"
        if pair == 8:
            return "split"
        if pair == 7:
            return "split" if 2 <= dealer_up <= 7 else "hit"
        if pair == 6:
            return "split" if 2 <= dealer_up <= 6 else "hit"
        if pair == 5:
            return _hard_strategy(10, dealer_up, allow_double)
        if pair == 4:
            return "split" if dealer_up in (5, 6) else "hit"
        if pair in (2, 3):
            return "split" if 2 <= dealer_up <= 7 else "hit"
    if soft:
        return _soft_strategy(total, dealer_up, allow_double)
    return _hard_strategy(total, dealer_up, allow_double)


def _fallback_after_declined_split(hand: HandState, dealer_up: int, allow_double: bool) -> str:
    total, soft = hand_value(hand.cards)
    if soft:
        return _soft_strategy(total, dealer_up, allow_double)
    return _hard_strategy(total, dealer_up, allow_double)


def _settle_hand(hand: HandState, dealer_cards: list[int]) -> tuple[float, int, int]:
    player_total, _ = hand_value(hand.cards)
    if player_total > 21:
        return -hand.wager_units, 0, 1

    dealer_total, _ = hand_value(dealer_cards)
    if dealer_total > 21:
        return hand.wager_units, 0, 0
    if player_total > dealer_total:
        return hand.wager_units, 0, 0
    if player_total < dealer_total:
        return -hand.wager_units, 0, 0
    return 0.0, 1, 0


def play_round(shoe: Shoe, bankroll_units: float) -> RoundResult:
    shoe.ensure_cards()
    player = [shoe.draw()]
    dealer = [shoe.draw()]
    player.append(shoe.draw())
    dealer.append(shoe.draw())
    dealer_up = dealer[0]

    if is_blackjack(player) or is_blackjack(dealer):
        exposure = 1.0
        if is_blackjack(player) and is_blackjack(dealer):
            return RoundResult(0.0, exposure, 0, 0, 0, 1, 1, 0, 0)
        if is_blackjack(player):
            return RoundResult(1.5, exposure, 0, 0, 0, 1, 0, 0, 0)
        return RoundResult(-1.0, exposure, 0, 0, 0, 0, 0, 0, 0)

    hands = [HandState(player)]
    committed = 1.0
    max_exposure = 1.0
    splits = 0
    doubles = 0
    das = 0
    player_busts = 0
    i = 0

    while i < len(hands):
        hand = hands[i]
        if hand.split_aces:
            total, _ = hand_value(hand.cards)
            hand.busted = total > 21
            player_busts += int(hand.busted)
            hand.stood = True
            i += 1
            continue

        while not hand.stood and not hand.busted:
            total, _ = hand_value(hand.cards)
            if total > 21:
                hand.busted = True
                player_busts += 1
                break

            allow_double = len(hand.cards) == 2
            allow_split = (
                len(hands) < MAX_SPLIT_HANDS
                and can_split(hand.cards)
                and not (hand.cards[0] == 1 and hand.from_split)
            )
            decision = basic_strategy(hand, dealer_up, allow_double, allow_split)

            if decision == "split":
                can_fund = committed + 1.0 <= bankroll_units
                if allow_split and can_fund:
                    first, second = hand.cards
                    split_aces = first == 1
                    hand.cards = [first, shoe.draw()]
                    hand.from_split = True
                    hand.split_aces = split_aces
                    hands.insert(
                        i + 1,
                        HandState([second, shoe.draw()], from_split=True, split_aces=split_aces),
                    )
                    committed += 1.0
                    max_exposure = max(max_exposure, committed)
                    splits += 1
                    if split_aces:
                        # Split aces receive exactly one card each and stand.
                        hand.stood = True
                        break
                    continue
                decision = _fallback_after_declined_split(hand, dealer_up, allow_double)

            if decision == "double":
                can_fund = committed + hand.wager_units <= bankroll_units
                if allow_double and can_fund:
                    committed += hand.wager_units
                    max_exposure = max(max_exposure, committed)
                    hand.wager_units *= 2.0
                    hand.doubled = True
                    doubles += 1
                    das += int(hand.from_split)
                    hand.cards.append(shoe.draw())
                    total, _ = hand_value(hand.cards)
                    hand.busted = total > 21
                    player_busts += int(hand.busted)
                    hand.stood = True
                    break
                decision = _fallback_after_declined_split(hand, dealer_up, False)

            if decision == "stand":
                hand.stood = True
            elif decision == "hit":
                hand.cards.append(shoe.draw())
            else:
                raise ValueError(f"Unknown strategy decision: {decision}")
        i += 1

    live_hands = [hand for hand in hands if hand_value(hand.cards)[0] <= 21]
    dealer_bust = 0
    if live_hands:
        while dealer_should_hit(dealer):
            dealer.append(shoe.draw())
        dealer_bust = int(hand_value(dealer)[0] > 21)

    net = 0.0
    pushes = 0
    for hand in hands:
        delta, push, bust = _settle_hand(hand, dealer)
        net += delta
        pushes += push

    return RoundResult(
        net_units=net,
        exposure_units=max_exposure,
        splits=splits,
        doubles=doubles,
        das=das,
        blackjack=0,
        pushes=pushes,
        player_busts=player_busts,
        dealer_bust=dealer_bust,
    )


def simulate(params: SimParams) -> SimResult:
    rng = np.random.default_rng(params.seed)

    ttl_hands = np.empty(params.sims, dtype=np.int64)
    ruined = np.empty(params.sims, dtype=bool)
    ending = np.empty(params.sims, dtype=np.float64)

    # Checkpoints include hand 0 (P = 1) and are de-duplicated for short sessions.
    survival_hands = np.unique(
        np.linspace(0, params.max_hands, N_CHECKPOINTS + 1).round().astype(np.int64)
    )
    survival_hits = np.zeros(survival_hands.size, dtype=np.int64)
    checkpoint_bankroll = np.empty((params.sims, survival_hands.size), dtype=np.float64)

    n_samples = min(params.n_trajectories, params.sims)
    sample_paths = np.empty((n_samples, params.max_hands + 1), dtype=np.float64)
    path = np.empty(params.max_hands + 1, dtype=np.float64)

    # Per-session sums feed a cluster-robust SE for EV: rounds within one
    # session share a shoe and a bankroll, so they are not independent.
    path_net = np.zeros(params.sims, dtype=np.float64)
    path_rounds = np.zeros(params.sims, dtype=np.int64)
    sum_sq_units = 0.0
    sum_exposure = 0.0
    max_exposure_seen = 0.0

    hands_played_total = 0
    splits_total = 0
    doubles_total = 0
    das_total = 0
    blackjacks_total = 0
    pushes_total = 0
    player_busts_total = 0
    dealer_busts_total = 0
    multi_bet_rounds_total = 0
    split_rounds_total = 0
    double_rounds_total = 0
    das_rounds_total = 0
    push_rounds_total = 0
    player_bust_rounds_total = 0

    # Bankroll is tracked in base-bet units. Round outcomes are exact
    # multiples of 0.5 units, so only the starting ratio can carry float
    # noise (e.g. 0.9 / 0.3); rounding it removes false ruin / gating.
    start_units = round(params.bankroll / params.bet, 9)

    for sim_idx in range(params.sims):
        shoe = Shoe(rng)
        bankroll_units = start_units
        bankroll = bankroll_units * params.bet
        path[0] = bankroll
        net_this_path = 0.0

        ttl = params.max_hands
        is_ruined = False

        for hand_idx in range(1, params.max_hands + 1):
            if bankroll_units < 1.0:
                ttl = hand_idx - 1
                is_ruined = True
                path[hand_idx:] = bankroll
                break

            result = play_round(shoe, bankroll_units)
            net = result.net_units
            bankroll_units += net
            bankroll = bankroll_units * params.bet
            path[hand_idx] = bankroll

            net_this_path += net
            sum_sq_units += net * net
            sum_exposure += result.exposure_units
            if result.exposure_units > max_exposure_seen:
                max_exposure_seen = result.exposure_units
            hands_played_total += 1
            splits_total += result.splits
            doubles_total += result.doubles
            das_total += result.das
            blackjacks_total += result.blackjack
            pushes_total += result.pushes
            player_busts_total += result.player_busts
            dealer_busts_total += result.dealer_bust
            multi_bet_rounds_total += int(result.exposure_units > 1.0)
            split_rounds_total += int(result.splits > 0)
            double_rounds_total += int(result.doubles > 0)
            das_rounds_total += int(result.das > 0)
            push_rounds_total += int(result.pushes > 0)
            player_bust_rounds_total += int(result.player_busts > 0)

            if bankroll_units < 1.0:
                ttl = hand_idx
                is_ruined = True
                path[hand_idx + 1 :] = bankroll
                break

        if sim_idx < n_samples:
            sample_paths[sim_idx] = path
        checkpoint_bankroll[sim_idx] = path[survival_hands]
        path_net[sim_idx] = net_this_path
        path_rounds[sim_idx] = ttl

        ttl_hands[sim_idx] = ttl
        ruined[sim_idx] = is_ruined
        ending[sim_idx] = bankroll
        survival_check = ttl if is_ruined else params.max_hands + 1
        survival_hits += (survival_check > survival_hands).astype(np.int64)

    n = hands_played_total
    ev_units = float(path_net.sum() / n) if n else 0.0
    sd_units = float(np.sqrt(max(sum_sq_units / n - ev_units**2, 0.0))) if n else 0.0
    ev_se_units = _clustered_ratio_se(path_net, path_rounds, ev_units)
    denom = float(n) if n else 1.0

    band_q = np.percentile(checkpoint_bankroll, [10, 50, 90], axis=0)

    return SimResult(
        ttl_hands=ttl_hands,
        ruined=ruined,
        ending_bankroll=ending,
        survival_hands=survival_hands,
        survival_prob=survival_hits / params.sims,
        sample_paths=sample_paths,
        band_p10=band_q[0],
        band_p50=band_q[1],
        band_p90=band_q[2],
        ev_units=ev_units,
        ev_se_units=ev_se_units,
        sd_units=sd_units,
        avg_exposure_units=sum_exposure / n if n else 0.0,
        max_exposure_units=max_exposure_seen,
        split_rate=split_rounds_total / denom,
        double_rate=double_rounds_total / denom,
        das_rate=das_rounds_total / denom,
        split_action_rate=splits_total / denom,
        double_action_rate=doubles_total / denom,
        das_action_rate=das_total / denom,
        blackjack_rate=blackjacks_total / denom,
        push_rate=push_rounds_total / denom,
        player_bust_rate=player_bust_rounds_total / denom,
        dealer_bust_rate=dealer_busts_total / denom,
        multi_bet_rate=multi_bet_rounds_total / denom,
        hands_played_total=hands_played_total,
        splits_total=splits_total,
        doubles_total=doubles_total,
        das_total=das_total,
        blackjacks_total=blackjacks_total,
        pushes_total=pushes_total,
        player_busts_total=player_busts_total,
        dealer_busts_total=dealer_busts_total,
        multi_bet_rounds_total=multi_bet_rounds_total,
        split_rounds_total=split_rounds_total,
        double_rounds_total=double_rounds_total,
        das_rounds_total=das_rounds_total,
        push_rounds_total=push_rounds_total,
        player_bust_rounds_total=player_bust_rounds_total,
    )


def _clustered_ratio_se(path_net: np.ndarray, path_rounds: np.ndarray, ev: float) -> float:
    """SE of EV = sum(net) / sum(rounds), clustered by session.

    Var(EV) ~= S/(S-1) * sum_i (net_i - EV * rounds_i)^2 / (sum_i rounds_i)^2
    """
    total_rounds = float(path_rounds.sum())
    s = path_net.size
    if s < 2 or total_rounds == 0:
        return float("nan")
    resid = path_net - ev * path_rounds
    return float(np.sqrt(s / (s - 1) * np.sum(resid**2)) / total_rounds)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, float(centre - half)), min(1.0, float(centre + half))


def ttl_quantile(result: SimResult, q: float) -> int | None:
    """Hands until ruin at quantile q of the time-to-ruin distribution.

    Every survivor is censored at the same point (max_hands), so the
    Kaplan-Meier estimate equals the empirical one. Returns None when fewer
    than q of sessions are ruined within the session ("> max_hands").
    """
    needed = int(np.ceil(q * result.ruined.size))
    ruined_ttl = np.sort(result.ttl_hands[result.ruined])
    if needed == 0 or ruined_ttl.size < needed:
        return None
    return int(ruined_ttl[needed - 1])


def ruin_by_hours(
    params: SimParams, result: SimResult, hours: tuple[float, ...] = DURATION_HOURS
) -> list[dict]:
    """P(ruined by H hours) with 95% Wilson CI, limited to the simulated session.

    Always ends with a row for the full session length.
    """
    hands_per_hour = 60.0 / params.minutes_per_hand
    targets = []
    for h in hours:
        hand_target = int(round(h * hands_per_hour))
        if 0 < hand_target < params.max_hands:
            targets.append((float(h), hand_target))
    targets.append((params.max_hands / hands_per_hour, params.max_hands))

    rows = []
    for h, hand_target in targets:
        k = int(np.sum(result.ruined & (result.ttl_hands <= hand_target)))
        lo, hi = wilson_interval(k, params.sims)
        rows.append({"hours": h, "hands": hand_target, "p_ruin": k / params.sims, "ci_low": lo, "ci_high": hi})
    return rows


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
    print(f" Rules: {RULE_SET}")
    print("=" * 60)
    bet = prompt_float("Dollars per hand", 25)
    bankroll = prompt_float("Starting bankroll ($)", 500)
    minutes_per_hand = prompt_float("Minutes per hand", 0.5)
    session_hours = prompt_float("Session length (hours)", 4.0)
    sims = prompt_int("Number of simulations", 1000)
    seed = prompt_int("Random seed", 20260424)
    if bet <= 0 or bankroll <= 0 or minutes_per_hand <= 0 or session_hours <= 0:
        raise ValueError("bet, bankroll, minutes_per_hand, and session_hours must be positive")
    if bankroll < bet:
        raise ValueError("bankroll must be at least one bet")
    max_hands = max(1, int(round((session_hours * 60.0) / minutes_per_hand)))
    return SimParams(
        bet=bet,
        bankroll=bankroll,
        minutes_per_hand=minutes_per_hand,
        sims=sims,
        max_hands=max_hands,
        seed=seed,
    )


TTL_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def ttl_rows(params: SimParams, result: SimResult) -> list[tuple[str, int | None]]:
    """Time-to-ruin quantiles in hands; None means beyond the session."""
    return [(f"p{int(q * 100)}", ttl_quantile(result, q)) for q in TTL_QUANTILES]


def _fmt_hands(h: int | None, params: SimParams) -> str:
    if h is None:
        return f"> {params.max_hands:,} hands (session end)"
    return f"{h:,} hands ({h * params.minutes_per_hand / 60.0:.2f} h)"


def print_terminal_report(params: SimParams, result: SimResult) -> None:
    ruin_k = int(result.ruined.sum())
    ruin_rate = ruin_k / params.sims
    ruin_lo, ruin_hi = wilson_interval(ruin_k, params.sims)

    dollar_loss_per_hand = -result.ev_units * params.bet
    hands_per_hour = 60.0 / params.minutes_per_hand
    ev_dollar_loss_per_hour = dollar_loss_per_hand * hands_per_hour
    ev_ci = 1.96 * result.ev_se_units

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
    print(f"  Rule set            : {RULE_SET}")

    print()
    print("-" * 60)
    print(" Engine statistics (per initial round, base-bet units)")
    print("-" * 60)
    print(f"  Rounds played       : {result.hands_played_total:,}")
    print(f"  {EV_LABEL}:")
    print(f"                        {result.ev_units:+.5f} units  (95% CI +/- {ev_ci:.5f})")
    print(f"  SD per round        : {result.sd_units:.5f} units")
    print(f"  Avg exposure        : {result.avg_exposure_units:.3f} units")
    print(f"  Max exposure        : {result.max_exposure_units:.1f} units")
    print(f"  Rounds with split   : {result.split_rate * 100:.2f}%")
    print(f"  Rounds with double  : {result.double_rate * 100:.2f}%")
    print(f"  Rounds with DAS     : {result.das_rate * 100:.2f}%")
    print(f"  Split actions/round : {result.split_action_rate * 100:.2f}%")
    print(f"  Double actions/round: {result.double_action_rate * 100:.2f}%")
    print(f"  Multi-bet rounds    : {result.multi_bet_rate * 100:.2f}%")
    print(f"  EV $ loss per hand  : ${dollar_loss_per_hand:,.4f}")
    print(f"  EV $ loss per hour  : ${ev_dollar_loss_per_hour:,.2f}")

    print()
    print("-" * 60)
    print(" Time to ruin (survivors censored at session end)")
    print("-" * 60)
    for label, h in ttl_rows(params, result):
        print(f"  {label:<6}: {_fmt_hands(h, params)}")
    mean_played = float(result.ttl_hands.mean())
    print(f"  mean hands played (capped at {params.max_hands:,}): {mean_played:,.1f}")

    print()
    print("-" * 60)
    print(" Ruin probability by play duration (95% Wilson CI)")
    print("-" * 60)
    for row in ruin_by_hours(params, result):
        print(
            f"  by {row['hours']:>5.2f}h ({row['hands']:>6,} hands) : "
            f"{row['p_ruin'] * 100:5.2f}%  [{row['ci_low'] * 100:5.2f}%, {row['ci_high'] * 100:5.2f}%]"
        )
    print(
        f"  overall ruin rate (<= {params.max_hands:,} hands): {ruin_rate * 100:.2f}% "
        f"[{ruin_lo * 100:.2f}%, {ruin_hi * 100:.2f}%]"
    )

    print(f"  mean ending bankroll (all sims, incl. ruined): ${float(np.mean(result.ending_bankroll)):,.2f}")
    survivors = result.ending_bankroll[~result.ruined]
    if survivors.size:
        print(f"  mean ending bankroll among survivors: ${float(np.mean(survivors)):,.2f}")


def write_excel(params: SimParams, result: SimResult, path: str) -> None:
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
            ("Rule set", RULE_SET),
            ("Generated (UTC)", _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")),
        ],
        columns=["Parameter", "Value"],
    )

    engine_df = pd.DataFrame(
        [
            ("Rounds played", result.hands_played_total),
            (f"{EV_LABEL} (units)", result.ev_units),
            ("EV standard error (units, clustered by session)", result.ev_se_units),
            ("SD per round (units)", result.sd_units),
            ("Average exposure (units)", result.avg_exposure_units),
            ("Max exposure (units)", result.max_exposure_units),
            ("Rounds with split", result.split_rate),
            ("Rounds with double", result.double_rate),
            ("Rounds with DAS", result.das_rate),
            ("Split actions per round", result.split_action_rate),
            ("Double actions per round", result.double_action_rate),
            ("DAS actions per round", result.das_action_rate),
            ("Blackjack rate", result.blackjack_rate),
            ("Push rate", result.push_rate),
            ("Player bust rate", result.player_bust_rate),
            ("Dealer bust rate (dealer draws only if a player hand is live)", result.dealer_bust_rate),
            ("Multi-bet round rate", result.multi_bet_rate),
            ("Total splits", result.splits_total),
            ("Total doubles", result.doubles_total),
            ("Total DAS", result.das_total),
            ("Total rounds with split", result.split_rounds_total),
            ("Total rounds with double", result.double_rounds_total),
            ("Total rounds with DAS", result.das_rounds_total),
        ],
        columns=["Metric", "Value"],
    )

    summary_rows: list[tuple[str, object]] = []
    for label, h in ttl_rows(params, result):
        summary_rows.append((f"Time to ruin {label} (hands)", h if h is not None else f"> {params.max_hands}"))
        summary_rows.append(
            (f"Time to ruin {label} (hours)", h * params.minutes_per_hand / 60.0 if h is not None else "beyond session")
        )
    summary_rows.append((f"Mean hands played (capped at {params.max_hands})", float(result.ttl_hands.mean())))
    ruin_k = int(result.ruined.sum())
    ruin_lo, ruin_hi = wilson_interval(ruin_k, params.sims)
    summary_rows.append(("Ruin rate", ruin_k / params.sims))
    summary_rows.append(("Ruin rate 95% CI low", ruin_lo))
    summary_rows.append(("Ruin rate 95% CI high", ruin_hi))
    summary_rows.append(("Mean ending bankroll (all simulations, incl. ruined)", float(np.mean(result.ending_bankroll))))
    summary_rows.append(
        (
            "Mean net result (all simulations)",
            float(np.mean(result.ending_bankroll)) - params.bankroll,
        )
    )
    survivors = result.ending_bankroll[~result.ruined]
    summary_rows.append(
        (
            "Mean ending bankroll (survivors)",
            float(np.mean(survivors)) if survivors.size else float("nan"),
        )
    )
    summary_rows.append(("EV $ loss per hour", -result.ev_units * params.bet * hands_per_hour))
    for row in ruin_by_hours(params, result):
        summary_rows.append((f"P(ruin) by {row['hours']:.2f}h ({row['hands']} hands)", row["p_ruin"]))
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

    # Histogram of time to ruin for ruined sessions; survivors reported separately.
    ruined_minutes = result.ttl_hands[result.ruined] * params.minutes_per_hand
    n_bins = 60
    session_minutes = params.max_hands * params.minutes_per_hand
    bin_edges = np.linspace(0, session_minutes, n_bins + 1)
    counts, _ = np.histogram(ruined_minutes, bins=bin_edges)
    n_survivors = int((~result.ruined).sum())
    ttl_dist_df = pd.DataFrame(
        {
            "bin_low_minutes": np.append(bin_edges[:-1], session_minutes),
            "bin_high_minutes": np.append(bin_edges[1:], np.inf),
            "count": np.append(counts, n_survivors),
            "probability": np.append(counts, n_survivors) / params.sims,
            "status": ["ruined"] * n_bins + ["survived session"],
        }
    )

    survival_df = pd.DataFrame(
        {
            "hand_index": result.survival_hands,
            "minutes": result.survival_hands * params.minutes_per_hand,
            "P(bankroll >= bet)": result.survival_prob,
            "bankroll_p10": result.band_p10,
            "bankroll_median": result.band_p50,
            "bankroll_p90": result.band_p90,
        }
    )

    trajectory_columns = {"hand_index": np.arange(result.sample_paths.shape[1])}
    for i in range(result.sample_paths.shape[0]):
        trajectory_columns[f"sim_{i + 1}"] = result.sample_paths[i]
    trajectories_df = pd.DataFrame(trajectory_columns)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        inputs_df.to_excel(writer, sheet_name="Inputs", index=False)
        engine_df.to_excel(writer, sheet_name="Engine_Stats", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        ttl_dist_df.to_excel(writer, sheet_name="TTL_Distribution", index=False)
        survival_df.to_excel(writer, sheet_name="Survival_Curve", index=False)
        trajectories_df.to_excel(writer, sheet_name="Sample_Trajectories", index=False)


def main() -> None:
    params = collect_inputs()
    print(f"\nRunning {params.sims:,} simulations with card-level engine ...")
    result = simulate(params)
    print_terminal_report(params, result)
    out_path = "bankroll_simulation.xlsx"
    write_excel(params, result, out_path)
    print(f"\nExcel workbook written to: {out_path}")


if __name__ == "__main__":
    main()
