import os
import tempfile
import unittest

import numpy as np
from openpyxl import load_workbook

from bankroll_sim import (
    SimParams,
    basic_strategy,
    dealer_should_hit,
    hand_value,
    is_blackjack,
    play_round,
    simulate,
    write_excel,
    HandState,
)

UPCARDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 1]  # 1 = ace

# Reference chart: 4-8 deck, H17, DAS, no surrender. Columns follow UPCARDS.
# H=hit, S=stand, D=double else hit, Ds=double else stand, P=split.
HARD_CHART = {
    5: "H H H H H H H H H H",
    6: "H H H H H H H H H H",
    7: "H H H H H H H H H H",
    8: "H H H H H H H H H H",
    9: "H D D D D H H H H H",
    10: "D D D D D D D D H H",
    11: "D D D D D D D D D D",
    12: "H H S S S H H H H H",
    13: "S S S S S H H H H H",
    14: "S S S S S H H H H H",
    15: "S S S S S H H H H H",
    16: "S S S S S H H H H H",
    17: "S S S S S S S S S S",
    18: "S S S S S S S S S S",
    19: "S S S S S S S S S S",
}
SOFT_CHART = {
    13: "H H H D D H H H H H",
    14: "H H H D D H H H H H",
    15: "H H D D D H H H H H",
    16: "H H D D D H H H H H",
    17: "H D D D D H H H H H",
    18: "Ds Ds Ds Ds Ds S S H H H",
    19: "S S S S Ds S S S S S",
    20: "S S S S S S S S S S",
}
PAIR_CHART = {
    1: "P P P P P P P P P P",
    2: "P P P P P P H H H H",
    3: "P P P P P P H H H H",
    4: "H H H P P H H H H H",
    5: "D D D D D D D D H H",
    6: "P P P P P H H H H H",
    7: "P P P P P P H H H H",
    8: "P P P P P P P P P P",
    9: "P P P P P S P P S S",
    10: "S S S S S S S S S S",
}


def _expected(code: str, allow_double: bool) -> str:
    if code == "D":
        return "double" if allow_double else "hit"
    if code == "Ds":
        return "double" if allow_double else "stand"
    return {"H": "hit", "S": "stand", "P": "split"}[code]


class FixedShoe:
    def __init__(self, cards):
        self.cards = list(cards)

    def ensure_cards(self):
        return None

    def draw(self):
        return self.cards.pop(0)


class BlackjackEngineTests(unittest.TestCase):
    def test_hand_totals_and_blackjack(self):
        self.assertEqual(hand_value([1, 7]), (18, True))
        self.assertEqual(hand_value([1, 7, 9]), (17, False))
        self.assertEqual(hand_value([10, 8, 5]), (23, False))
        self.assertTrue(is_blackjack([1, 10]))
        self.assertFalse(is_blackjack([1, 10, 10]))

    def test_dealer_hits_soft_17(self):
        self.assertTrue(dealer_should_hit([1, 6]))
        self.assertFalse(dealer_should_hit([10, 7]))
        self.assertFalse(dealer_should_hit([1, 7]))

    def test_blackjack_pays_three_to_two(self):
        result = play_round(FixedShoe([1, 9, 10, 7]), bankroll_units=10)
        self.assertEqual(result.net_units, 1.5)
        self.assertEqual(result.blackjack, 1)

    def test_normal_push(self):
        result = play_round(FixedShoe([10, 10, 7, 7]), bankroll_units=10)
        self.assertEqual(result.net_units, 0.0)
        self.assertEqual(result.pushes, 1)

    def test_double_down_win(self):
        result = play_round(FixedShoe([5, 6, 6, 10, 10, 10]), bankroll_units=10)
        self.assertEqual(result.net_units, 2.0)
        self.assertEqual(result.exposure_units, 2.0)
        self.assertEqual(result.doubles, 1)

    def test_double_after_split_is_allowed_when_funded(self):
        result = play_round(FixedShoe([8, 6, 8, 10, 3, 3, 10, 10, 10]), bankroll_units=10)
        self.assertEqual(result.splits, 1)
        self.assertEqual(result.doubles, 2)
        self.assertEqual(result.das, 2)
        self.assertEqual(result.exposure_units, 4.0)
        self.assertEqual(result.net_units, 4.0)

    def test_split_aces_receive_one_card_only(self):
        # A,A vs 6: first ace draws a 5 (soft 16 would double vs 6 if played).
        shoe = FixedShoe([1, 6, 1, 10, 5, 7, 10, 9])
        result = play_round(shoe, bankroll_units=10)
        self.assertEqual(result.splits, 1)
        self.assertEqual(result.doubles, 0)
        self.assertEqual(result.exposure_units, 2.0)
        # Dealer 16 draws 10 and busts; both one-card ace hands win.
        self.assertEqual(result.net_units, 2.0)
        self.assertEqual(shoe.cards, [9])

    def test_unfunded_split_falls_back_to_playing_hand(self):
        result = play_round(FixedShoe([8, 6, 8, 10, 10]), bankroll_units=1)
        self.assertEqual(result.splits, 0)
        self.assertEqual(result.exposure_units, 1.0)
        self.assertEqual(result.net_units, 1.0)

    def test_unfunded_double_falls_back_to_hit_or_stand(self):
        result = play_round(FixedShoe([5, 10, 6, 7, 2, 10]), bankroll_units=1)
        self.assertEqual(result.doubles, 0)
        self.assertEqual(result.exposure_units, 1.0)
        self.assertEqual(result.net_units, -1.0)

    def test_pair_strategy_respects_allow_split(self):
        hand = HandState([8, 8])
        self.assertEqual(basic_strategy(hand, dealer_up=6, allow_double=True, allow_split=True), "split")
        self.assertNotEqual(basic_strategy(hand, dealer_up=6, allow_double=True, allow_split=False), "split")

    def test_resplit_capped_at_four_hands(self):
        shoe = FixedShoe([8, 6, 8, 10] + [8] * 6 + [10] * 5)
        result = play_round(shoe, bankroll_units=20)
        self.assertEqual(result.splits, 3)
        self.assertEqual(result.exposure_units, 4.0)
        # Four 8,8 hands stand vs 6; dealer 16 draws 10 and busts.
        self.assertEqual(result.net_units, 4.0)

    def test_resplit_gated_by_bankroll(self):
        shoe = FixedShoe([8, 6, 8, 10] + [8] * 6 + [10] * 5)
        result = play_round(shoe, bankroll_units=2)
        self.assertEqual(result.splits, 1)
        self.assertEqual(result.exposure_units, 2.0)

    def test_dealer_blackjack_takes_only_original_bet(self):
        result = play_round(FixedShoe([8, 1, 8, 10]), bankroll_units=10)
        self.assertEqual(result.net_units, -1.0)
        self.assertEqual(result.splits, 0)
        self.assertEqual(result.exposure_units, 1.0)

    def test_blackjack_vs_blackjack_pushes(self):
        result = play_round(FixedShoe([1, 1, 10, 10]), bankroll_units=10)
        self.assertEqual(result.net_units, 0.0)
        self.assertEqual(result.pushes, 1)
        self.assertEqual(result.blackjack, 1)

    def test_player_bust_dealer_does_not_draw(self):
        shoe = FixedShoe([10, 10, 6, 10, 7, 5])
        result = play_round(shoe, bankroll_units=10)
        self.assertEqual(result.net_units, -1.0)
        self.assertEqual(result.player_busts, 1)
        self.assertEqual(result.dealer_bust, 0)
        self.assertEqual(shoe.cards, [5])

    def test_three_card_soft_18_stands_vs_small_upcards(self):
        for up in (2, 3, 4, 5, 6, 7, 8):
            self.assertEqual(basic_strategy(HandState([1, 2, 5]), up, False, False), "stand", up)
        for up in (9, 10, 1):
            self.assertEqual(basic_strategy(HandState([1, 2, 5]), up, False, False), "hit", up)

    def test_strategy_grid_matches_reference_chart(self):
        for allow_double in (True, False):
            for total, row in HARD_CHART.items():
                cards = [2, total - 2] if total <= 11 else [10, total - 10]
                if total == 20:
                    cards = [10, 6, 4]
                for up, code in zip(UPCARDS, row.split()):
                    got = basic_strategy(HandState(cards), up, allow_double, False)
                    self.assertEqual(got, _expected(code, allow_double), (cards, up, allow_double))
            for total, row in SOFT_CHART.items():
                cards = [1, total - 11]
                for up, code in zip(UPCARDS, row.split()):
                    got = basic_strategy(HandState(cards), up, allow_double, False)
                    self.assertEqual(got, _expected(code, allow_double), (cards, up, allow_double))
            for pair, row in PAIR_CHART.items():
                for up, code in zip(UPCARDS, row.split()):
                    got = basic_strategy(HandState([pair, pair]), up, allow_double, True)
                    self.assertEqual(got, _expected(code, allow_double), ([pair, pair], up, allow_double))

    def test_seeded_simulation_is_deterministic(self):
        params = SimParams(
            bet=25,
            bankroll=500,
            minutes_per_hand=0.5,
            sims=25,
            max_hands=50,
            seed=12345,
            n_trajectories=5,
        )
        first = simulate(params)
        second = simulate(params)
        np.testing.assert_array_equal(first.ttl_hands, second.ttl_hands)
        np.testing.assert_allclose(first.ending_bankroll, second.ending_bankroll)
        self.assertAlmostEqual(first.ev_units, second.ev_units)

    def test_simulation_ev_is_plausible_for_basic_strategy(self):
        params = SimParams(
            bet=25,
            bankroll=10000,
            minutes_per_hand=0.5,
            sims=100,
            max_hands=100,
            seed=20260424,
            n_trajectories=5,
        )
        result = simulate(params)
        self.assertGreater(result.hands_played_total, 0)
        self.assertGreater(result.sd_units, 0.8)
        self.assertLess(result.ev_units, 0.20)
        self.assertGreater(result.ev_units, -0.30)


    def test_bankroll_below_bet_is_immediate_ruin_with_clean_paths(self):
        params = SimParams(bet=25, bankroll=10, minutes_per_hand=0.5, sims=5,
                           max_hands=20, seed=1, n_trajectories=5)
        result = simulate(params)
        self.assertTrue(result.ruined.all())
        self.assertTrue((result.ttl_hands == 0).all())
        np.testing.assert_array_equal(result.sample_paths, np.full((5, 21), 10.0))

    def test_bankroll_equal_to_bet_plays_one_hand(self):
        params = SimParams(bet=25, bankroll=25, minutes_per_hand=0.5, sims=50,
                           max_hands=20, seed=3, n_trajectories=5)
        result = simulate(params)
        self.assertTrue((result.ttl_hands >= 1).all())

    def test_fractional_bet_is_scale_invariant(self):
        # 0.3 is not exact in binary; ruin must match the integer-scaled run.
        base = dict(minutes_per_hand=0.5, sims=200, max_hands=60, seed=7, n_trajectories=5)
        small = simulate(SimParams(bet=0.3, bankroll=0.9, **base))
        large = simulate(SimParams(bet=3.0, bankroll=9.0, **base))
        np.testing.assert_array_equal(small.ruined, large.ruined)
        np.testing.assert_array_equal(small.ttl_hands, large.ttl_hands)
        self.assertEqual(small.hands_played_total, large.hands_played_total)

    def test_survival_curve_is_non_increasing(self):
        params = SimParams(bet=25, bankroll=200, minutes_per_hand=0.5, sims=200,
                           max_hands=200, seed=11, n_trajectories=5)
        result = simulate(params)
        self.assertTrue(np.all(np.diff(result.survival_prob) <= 0))

    def test_write_excel_produces_all_sheets(self):
        params = SimParams(bet=25, bankroll=200, minutes_per_hand=0.5, sims=20,
                           max_hands=40, seed=5, n_trajectories=5)
        result = simulate(params)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.xlsx")
            write_excel(params, result, path)
            names = load_workbook(path, read_only=True).sheetnames
        self.assertEqual(
            set(names),
            {"Inputs", "Engine_Stats", "Summary", "TTL_Distribution", "Survival_Curve", "Sample_Trajectories"},
        )

    @unittest.skipUnless(os.environ.get("BLACKJACK_SLOW_TESTS"), "set BLACKJACK_SLOW_TESTS=1")
    def test_unconstrained_ev_matches_published_edge(self):
        # ~1M rounds, bankroll never binds. SE ~0.12% of a unit.
        params = SimParams(bet=1, bankroll=1e9, minutes_per_hand=0.5, sims=1000,
                           max_hands=1000, seed=20260424, n_trajectories=1)
        result = simulate(params)
        self.assertGreater(result.ev_units, -0.012)
        self.assertLess(result.ev_units, -0.002)


if __name__ == "__main__":
    unittest.main()
