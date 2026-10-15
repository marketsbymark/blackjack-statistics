import unittest
import contextlib
import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bankroll_sim import (
    SimParams,
    basic_strategy,
    dealer_should_hit,
    hand_value,
    is_blackjack,
    play_round,
    simulate,
    HandState,
    RoundResult,
    Shoe,
    write_excel,
    print_terminal_report,
)


class FixedShoe:
    def __init__(self, cards):
        self.cards = list(cards)

    def ensure_cards(self):
        return None

    def draw(self):
        return self.cards.pop(0)


class BlackjackEngineTests(unittest.TestCase):
    def params(self, **overrides):
        values = dict(bet=25, bankroll=500, minutes_per_hand=0.5,
                      sims=4, max_hands=10, seed=42, n_trajectories=4)
        values.update(overrides)
        return SimParams(**values)

    def test_invalid_parameters_fail_before_simulation(self):
        for overrides in (dict(bet=24.99), dict(bet=0), dict(bet=float("nan")),
                          dict(bankroll=-1), dict(bankroll=float("inf")),
                          dict(bankroll=24.999), dict(bet=25.01),
                          dict(minutes_per_hand=0), dict(sims=0),
                          dict(max_hands=0), dict(sims=1.5), dict(seed=-1),
                          dict(n_trajectories=-1)):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                simulate(self.params(**overrides))

    def test_underfunded_round_never_deals(self):
        shoe = FixedShoe([])
        for units in (0, 0.999999, -1, float("nan")):
            with self.subTest(units=units), self.assertRaises(ValueError):
                play_round(shoe, units)

    def test_initially_unplayable_balances_are_frozen(self):
        for bankroll in (0, 0.01, 12.50, 24.99):
            with self.subTest(bankroll=bankroll), patch("bankroll_sim.play_round") as play:
                result = simulate(self.params(bankroll=bankroll))
                play.assert_not_called()
                self.assertTrue(result.ruined.all())
                self.assertTrue((result.ttl_hands == 0).all())
                self.assertTrue((result.sample_paths == bankroll).all())
                self.assertTrue((result.ending_bankroll == bankroll).all())
                self.assertTrue((result.survival_prob == 0).all())

    def test_exact_minimum_plays_once_then_stops(self):
        loss = RoundResult(-1, 1, 0, 0, 0, 0, 0, 0, 0)
        with patch("bankroll_sim.play_round", return_value=loss) as play:
            result = simulate(self.params(bankroll=25))
        self.assertEqual(play.call_count, 4)
        self.assertTrue((result.ttl_hands == 1).all())
        self.assertTrue((result.sample_paths[:, 1:] == 0).all())

    def test_positive_remainder_is_never_rebet_or_rounded_up(self):
        loss = RoundResult(-1, 1, 0, 0, 0, 0, 0, 0, 0)
        for start, bet, remainder in ((37.50, 25, 12.50), (49.99, 25, 24.99),
                                      (75, 50, 25), (50.03, 25.02, 25.01)):
            with self.subTest(start=start, bet=bet), patch("bankroll_sim.play_round", return_value=loss):
                result = simulate(self.params(bankroll=start, bet=bet))
                self.assertTrue(result.ruined.all())
                self.assertTrue((result.ending_bankroll == remainder).all())
                self.assertTrue((result.sample_paths[:, -1] == remainder).all())

    def test_final_round_ruin_is_included_in_survival(self):
        loss = RoundResult(-1, 1, 0, 0, 0, 0, 0, 0, 0)
        with patch("bankroll_sim.play_round", return_value=loss):
            result = simulate(self.params(bankroll=25, max_hands=1))
        np.testing.assert_array_equal(result.survival_hands, [0, 1])
        np.testing.assert_array_equal(result.survival_prob, [1, 0])
        self.assertTrue(result.ruined.all())

    def test_soft_h17_strategy_and_unfunded_fallbacks(self):
        self.assertEqual(basic_strategy(HandState([1, 8]), 6, True, False), "double")
        self.assertEqual(basic_strategy(HandState([1, 8]), 6, False, False), "stand")
        for up in range(2, 9):
            self.assertEqual(basic_strategy(HandState([1, 7]), up, False, False), "stand")

    def test_split_aces_neither_hits_nor_doubles_first_hand(self):
        # A,2 and A,3 both remain at their two-card totals; dealer 17 wins.
        result = play_round(FixedShoe([1, 6, 1, 10, 2, 3, 1]), 10)
        self.assertEqual(result.net_units, -2)
        self.assertEqual(result.splits, 1)
        self.assertEqual(result.doubles, 0)

    def test_split_aces_cannot_resplit(self):
        result = play_round(FixedShoe([1, 6, 1, 10, 1, 1, 1]), 10)
        self.assertEqual(result.splits, 1)
        self.assertEqual(result.exposure_units, 2)
        self.assertEqual(result.net_units, -2)

    def test_dealer_natural_prevents_extra_wagers(self):
        result = play_round(FixedShoe([8, 1, 8, 10]), 10)
        self.assertEqual(result.net_units, -1)
        self.assertEqual(result.splits, 0)
        self.assertEqual(result.exposure_units, 1)

    def test_both_naturals_push(self):
        result = play_round(FixedShoe([1, 10, 10, 1]), 10)
        self.assertEqual(result.net_units, 0)
        self.assertEqual(result.pushes, 1)

    def test_split_and_double_funding_boundary(self):
        for bankroll, expected_exposure in ((1.9999, 1), (2, 2), (3, 3), (4, 4)):
            with self.subTest(bankroll=bankroll):
                result = play_round(FixedShoe([8, 6, 8, 10, 3, 3, 10, 10, 10]), bankroll)
                self.assertEqual(result.exposure_units, expected_exposure)
                self.assertLessEqual(result.exposure_units, bankroll)

    def test_four_hand_split_limit_and_eight_unit_exposure(self):
        result = play_round(FixedShoe([8, 6, 8, 10, 8, 3, 8, 3, 3, 3,
                                     10, 10, 10, 10, 10]), 8)
        self.assertEqual(result.splits, 3)
        self.assertEqual(result.doubles, 4)
        self.assertEqual(result.exposure_units, 8)
        self.assertEqual(result.net_units, 8)

    def test_random_rounds_never_risk_more_than_available_cash(self):
        shoe = Shoe(np.random.default_rng(281))
        for bankroll in (1, 1.5, 2, 3, 4, 8):
            for _ in range(1000):
                result = play_round(shoe, bankroll)
                self.assertLessEqual(result.exposure_units, bankroll)
                self.assertLessEqual(abs(result.net_units), result.exposure_units * 1.5)
                self.assertGreaterEqual(bankroll + result.net_units, 0)
                self.assertLessEqual(result.splits, 3)
                self.assertLessEqual(result.das, result.doubles)

    def test_simulation_cash_and_statistics_reconcile(self):
        params = self.params(sims=50, n_trajectories=50, max_hands=100)
        result = simulate(params)
        self.assertAlmostEqual(float((result.ending_bankroll - params.bankroll).sum()),
                               float(result.round_units.sum() * params.bet))
        self.assertEqual(result.hands_played_total, int(result.ttl_hands.sum()))
        np.testing.assert_array_equal(result.ending_bankroll, result.sample_paths[:, -1])
        np.testing.assert_array_equal(result.ruined, result.ending_bankroll < params.bet)
        self.assertTrue((result.sample_paths >= 0).all())
        for i, ttl in enumerate(result.ttl_hands):
            if result.ruined[i]:
                self.assertTrue((result.sample_paths[i, ttl:] == result.ending_bankroll[i]).all())
        for h, probability in zip(result.survival_hands, result.survival_prob):
            self.assertAlmostEqual(probability, np.mean(~result.ruined | (result.ttl_hands > h)))

    def test_export_excludes_survivors_from_ruin_histogram(self):
        import pandas as pd
        params = self.params(bankroll=25, sims=30, max_hands=5)
        result = simulate(params)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "results.xlsx")
            write_excel(params, result, path)
            histogram = pd.read_excel(path, sheet_name="TTL_Distribution")
            self.assertEqual(histogram["count"].sum(), result.ruined.sum())
            self.assertAlmostEqual(histogram["probability"].sum(), result.ruined.mean())
            summary = pd.read_excel(path, sheet_name="Summary")
            self.assertFalse(summary["Metric"].str.startswith("P(ruin) by").any())

    def test_terminal_does_not_extrapolate_beyond_horizon(self):
        params = self.params()
        result = simulate(params)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_terminal_report(params, result)
        self.assertNotIn("by  1h", output.getvalue())

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
        result = play_round(FixedShoe([1, 6, 1, 10, 10, 10, 10]), bankroll_units=10)
        self.assertEqual(result.splits, 1)
        self.assertEqual(result.doubles, 0)
        self.assertEqual(result.net_units, 2.0)

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

    def test_pair_strategy_caps_split_count_in_round_rules(self):
        hand = HandState([8, 8])
        self.assertEqual(basic_strategy(hand, dealer_up=6, allow_double=True, allow_split=True), "split")
        self.assertNotEqual(basic_strategy(hand, dealer_up=6, allow_double=True, allow_split=False), "split")

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


if __name__ == "__main__":
    unittest.main()
