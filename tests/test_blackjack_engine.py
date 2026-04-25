import unittest

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
)


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
