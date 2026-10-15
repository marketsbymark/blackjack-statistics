"""Exercise the actual dashboard and its engine through Streamlit's test runner."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class DashboardTests(unittest.TestCase):
    def test_render_changed_inputs_and_unplayable_bankroll(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 10)

        app.number_input[1].set_value(24.99).run()
        self.assertTrue(any("Inputs changed" in item.value for item in app.info))
        app.button[0].click().run()
        self.assertFalse(app.exception)
        result = app.session_state["result"]
        self.assertEqual(result.hands_played_total, 0)
        self.assertTrue((result.sample_paths == 24.99).all())
        self.assertEqual(app.metric[0].value, "100.0%")
        self.assertEqual(app.metric[5].value, "N/A")

        app.number_input[0].set_value(25.01).run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)


if __name__ == "__main__":
    unittest.main()
