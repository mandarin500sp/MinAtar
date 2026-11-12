"""Run with: python -m unittest discover -s tests -v.

The exhaustive check uses the actual game dynamics, without sticky actions.
An error must move the paddle one cell farther from every successful contact
column, and the next action must undo that move. A blocked command at a wall
does not count. The error may occur after the first action.
"""
import importlib.util
import itertools
import unittest
from unittest.mock import Mock

import numpy as np

from minatar import Environment
from minatar.environments.breakout import Env


class BreakoutResetTests(unittest.TestCase):
    ACTIONS = (0, 1, 3)  # No-op, left, right in the game's full action set.
    STEPS_TO_IMPACT = 6

    def run_sequence(self, env, initial_state, actions):
        """Restore a start and execute actions through the real Env.act()."""
        env.__dict__.update(initial_state)
        env.brick_map = initial_state["brick_map"].copy()
        positions = [env.pos]
        for action in actions:
            env.act(action)
            positions.append(env.pos)

        # Merely surviving is insufficient: require the upward reflection at
        # the first paddle contact, which leaves the ball on row 8.
        bounced = (
            not env.terminal and env.ball_y == 8 and env.ball_dir in (0, 1)
        )
        return bounced, positions

    def contact_columns(self, env, initial_state):
        """Discover contact columns experimentally, without the reset formula."""
        columns = set()
        for paddle_x in range(10):
            start = dict(initial_state, pos=paddle_x)
            bounced, _ = self.run_sequence(
                env, start, (0,) * self.STEPS_TO_IMPACT
            )
            if bounced:
                columns.add(paddle_x)
        self.assertTrue(columns)
        return columns

    def find_recovery(self, env, initial_state, contact_columns):
        """Search all 3**6 action sequences for a bounce with a repaired error."""
        def distance(p):
            return min(abs(p - target) for target in contact_columns)

        for actions in itertools.product(
                self.ACTIONS, repeat=self.STEPS_TO_IMPACT):
            bounced, positions = self.run_sequence(env, initial_state, actions)
            if not bounced:
                continue
            for step in range(self.STEPS_TO_IMPACT - 1):
                before, after, recovered = positions[step:step + 3]
                if (abs(after - before) == 1
                        and distance(after) > distance(before)
                        and recovered == before):
                    return actions
        return None

    def test_every_possible_initial_position(self):
        """Check all 100 ball/paddle column pairs, including excluded pairs.

        Force the ball's random draw so coverage does not depend on a seed
        eventually visiting every column. Inspect the candidate list handed
        to choice(), then independently search the actual dynamics for each
        paddle position. Every admitted pair must have a witness, and every
        excluded pair must have none. Finally replay each witness from an
        actual randomized reset forced to select that admitted position.
        """
        for ball_x in range(10):
            env = Env(reset_mode="randomized")
            rng = Mock()
            rng.randint.return_value = ball_x
            rng.choice.side_effect = lambda candidates: candidates[0]
            env.random = rng
            env.reset()

            self.assertEqual(env.ball_x, ball_x)
            self.assertEqual(env.ball_y, 3)
            self.assertEqual(env.ball_dir, 2 if ball_x < 5 else 3)
            candidates = list(rng.choice.call_args[0][0])
            self.assertEqual(len(candidates), len(set(candidates)))
            self.assertTrue(all(p in range(10) for p in candidates))
            initial_state = dict(env.__dict__)
            initial_state["brick_map"] = env.brick_map.copy()
            columns = self.contact_columns(env, initial_state)

            for paddle_x in range(10):
                with self.subTest(ball_x=ball_x, paddle_x=paddle_x):
                    start = dict(initial_state, pos=paddle_x)
                    witness = self.find_recovery(env, start, columns)
                    self.assertEqual(paddle_x in candidates, witness is not None)
                    if witness is not None:
                        rng.choice.side_effect = None
                        rng.choice.return_value = paddle_x
                        env.reset()
                        self.assertEqual(env.pos, paddle_x)
                        bounced, _ = self.run_sequence(
                            env, dict(env.__dict__), witness
                        )
                        self.assertTrue(bounced)

    def test_default_reset_preserves_seeded_start_sequence(self):
        env = Env()
        env.random = np.random.RandomState(42)
        reference = np.random.RandomState(42)
        bricks = np.zeros((10, 10))
        bricks[1:4, :] = 1
        for _ in range(20):
            expected_x, expected_dir = [(0, 2), (9, 3)][reference.randint(2)]
            env.reset()
            self.assertEqual((env.ball_x, env.ball_dir), (expected_x, expected_dir))
            self.assertEqual((env.pos, env.ball_y), (4, 3))
            self.assertEqual((env.last_x, env.last_y), (env.ball_x, env.ball_y))
            self.assertFalse(env.strike)
            self.assertFalse(env.terminal)
            np.testing.assert_array_equal(env.brick_map, bricks)

    def test_randomized_resets_clear_episode_state(self):
        env = Env(reset_mode="randomized")
        env.brick_map.fill(0)
        env.strike = True
        env.terminal = True
        env.last_x, env.last_y = -1, -1
        env.reset()
        bricks = np.zeros((10, 10))
        bricks[1:4, :] = 1
        np.testing.assert_array_equal(env.brick_map, bricks)
        self.assertEqual((env.last_x, env.last_y), (env.ball_x, env.ball_y))
        self.assertFalse(env.strike)
        self.assertFalse(env.terminal)

    def test_wrapper_preserves_mode_and_seed_reproducibility(self):
        first = Environment("breakout", reset_mode="randomized")
        second = Environment("breakout", reset_mode="randomized")
        first.seed(42)
        second.seed(42)
        for _ in range(20):
            first.reset()
            second.reset()
            self.assertEqual(first.env.reset_mode, "randomized")
            self.assertEqual(second.env.reset_mode, "randomized")
            np.testing.assert_array_equal(first.state(), second.state())
            self.assertEqual(first.env.ball_dir, second.env.ball_dir)

    def test_invalid_modes_are_rejected(self):
        for mode in ("unknown", "", None):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "Unsupported reset_mode"):
                    Env(reset_mode=mode)
                with self.assertRaisesRegex(ValueError, "Unsupported reset_mode"):
                    Environment("breakout", reset_mode=mode)

    def test_other_games_keep_default_reset(self):
        for game in ("asterix", "freeway", "seaquest", "space_invaders"):
            with self.subTest(game=game):
                env = Environment(game)
                env.reset()
                self.assertEqual(env.state().shape, tuple(env.state_shape()))
                with self.assertRaisesRegex(ValueError, "only supported for breakout"):
                    Environment(game, reset_mode="randomized")

    @unittest.skipUnless(importlib.util.find_spec("gymnasium"),
                         "Gymnasium is not installed")
    def test_gymnasium_forwards_reset_mode(self):
        import gymnasium  # Load Gymnasium plugins before importing the wrapper.
        from minatar.gym import BaseEnv

        for mode in ("default", "randomized"):
            with self.subTest(mode=mode):
                env = BaseEnv("breakout", reset_mode=mode, sticky_action_prob=0.0)
                try:
                    observation, _ = env.reset(seed=42)
                    self.assertEqual(env.game.env.reset_mode, mode)
                    self.assertEqual(observation.shape, (10, 10, 4))
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
