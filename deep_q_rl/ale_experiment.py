"""The ALEExperiment class handles the logic for training a deep
Q-learning agent in the Arcade Learning Environment.

Author: Nathan Sprague

"""
import logging
import numpy as np
import cv2

# Number of rows to crop off the bottom of the (downsampled) screen.
# This is appropriate for breakout, but it may need to be modified
# for other games.
CROP_OFFSET = 8


class ALEExperiment(object):
    def __init__(self, gym_env, agent, resized_width, resized_height,
                 resize_method, num_epochs, epoch_length, test_length,
                 frame_skip, death_ends_episode, max_start_nullops, rng):
        self.gym_env = gym_env
        self.agent = agent
        self.num_epochs = num_epochs
        self.epoch_length = epoch_length
        self.test_length = test_length
        self.frame_skip = frame_skip
        self.death_ends_episode = death_ends_episode
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.resize_method = resize_method

        self.buffer_length = 2
        self.buffer_count = 0
        self.ram_size = 128  # TODO: pass as an argument
        self.current_ram = np.empty((self.ram_size,), dtype=np.uint8)

        self.terminal_lol = False  # Most recent episode ended on a loss of life
        self.max_start_nullops = max_start_nullops
        self.rng = rng

    def run(self):
        """
        Run the desired number of training epochs, a testing epoch
        is conducted after each training epoch.
        """
        for epoch in range(1, self.num_epochs + 1):
            self.run_epoch(epoch, self.epoch_length)
            self.agent.finish_epoch(epoch)

            if self.test_length > 0:
                self.run_tests(epoch, self.test_length)

    def run_tests(self, epoch, test_length):
        self.agent.start_testing()
        self.run_epoch(epoch, test_length, True)
        self.agent.finish_testing(epoch)

    def run_epoch(self, epoch, num_steps, testing=False):
        """ Run one 'epoch' of training or testing, where an epoch is defined
        by the number of steps executed.  Prints a progress report after
        every trial

        Arguments:
        epoch - the current epoch number
        num_steps - steps per epoch
        testing - True if this Epoch is used for testing and not training

        """
        self.terminal_lol = False  # Make sure each epoch starts with a reset.
        steps_left = num_steps
        while steps_left > 0:
            _, num_steps = self.run_episode(steps_left, testing)

            steps_left -= num_steps

    def _init_episode(self):
        """ This method resets the game if needed, performs enough null
        actions to ensure that the screen buffer is ready and optionally
        performs a randomly determined number of null action to randomize
        the initial game state."""

        self.gym_env.reset()

        if self.max_start_nullops > 0:
            random_actions = self.rng.randint(0, self.max_start_nullops+1)
            for _ in range(random_actions):
                self._act(0)  # Null action

        # Make sure the screen buffer is filled at the beginning of
        # each episode...
        self._act(0)
        self._act(0)

    def _act(self, action_id):
        """Perform the indicated action for a single frame, return the
        resulting reward and store the resulting screen image in the
        buffer

        """
        obs, reward, done, _ = self.gym_env.step(action_id)
        self.current_ram = obs
        self.buffer_count += 1
        return reward, done

    def _step(self, action_id):
        """ Repeat one action the appopriate number of times and return
        the summed reward. """
        reward = 0
        done = False
        for _ in range(self.frame_skip):  # TODO: reduce frameskip
            local_reward, local_done = self._act(action_id)
            reward += local_reward
            if local_done:
                done = True
                break

        return reward, done

    def run_episode(self, max_steps, testing):
        """Run a single training episode.

        The boolean terminal value returned indicates whether the
        episode ended because the game ended or the agent died (True)
        or because the maximum number of steps was reached (False).
        Currently this value will be ignored.

        Return: (terminal, num_steps)

        """
        print "running episode, steps left", max_steps

        self._init_episode()

        action = self.agent.start_episode(self.current_ram)
        num_steps = 0
        while True:
            reward, terminal = self._step(action)
            num_steps += 1

            if terminal or num_steps >= max_steps:
                self.agent.end_episode(reward, terminal)
                break

            action = self.agent.step(reward, self.current_ram)
        return terminal, num_steps
