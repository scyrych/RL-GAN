import gym
import numpy as np
import cv2

IMAGE_SIZE = 64


class InputWrapper(gym.ObservationWrapper):
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(self.observation(old_space.low),
                                                self.observation(old_space.high), dtype=np.float32)

    def observation(self, observation):
        new_obs = cv2.resize(observation, (IMAGE_SIZE, IMAGE_SIZE))
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32) / 255.0
