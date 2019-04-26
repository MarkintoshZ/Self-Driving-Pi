# -*- coding: utf-8 -*-

import numpy as np


class ExperienceReplay():
    def __init__(self, _max_memory=10000, env_dim=2, num_actions=2,
                 discount=.9):
        self.max_memory = _max_memory
        self.memory = np.array(tuple(tuple()))
        self.discount = discount
        self.num_actions = num_actions
        self.env_dim = [0, env_dim]

    def remember(self, states, _game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        try:
            self.memory = np.append(self.memory, [[states, _game_over]], axis=0)
        except:
            self.memory = np.array([[states, _game_over]])
        if len(self.memory) > self.max_memory:
            self.memory = self.memory[1:]

    def get_batch(self, _model, _batch_size=32):
        len_memory = len(self.memory)
        self.env_dim[0] = min(len_memory, _batch_size)
        _inputs = np.zeros(self.env_dim)
        _targets = np.zeros((self.env_dim[0], self.num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=_inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            _game_over = self.memory[idx][1]
            _inputs[i:i+1] = state_t
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            _targets[i] = _model.predict(np.expand_dims(state_t, 0))[0]
            if _game_over:  # if game_over is True
                _targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                q_sa = np.max(_targets[i])
                _targets[i] = reward_t + self.discount * q_sa
        return _inputs, _targets
