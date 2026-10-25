"""Filter native environment goals before any Actor interaction, including auto-resets."""
import numpy as np
from task_split import manifest, partition_goals, assert_allowed, task_identity
from agent_system.environments.env_package.webshop.envs import WebshopWorker as NativeWebshopWorker


class DisjointWebshopWorker(NativeWebshopWorker):
    def __init__(self, seed, env_kwargs, is_train):
        self.is_train = bool(is_train)
        self.worker_seed = int(seed)
        super().__init__(seed, dict(env_kwargs))
        server = self.env.unwrapped.server
        original = server.goals
        server.goals = partition_goals(original, self.is_train)
        server.weights = [goal['weight'] for goal in server.goals]
        server.cum_weights = [0] + np.cumsum(server.weights).tolist()
        # Discard the native constructor's unfiltered initial random session.
        server.user_sessions.clear()
        self.env.reset(session=0)
        self._current_id()

    def _current_id(self):
        env = self.env.unwrapped
        return assert_allowed(env.server.user_sessions[env.session]['goal'], self.is_train)

    def reset(self, idx):
        if not isinstance(idx, (int, np.integer)) or not 0 <= idx < len(self.env.unwrapped.server.goals):
            raise RuntimeError('Task index outside this split')
        observation, info = super().reset(int(idx))
        info['task_id'] = self._current_id()
        info['task_split'] = 'train' if self.is_train else 'validation'
        return observation, info

    def step(self, action):
        current = self._current_id()
        observation, reward, done, info = super().step(action)
        self._current_id()  # Native automatic resets must also stay in this split.
        info['task_id'] = current
        info['task_split'] = 'train' if self.is_train else 'validation'
        return observation, reward, done, info

    def get_split_audit(self):
        goals = self.env.unwrapped.server.goals
        ids = {task_identity(goal) for goal in goals}
        value = manifest()
        validation = set(value['validation_task_ids'])
        overlap = len(ids & validation) if self.is_train else len(ids - validation)
        return {'is_train': self.is_train, 'worker_seed': self.worker_seed,
                'goal_entries': len(goals), 'unique_task_families': len(ids),
                'opposite_split_overlap': overlap}
