from stable_baselines3.common.callbacks import BaseCallback
from maskenv import MicroserviceMaskEnv
from env import MicroserviceEnv
class LatencyCallback(BaseCallback):
    def __init__(self, verbose=0, repeat_target=10, num_nodes=7, num_pods=13):
        super().__init__(verbose)
        self.repeat_target = repeat_target
        self.num_nodes = num_nodes
        self.num_pods = num_pods
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        static_env = MicroserviceMaskEnv(is_testing=False, 
                                  num_nodes=self.num_nodes, 
                                  num_pods=self.num_pods, 
                                  dynamic_env=False)
        dynamic_env = MicroserviceMaskEnv(is_testing=False, 
                                  num_nodes=self.num_nodes, 
                                  num_pods=self.num_pods, 
                                  dynamic_env=True)
        acc_after_latency_static = 0
        acc_after_latency_dynamic = 0
        acc_start_latency_static = 0
        acc_start_latency_dynamic = 0
        maximum_step_static = 0
        maximum_step_dynamic = 0
        for _ in range(self.repeat_target):
            obs_static, info_static = static_env.reset()
            obs_dynamic, info_dynamic = dynamic_env.reset()
            done_static = False
            done_dynamic = False
            step_static = 0
            step_dynamic = 0
            acc_start_latency_static += static_env.latency_func()
            acc_start_latency_dynamic += dynamic_env.latency_func()
            while not done_static:
                action_masks = static_env.action_masks()
                action, _states = self.model.predict(obs_static, deterministic=True, action_masks=action_masks)
                obs_static, reward_static, done_static, _, info_static = static_env.step(action)
                static_env.render()
                step_static += 1
                if step_static > maximum_step_static:
                    maximum_step_static = step_static
            while not done_dynamic:
                action_masks = dynamic_env.action_masks()
                action, _states = self.model.predict(obs_dynamic, deterministic=True, action_masks=action_masks)
                obs_dynamic, reward_dynamic, done_dynamic, _, info_dynamic = dynamic_env.step(action)
                dynamic_env.render()
                step_dynamic += 1
                if step_dynamic > maximum_step_dynamic:
                    maximum_step_dynamic = step_dynamic
            acc_after_latency_static += static_env.latency_func()
            acc_after_latency_dynamic += dynamic_env.latency_func()
        self.logger.record('custom_metric/avg_after_latency_static', acc_after_latency_static / self.repeat_target)
        self.logger.record('custom_metric/avg_after_latency_dynamic', acc_after_latency_dynamic / self.repeat_target)
        self.logger.record('custom_metric/avg_start_latency_static', acc_start_latency_static / self.repeat_target)
        self.logger.record('custom_metric/avg_start_latency_dynamic', acc_start_latency_dynamic / self.repeat_target)
        self.logger.record('custom_metric/maximum_step_static', maximum_step_static)
        self.logger.record('custom_metric/maximum_step_dynamic', maximum_step_dynamic)
        return True

class NoMaskLatencyCallback(BaseCallback):
    def __init__(self, verbose=0, repeat_target=10, num_nodes=7, num_pods=13):
        super().__init__(verbose)
        self.repeat_target = repeat_target
        self.num_nodes = num_nodes
        self.num_pods = num_pods
    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            static_env = MicroserviceEnv(num_nodes=self.num_nodes, 
                                  num_pods=self.num_pods, 
                                  dynamic_env=False)
            dynamic_env = MicroserviceEnv(num_nodes=self.num_nodes, 
                                  num_pods=self.num_pods, 
                                  dynamic_env=True)
            acc_after_latency_static = 0
            acc_after_latency_dynamic = 0
            acc_start_latency_static = 0
            acc_start_latency_dynamic = 0
            maximum_step_static = 0
            maximum_step_dynamic = 0
            for _ in range(self.repeat_target):
                obs_static, info_static = static_env.reset()
                obs_dynamic, info_dynamic = dynamic_env.reset()
                done_static = False
                done_dynamic = False
                step_static = 0
                step_dynamic = 0
                acc_start_latency_static += static_env.latency_func()
                acc_start_latency_dynamic += dynamic_env.latency_func()
                while not done_static:
                    action, _states = self.model.predict(obs_static, deterministic=True)
                    obs_static, reward_static, done_static, _, info_static = static_env.step(action)
                    static_env.render()
                    step_static += 1
                if step_static > maximum_step_static:
                    maximum_step_static = step_static
                while not done_dynamic:
                    action, _states = self.model.predict(obs_dynamic, deterministic=True)
                    obs_dynamic, reward_dynamic, done_dynamic, _, info_dynamic = dynamic_env.step(action)
                    dynamic_env.render()
                    step_dynamic += 1
                if step_dynamic > maximum_step_dynamic:
                    maximum_step_dynamic = step_dynamic
                acc_after_latency_static += static_env.latency_func()
                acc_after_latency_dynamic += dynamic_env.latency_func()
            self.logger.record('custom_metric/avg_after_latency_static', acc_after_latency_static / self.repeat_target)
            self.logger.record('custom_metric/avg_after_latency_dynamic', acc_after_latency_dynamic / self.repeat_target)
            self.logger.record('custom_metric/avg_start_latency_static', acc_start_latency_static / self.repeat_target)
            self.logger.record('custom_metric/avg_start_latency_dynamic', acc_start_latency_dynamic / self.repeat_target)
            self.logger.record('custom_metric/maximum_step_static', maximum_step_static)
            self.logger.record('custom_metric/maximum_step_dynamic', maximum_step_dynamic)
        return True