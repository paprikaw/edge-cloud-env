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
        env1 = MicroserviceMaskEnv(is_testing=False, 
                                  num_nodes=self.num_nodes, 
                                  num_pods=self.num_pods, 
                                  dynamic_env=False)
        env2 = MicroserviceMaskEnv(is_testing=False, 
                                  num_nodes=self.num_nodes, 
                                  num_pods=self.num_pods, 
                                  dynamic_env=True)
        acc_after_latency1 = 0
        acc_after_latency2 = 0
        maximum_step1 = 0
        maximum_step2 = 0
        for _ in range(self.repeat_target):
            obs1, info1 = env1.reset()
            obs2, info2 = env2.reset()
            done1 = False
            done2 = False
            step1 = 0
            step2 = 0
            acc_start_latency1 = env1.latency_func()
            acc_start_latency2 = env2.latency_func()
            while not done1:
                action_masks = env1.action_masks()
                action, _states = self.model.predict(obs1, deterministic=True, action_masks=action_masks)
                obs1, reward1, done1, _, info1 = env1.step(action)
                env1.render()
                step1 += 1
                if step1 > maximum_step1:
                    maximum_step1 = step1
            while not done2:
                action_masks = env2.action_masks()
                action, _states = self.model.predict(obs2, deterministic=True, action_masks=action_masks)
                obs2, reward2, done2, _, info2 = env2.step(action)
                env2.render()
                step2 += 1
                if step2 > maximum_step2:
                    maximum_step2 = step2
            acc_after_latency1 += env1.latency_func()
            acc_after_latency2 += env2.latency_func()
        self.logger.record('custom_metric/avg_after_latency_static', acc_after_latency1 / self.repeat_target)
        self.logger.record('custom_metric/avg_after_latency_dynamic', acc_after_latency2 / self.repeat_target)
        self.logger.record('custom_metric/avg_start_latency_static', acc_start_latency1 / self.repeat_target)
        self.logger.record('custom_metric/avg_start_latency_dynamic', acc_start_latency2 / self.repeat_target)
        self.logger.record('custom_metric/maximum_step_static', maximum_step1)
        self.logger.record('custom_metric/maximum_step_dynamic', maximum_step2)
        return True

class NoMaskLatencyCallback(BaseCallback):
    def __init__(self, verbose=0, repeat_target=10, num_nodes=7, num_pods=13):
        super().__init__(verbose)
        self.repeat_target = repeat_target
        self.num_nodes = num_nodes
        self.num_pods = num_pods
    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            env1 = MicroserviceEnv(num_nodes=self.num_nodes, 
                                  num_pods=self.num_pods, 
                                  dynamic_env=False,
                                  is_eval=True
                                  )
            env2 = MicroserviceEnv(num_nodes=self.num_nodes, 
                                  num_pods=self.num_pods, 
                                  dynamic_env=True,
                                  is_eval=True)
            acc_after_latency1 = 0
            acc_after_latency2 = 0
            maximum_step1 = 0
            maximum_step2 = 0
            for _ in range(self.repeat_target):
                obs1, info1 = env1.reset()
                obs2, info2 = env2.reset()
                done1 = False
                done2 = False
                step1 = 0
                step2 = 0
                acc_start_latency1 = env1.latency_func()
                acc_start_latency2 = env2.latency_func()
                while not done1:
                    action, _states = self.model.predict(obs1, deterministic=True)
                    obs1, reward1, done1, _, info1 = env1.step(action)
                    env1.render()
                    step1 += 1
                    if step1 > maximum_step1:
                        maximum_step1 = step1
                while not done2:
                    action, _states = self.model.predict(obs2, deterministic=True)
                    obs2, reward2, done2, _, info2 = env2.step(action)
                    env2.render()
                    step2 += 1
                    if step2 > maximum_step2:
                        maximum_step2 = step2
                acc_after_latency1 += env1.latency_func()
                acc_after_latency2 += env2.latency_func()
            self.logger.record('custom_metric/avg_after_latency_static', acc_after_latency1 / self.repeat_target)
            self.logger.record('custom_metric/avg_after_latency_dynamic', acc_after_latency2 / self.repeat_target)
            self.logger.record('custom_metric/avg_start_latency_static', acc_start_latency1 / self.repeat_target)
            self.logger.record('custom_metric/avg_start_latency_dynamic', acc_start_latency2 / self.repeat_target)
            self.logger.record('custom_metric/maximum_step_static', maximum_step1)
            self.logger.record('custom_metric/maximum_step_dynamic', maximum_step2)
        return True