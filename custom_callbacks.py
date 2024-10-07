from stable_baselines3.common.callbacks import BaseCallback
from maskenv import MicroserviceMaskEnv
class LatencyCallback(BaseCallback):
    def __init__(self, verbose=0, repeat_target=10, num_nodes=7, num_pods=13, relative_para=20, accumulated_para=0.05, final_reward=100):
        super().__init__(verbose)
        self.repeat_target = repeat_target
        self.num_nodes = num_nodes
        self.num_pods = num_pods
        self.relative_para = relative_para
        self.accumulated_para = accumulated_para
        self.final_reward = final_reward
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        env1 = MicroserviceMaskEnv(is_training=True, 
                                  num_nodes=self.num_nodes, 
                                  num_pods=self.num_pods, 
                                  dynamic_env=False, 
                                  relative_para=self.relative_para, 
                                  accumulated_para=self.accumulated_para,
                                  final_reward=self.final_reward)
        env2 = MicroserviceMaskEnv(is_training=True, 
                                  num_nodes=self.num_nodes, 
                                  num_pods=self.num_pods, 
                                  dynamic_env=True, 
                                  relative_para=self.relative_para, 
                                  accumulated_para=self.accumulated_para,
                                  final_reward=self.final_reward)
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
        self.logger.record('custom_metric/maximum_step_static', maximum_step1)
        self.logger.record('custom_metric/maximum_step_dynamic', maximum_step2)
        return True