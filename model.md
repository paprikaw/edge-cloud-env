v5: 复杂node模型，停止惩罚高
v6: 简单node模型，停止惩罚低
v7: 复杂node模型，停止惩罚低
v12: 开始训练dynamic的模型
    Parameters:
        step_panelty
        stop_reward
        Larger worse latency reward
V13:
使用step acumulated reward可以帮助我们稳定RL agent的steps
使用固定的，相对layer latency的step latency可以帮助我们在较少的step中找到最好的step
	如果layer latency比较小这个时候reward可能不会特别高。在这种情况下，我们就需要使用小一点儿的step pelnalty，增大agent选择valid action的概率
将latency限制在比较小的范围内，因为latency之间影响rewards，所以对agent的选择影响比较大
	如果latency太大，那么RL agent无论如何也应该将pod 尽量schedule到edge layer中
在dynamic environment中训练.
10 log:
    native-exponential-reward: 很天真的把relative reward直接赋给exponential reward
v14:
    试图使用使用acumulated reward, 加上当前的latency作为step reward.
    80-0.1:不行，太保守了，学习的比较慢
    40-0.5: 同样的问题
    10000-1: 效率太低了
    10000-0.5: 候选
    200-0.05: 不稳定(step)
    400-0.05: 不稳定(step)
    80-0.05: 如果后面这个项太大，最后的步数也会多
    200-0.5: 候选
    我感觉关键在于agent学习的速度
v15: 
    使用beta variate，增加集群high availability下的数据
    