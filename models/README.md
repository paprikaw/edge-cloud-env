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

log:
    native-exponential-reward: 很天真的把relative reward直接赋给exponential reward

v14:
    试图使用relative reward