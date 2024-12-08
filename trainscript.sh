# ts python training.py --tag=complete --pattern=aggregator_parallel --total_timesteps=10000000
# ts python training.py --tag=complete --pattern=aggregator_sequential --total_timesteps=10000000
# ts python training.py --tag=complete --pattern=chain --total_timesteps=10000000
ts python trainingmask.py --tag=paradise --pattern=aggregator_parallel --total_timesteps=5000000
# ts python trainingmask.py --tag=complete --pattern=aggregator_sequential --total_timesteps=10000000
# ts python trainingmask.py --tag=complete --pattern=chain --total_timesteps=10000000