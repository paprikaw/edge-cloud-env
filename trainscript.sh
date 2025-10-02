# ts python trainingmask.py --tag=complete3 --pattern=chain --total_timesteps=10000000
# ts python training.py --tag=complete3 --pattern=chain --total_timesteps=10000000
if [ ! -f scalability.csv ]; then
    echo "num_nodes num_pods pattern algorithm total_timesteps time" >> scalability.csv
fi

python trainingmask.py \
  --device cuda \
  --cpu_num 512 \
  --n_steps 128 \
  --batch_size 1024 \
  --n_epochs 5 \
  --tag gpu-panelty-wr-step \
  --pattern aggregator_sequential \
  --nodes 7 \
  --pods 21 \
  --total_timesteps 20000000 \

# ts python training.py --tag=scalability --nodes=19 --pods=21 --pattern=chain --total_timesteps=20000000
# ts python trainingmask.py --tag=scalability --nodes=13 --pods=21 --pattern=aggregator_parallel --total_timesteps=8000000 >> scalability.csv
# ts python trainingmask.py --tag=scalability --nodes=25 --pods=21 --pattern=chain --total_timesteps=8000000 >> scalability.csv
# ts python trainingmask.py --tag=scalability --nodes=13 --pods=21 --pattern=chain --total_timesteps=8000000 >> scalability.csv

# ts python trainingmask.py --tag=scalability --nodes=10 --pods=25 --pattern=aggregator_parallel --total_timesteps=10000000 >> scalability.csv
# ts python trainingmask.py --tag=scalability --nodes=13 --pods=29 --pattern=aggregator_sequential --total_timesteps=10000000 >> scalability.csv
# ts python trainingmask.py --tag=scalability --nodes=19 --pods=37 --pattern=aggregator_sequential --total_timesteps=10000000 >> scalability.csv
# ts python trainingmask.py --tag=scalability --nodes=10 --pods=25 --pattern=aggregator_sequential --total_timesteps=10000000 >> scalability.csv

# ts python trainingmask.py --tag=scalability --nodes=22 --pods=41 --pattern=aggregator_sequential --total_timesteps=10000000 >> scalability.csv
# ts python trainingmask.py --tag=scalability --nodes=19 --pods=21 --pattern=aggregator_sequential --total_timesteps=8000000 >> scalability.csv

# ts python trainingmask.py --tag=scalability --nodes=13 --pods=21 --pattern=aggregator_parallel --total_timesteps=8000000 >> scalability.csv
# ts python trainingmask.py --tag=scalability --nodes=13 --pods=21 --pattern=aggregator_sequential --total_timesteps=8000000 >> scalability.csv
# ts python training.py --tag=complete3 --nodes=22 --pods=41 --pattern=aggregator_sequential --total_timesteps=10000000 >> scalability.csv
# ts python trainingmask.py --tag=complete3 --nodes=22 --pods=41 --pattern=aggregator_sequential --total_timesteps=10000000 >> scalability.csv

# ts python training.py --tag=scalability --nodes=13 --pods=21 --pattern=chain --total_timesteps=20000000
# ts python trainingmask.py --tag=scalability --nodes=13 --pods=21 --pattern=chain --total_timesteps=20000000 >> scalability.csv

# ts python training.py --tag=scalability --nodes=7 --pods=21 --pattern=chain --total_timesteps=20000000
# ts python trainingmask.py --tag=scalability --nodes=7 --pods=21 --pattern=chain --total_timesteps=20000000 >> scalability.csv

# ts python trainingmask.py --tag=scalability --nodes=7 --pods=21 --pattern=aggregator_parallel --total_timesteps=20000000
# ts python trainingmask.py --tag=scalability --nodes=7 --pods=21 --pattern=aggregator_sequential --total_timesteps=20000000

# ts python trainingmask.py --tag=scalability --nodes=13 --pods=21 --pattern=aggregator_parallel --total_timesteps=20000000
# ts python trainingmask.py --tag=scalability --nodes=13 --pods=21 --pattern=aggregator_sequential --total_timesteps=20000000

# ts python trainingmask.py --tag=scalability --nodes=19 --pods=21 --pattern=aggregator_parallel --total_timesteps=20000000
# ts python trainingmask.py --tag=scalability --nodes=19 --pods=21 --pattern=aggregator_sequential --total_timesteps=20000000
# ts python trainingmask.py --tag=scalability --nodes=25 --pods=21 --pattern=aggregator_parallel --total_timesteps=20000000
# ts python trainingmask.py --tag=scalability --nodes=25 --pods=21 --pattern=aggregator_sequential --total_timesteps=20000000
# ts python trainingmask.py --tag=scalability --nodes=25 --pods=21 --pattern=chain --total_timesteps=20000000
# ts python training.py --tag=scalability --pattern=aggregator_parallel --total_timesteps=10000000
# ts python trainingmask.py --tag=scalability --pattern=aggregator_sequential --total_timesteps=10000000
# ts python training.py --tag=scalability --pattern=aggregator_sequential --total_timesteps=10000000
# ts python trainingmask.py --tag=complete --pattern=aggregator_sequential --total_timesteps=10000000
# ts python trainingmask.py --tag=complete --pattern=chain --total_timesteps=10000000