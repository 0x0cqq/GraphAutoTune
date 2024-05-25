GRAPHAUTOTUNE_HOME=/home/cqq/GraphMining/GraphAutoTuner

# srun -p V100 --gres=gpu:1 
python $GRAPHAUTOTUNE_HOME/tuning/main.py data/data_graph_500.bin 0100110110010110110010100 -c 