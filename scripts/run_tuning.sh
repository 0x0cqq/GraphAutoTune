GRAPHAUTOTUNE_HOME=/home/cqq/GraphMining/GraphAutoTuner
source $GRAPHAUTOTUNE_HOME/env.sh

# srun -p V100 --gres=gpu:1 
python $GRAPHAUTOTUNE_HOME/tuning/main.py data/patents.bin 0100110110010110110010100 -c 