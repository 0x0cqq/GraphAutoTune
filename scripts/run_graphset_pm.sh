GRAPHAUTOTUNE_HOME=/home/cqq/GraphMining/GraphAutoTuner
GRAPHSET_PATH=/home/cqq/GraphMining/GraphAutoTune

NAME=orkut.g

srun -p V100 --gres=gpu:1 --exclusive $GRAPHSET_PATH/build/bin/gpu_graph /home/cqq/data/$NAME 4 0111101111011110
srun -p V100 --gres=gpu:1 --exclusive $GRAPHSET_PATH/build/bin/gpu_graph /home/cqq/data/$NAME 5 0100110110010110110010100
srun -p V100 --gres=gpu:1 --exclusive $GRAPHSET_PATH/build/bin/gpu_graph /home/cqq/data/$NAME 6 011011101110110101011000110000101000
srun -p V100 --gres=gpu:1 --exclusive $GRAPHSET_PATH/build/bin/gpu_graph /home/cqq/data/$NAME 7 0111111101111111011101110100111100011100001100000

# 5 0111110111110111110111110

# $GRAPHAUTOTUNE_HOME/data/livejournal.txt
# ncu --call-stack --nvtx --set full --import-source 1 -o report_graph_set%i