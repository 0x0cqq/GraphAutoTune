GRAPHAUTOTUNE_HOME=/home/cqq/GraphMining/GraphAutoTuner
GRAPHSET_PATH=/home/cqq/GraphMining/GraphAutoTune

srun -p V100 --gres=gpu:1 $GRAPHSET_PATH/build/bin/gpu_graph /home/cqq/data/livejournal.g 6 011011101110110101011000110000101000

# 5 0111110111110111110111110

# $GRAPHAUTOTUNE_HOME/data/livejournal.txt
# ncu --call-stack --nvtx --set full --import-source 1 -o report_graph_set%i