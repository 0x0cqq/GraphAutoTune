GRAPHAUTOTUNE_HOME=/home/cqq/GraphMining/GraphAutoTuner
GRAPHSET_PATH=/home/cqq/GraphMining/GraphAutoTune

srun -p V100 --gres=gpu:1 $GRAPHSET_PATH/build/bin/gpu_graph /home/cqq/data/orkut.g 7 0111111101111111011101110100111100011100001100000

# 5 0111110111110111110111110

# $GRAPHAUTOTUNE_HOME/data/livejournal.txt
# ncu --call-stack --nvtx --set full --import-source 1 -o report_graph_set%i