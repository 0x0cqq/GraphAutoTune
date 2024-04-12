GRAPHAUTOTUNE_HOME=/home/cqq/GraphMining/GraphAutoTuner
GRAPHSET_PATH=/home/cqq/GraphMining/GraphAutoTune

srun -p V100 --gres=gpu:1 $GRAPHSET_PATH/build/bin/gpu_graph $GRAPHAUTOTUNE_HOME/data/wiki_vote.txt 5 0100110110010110110010100

# ncu --call-stack --nvtx --set full --import-source 1 -o report_graph_set%i