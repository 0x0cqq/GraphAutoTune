GRAPHAUTOTUNE_HOME=/home/cqq/GraphMining/GraphAutoTuner
GRAPHSET_PATH=/home/cqq/GraphMining/GraphAutoTune

srun -p V100 --gres=gpu:1 ncu --call-stack --nvtx --section SourceCounters --import-source 1 -o report_graph_set%i $GRAPHSET_PATH/build/bin/gpu_graph $GRAPHAUTOTUNE_HOME/data/data_graph_500.txt 4 0111101111011110

# ncu --call-stack --nvtx --section SourceCounters --import-source 1 -o report_graph_set%i