make -j && srun -p V100 --gres=gpu:1 ncu --section SourceCounters --import-source 1 -o report%i  ./bin/pm data/data_graph_20.bin 0111101111011110
# ncu --section SourceCounters --import-source 1 -o report%i 
# 0111101111011110
# 0100110110010110110010100