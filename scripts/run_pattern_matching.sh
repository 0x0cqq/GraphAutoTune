make -j && srun -p V100 --gres=gpu:1  ./bin/naive_pm data/data_graph_20.bin 0100110110010110110010100
# ncu --section SourceCounters --import-source 1 -o report%i 
# compute-sanitizer --tool memcheck --print-limit 5
# 0111101111011110
# 0100110110010110110010100
# srun -p V100 --gres=gpu:1 ./bin/gpu_graph ../../GraphAutoTuner/data/data_graph_20.txt 5 0100110110010110110010100