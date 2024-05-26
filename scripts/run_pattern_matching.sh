make -j && srun -p Big -w nico2 --gres=gpu:1 ./bin/pm data/livejournal.bin 011011101110110101011000110000101000

# nsys profile  --stats=true --sample=cpu --trace=cuda,cudnn,cublas,nvtx,osrt,oshmem --cudabacktrace=kernel:1000000,sync:1000000,memory:1000000 --sampling-period=1000000 --wait=all 
# ncu --call-stack --nvtx --set full --import-source 1 -o report%i 
# compute-sanitizer --tool memcheck --print-limit 5
# 0111101111011110
# 0100110110010110110010100
# 0111110111110111110111110

# 011011101110110101011000110000101000
# 011111101111110111111011111101111110
# srun -p V100 --gres=gpu:1 ./bin/gpu_graph ../../GraphAutoTuner/data/data_graph_20.txt 5 0100110110010110110010100