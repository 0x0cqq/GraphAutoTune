make -j && srun -p Big --gres=gpu:1 ./bin/pm data/data_graph_5000.bin 0100110110010110110010100

# nsys profile  --stats=true --sample=cpu --trace=cuda,cudnn,cublas,nvtx,osrt,oshmem --cudabacktrace=kernel:1000000,sync:1000000,memory:1000000 --sampling-period=1000000  --delay=10 --duration=60 --wait=all 
# ncu --call-stack --nvtx --set full --import-source 1 -o report%i 
# compute-sanitizer --tool memcheck --print-limit 5
# 0111101111011110
# 0100110110010110110010100
# 0111110111110111110111110
# srun -p V100 --gres=gpu:1 ./bin/gpu_graph ../../GraphAutoTuner/data/data_graph_20.txt 5 0100110110010110110010100