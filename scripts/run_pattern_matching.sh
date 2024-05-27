make -j

NAME=data/patents.bin

# srun -p Big -w nico2 --gres=gpu:1 ./bin/pm $NAME 0111101111011110
srun -p Big -w nico2 --gres=gpu:1 ./bin/pm $NAME 0100110110010110110010100
# srun -p Big -w nico2 --gres=gpu:1 ./bin/pm $NAME 011011101110110101011000110000101000
# srun -p Big -w nico2 --gres=gpu:1 ./bin/pm $NAME 0111111101111111011101110100111100011100001100000


# nsys profile  --stats=true --sample=cpu --trace=cuda,cudnn,cublas,nvtx,osrt,oshmem --cudabacktrace=kernel:1000000,sync:1000000,memory:1000000 --sampling-period=1000000 --wait=all 
# ncu --call-stack --nvtx --set full --import-source 1 -o report%i 
# compute-sanitizer --tool memcheck --print-limit 5
# P1
# 0111101111011110 
# P2
# 0100110110010110110010100
# P3
# 011011101110110101011000110000101000
# P4
# 0111111101111111011101110100111100011100001100000

# 011111101111110111111011111101111110
# 0111110111110111110111110
# srun -p V100 --gres=gpu:1 ./bin/gpu_graph ../../GraphAutoTuner/data/data_graph_20.txt 5 0100110110010110110010100

# 0111111
# 1011111
# 1101110
# 1110100
# 1111000
# 1110000
# 1100000