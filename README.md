# GraphAutoTune

```bash
source env.sh
mkdir build
cd build
cmake ..
make -j
```

```bash
srun --gres=gpu:1 -p V100 ./build/bin/main
```