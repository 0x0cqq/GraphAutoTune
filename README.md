# GraphAutoTune

## Build

```bash
source env.sh
mkdir build
cd build
cmake ..
make -j
```

## Run

```bash
srun --gres=gpu:1 -p V100 ./build/bin/main
```