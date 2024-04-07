N=5000
M=500000

./bin/random_graph $N $M > ../data/data_graph_$N.txt

./bin/graph_transform data_graph_$N