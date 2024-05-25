N=1000
M=50000

./bin/random_graph $N $M > ../data/data_graph_$N.txt

./bin/graph_transform data_graph_$N