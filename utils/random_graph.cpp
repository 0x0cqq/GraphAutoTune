#include <iostream>

int main(int argc, char *argv[]) {
    // N vertexes and M edges
    int N;
    int M;
    // first parameter: N
    if (argc > 1) {
        N = atoi(argv[1]);
    } else {
        N = 20;
    }
    // second parameter: M
    if (argc > 2) {
        M = atoi(argv[2]);
    } else {
        M = 100;
    }

    bool **adj = new bool *[N];
    for (int i = 0; i < N; i++) adj[i] = new bool[N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) adj[i][j] = false;
    }
    for (int i = 0; i < M; i++) {
        int u = rand() % N, v = rand() % N;
        while (u == v || adj[u][v]) {
            u = rand() % N;
            v = rand() % N;
        }
        adj[u][v] = adj[v][u] = true;
    }

    // 输出点数和边数
    std::cout << N << " " << M << std::endl;
    // 输出边
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            if (adj[i][j]) {
                std::cout << i << " " << j << std::endl;
            }
        }
    }
    return 0;
}