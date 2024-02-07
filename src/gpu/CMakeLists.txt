set(SRCS
    gpu_vertex_set.cu
)

add_library(gpu_graph_mining SHARED ${SRCS})