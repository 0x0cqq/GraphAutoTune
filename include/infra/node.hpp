#pragma once

enum NodeType_t {
    CPU_NODE,
    GPU_NODE,
};

class NodeInfo {
    int node_id;
    int node_num;
    NodeType_t node_type;
};