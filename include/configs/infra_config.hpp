#pragma once

enum GraphBackendType { InMemory };

struct InfraConfig {
    GraphBackendType graph_backend_type = InMemory;
};