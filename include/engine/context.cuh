#pragma once
#include "configs/config.hpp"
#include "core/schedule.hpp"

namespace Engine {
// 整个 Device 所需要的所有变量一起打包进来，如 ScheduleData 等。
// 这里面的变量应当是不会改变的，我们将会将其整体放进 constant memory 中。
// 通过直接给 Kernel 函数传参
// 之后考虑通过元编程传递一部分。
template <Config config>
struct DeviceContext {
    using GraphBackend = GraphBackendTypeDispatcher<config>::type;
    // 图挖掘的 Schedule
    Core::ScheduleData schedule_data;
    // 提供图数据访问的后端
    GraphBackend graph_backend;

    __host__ DeviceContext(const Core::Schedule &_schedule,
                           const GraphBackend &_graph_backend)
        : schedule_data(_schedule), graph_backend(_graph_backend) {}

    __host__ void to_device() {
        schedule_data.to_device();
        graph_backend.to_device();
    }
};

}