list(APPEND ELEMENTS "vertex_set" "schedule" "graph")

foreach(ELEMENT IN LISTS ELEMENTS)
    add_executable(test_${ELEMENT} test_${ELEMENT}.cu)
    target_link_libraries(test_${ELEMENT} PRIVATE Catch2::Catch2WithMain)
    catch_discover_tests(test_${ELEMENT})
endforeach()