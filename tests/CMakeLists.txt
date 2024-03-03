add_executable(vertex_set_test vertex_set_test.cu)
target_link_libraries(vertex_set_test PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(vertex_set_test)


add_executable(engine_test engine_test.cu)
target_link_libraries(engine_test PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(engine_test)
