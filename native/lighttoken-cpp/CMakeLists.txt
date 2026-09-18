cmake_minimum_required(VERSION 3.24)
project(lighttoken_accel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(lighttoken_accel SHARED src/lighttoken_accel.cpp)
target_include_directories(lighttoken_accel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(lighttoken_accel_tests tests/test_accel.cpp)
target_link_libraries(lighttoken_accel_tests PRIVATE lighttoken_accel)

enable_testing()
add_test(NAME lighttoken_accel_known_vectors COMMAND lighttoken_accel_tests)
