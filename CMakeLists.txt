cmake_minimum_required(VERSION 2.6)
project(include/chianti)

find_package( OpenCV REQUIRED )

enable_testing()

include_directories(include lib/googletest-1.8.0/googlemock lib/googletest-1.8.0/googletest /media/toby/d/cntk/Include lib/eigen-3.3.0)
link_directories(/media/toby/d/cntk/cntk/lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_subdirectory(lib/googletest-1.8.0)

# Build the test suite
add_executable(tests
        test/layers.cpp
        test/values.cpp)

target_link_libraries(tests
        cntklibrary-2.0
        ${OpenCV_LIBS}
        gtest gtest_main
        gmock)