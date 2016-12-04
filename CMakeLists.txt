cmake_minimum_required(VERSION 2.6)
project(include/chianti)

find_package( OpenCV REQUIRED )

include_directories(include /media/toby/d/cntk/Include lib/eigen-3.3.0)
link_directories(/media/toby/d/cntk/cntk/lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#set(SOURCE_FILES main.cpp Common.cpp)
#add_executable(cntk_demo ${SOURCE_FILES})
add_executable(frrn src/main.cpp)
#target_link_libraries(cntk_demo cntklibrary-2.0 cntkmath)
target_link_libraries(frrn cntklibrary-2.0 cntkmath ${OpenCV_LIBS})