cmake_minimum_required(VERSION 3.10)
project(yolo11_musa LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")

# OpenCV
find_package(OpenCV REQUIRED)

# 直接指定 MTNN 头文件和库（因为你已经知道路径）
set(MTNN_INCLUDE_DIR "/usr/include/npu/include")
set(MTNN_LIBRARY "/usr/lib/libmtnnrt.so")

if(NOT EXISTS ${MTNN_INCLUDE_DIR}/mtnn_api.h)
    message(FATAL_ERROR "mtnn_api.h not found in ${MTNN_INCLUDE_DIR}")
endif()
if(NOT EXISTS ${MTNN_LIBRARY})
    message(FATAL_ERROR "libmtnnrt.so not found at ${MTNN_LIBRARY}")
endif()

add_executable(yolo11_musa src/yolo11_musa.cpp)

target_include_directories(yolo11_musa PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${MTNN_INCLUDE_DIR}
)

target_link_libraries(yolo11_musa
    ${OpenCV_LIBS}
    ${MTNN_LIBRARY}
)

message(STATUS "MTNN include: ${MTNN_INCLUDE_DIR}")
message(STATUS "MTNN library:  ${MTNN_LIBRARY}")