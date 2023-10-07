cmake_minimum_required(VERSION 3.0.2)
project(tao_Pointpillars)
set(CMAKE_CXX_STANDARD 14)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)

add_message_files(
  FILES
  bbox.msg
  result.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES lidar_det
  CATKIN_DEPENDS 
  roscpp 
  rospy
  std_msgs 
  message_runtime
#  DEPENDS system_lib
)

##############CUDA###################
find_package(CUDA REQUIRED)

set(CUDA_VERSION 11.3)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-${CUDA_VERSION})

SET(CMAKE_BUILD_TYPE "Release")
add_compile_options(-W)
add_compile_options(-std=c++11)

set(SMS 50 52 53 60 61 62 70 72 75 80 86)
foreach(sm ${SMS})
    set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()
list(GET SMS -1 LATEST_SM)
set(GENCODE "${GENCODE} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -ccbin ${CMAKE_CXX_COMPILER}
    -Xcompiler -DWIN_INTERFACE_CUSTOM
    -Xcompiler -I/usr/${ARCH}-linux-gnu/include/
    -Xlinker -lsocket
    -Xlinker -rpath=/usr/lib/${ARCH}-linux-gnu/
    -Xlinker -rpath=/usr/${ARCH}-linux-gnu/lib/
    -Xlinker -L/usr/lib/${ARCH}-linux-gnu/
    -Xlinker -L/usr/${ARCH}-linux-gnu/lib/
)

set(TENSORRT_INCLUDE_DIRS /usr/include/${ARCH}-linux-gnu/)
set(TENSORRT_LIBRARY_DIRS /usr/lib/${ARCH}-linux-gnu/)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
    ./include/
)

link_directories(
    ${TENSORRT_LIBRARY_DIRS}
    /usr/lib/${ARCH}-linux-gnu
    /usr/${ARCH}-linux-gnu/lib/
)
##############CUDA###################

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

file(GLOB_RECURSE SOURCE_FILES
    src/*.cpp
)


add_executable(${PROJECT_NAME} demo.cpp ${SOURCE_FILES})
add_dependencies(${PROJECT_NAME} tao_Pointpillars_generate_messages_cpp ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES}  ${PCL_LIBRARIES} ${CUDA_LIBRARIES})

target_link_libraries(${PROJECT_NAME}
    libnvinfer.so
    libnvonnxparser.so
    libnvinfer_plugin.so
)
