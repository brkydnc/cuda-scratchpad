cmake_minimum_required(VERSION 3.29)
project(CUDA_Playground LANGUAGES CXX CUDA)

add_compile_options(-g -G)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(CUDAToolkit REQUIRED)
add_executable(Main main.cu)
target_link_libraries(Main PRIVATE CUDA::curand)
target_link_libraries(Main PRIVATE CUDA::cublas)
