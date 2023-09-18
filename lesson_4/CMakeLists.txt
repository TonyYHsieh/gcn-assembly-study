cmake_minimum_required(VERSION 3.16)
project(lesson_4)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
add_definitions(-D__HIP__)
add_executable(${PROJECT_NAME} main.cpp)

find_package(hip REQUIRED)
target_link_libraries(${PROJECT_NAME} hip::device)
include(../asm.cmake)
add_custom_target(asm_amax_target)
add_dependencies(${PROJECT_NAME} asm_amax_target)
gen_and_build_asm(amax asm_amax_target)
