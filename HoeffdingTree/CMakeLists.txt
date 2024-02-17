cmake_minimum_required(VERSION 3.24)

project(HoeffdingTree VERSION 1.0 DESCRIPTION "c implementation of the Hoeffding Tree algorithm")

add_library(HoeffdingTree STATIC HoeffdingTree.c)

target_include_directories(HoeffdingTree PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
