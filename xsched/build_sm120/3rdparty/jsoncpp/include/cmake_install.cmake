# Install script for directory: /mnt/data/1file/Desktop/code/code/xsched/level3opencode/xsched/3rdparty/jsoncpp/include

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/mnt/data/1file/Desktop/code/code/xsched/level3opencode/xsched/output")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/json" TYPE FILE FILES
    "/mnt/data/1file/Desktop/code/code/xsched/level3opencode/xsched/3rdparty/jsoncpp/include/json/allocator.h"
    "/mnt/data/1file/Desktop/code/code/xsched/level3opencode/xsched/3rdparty/jsoncpp/include/json/assertions.h"
    "/mnt/data/1file/Desktop/code/code/xsched/level3opencode/xsched/3rdparty/jsoncpp/include/json/config.h"
    "/mnt/data/1file/Desktop/code/code/xsched/level3opencode/xsched/3rdparty/jsoncpp/include/json/forwards.h"
    "/mnt/data/1file/Desktop/code/code/xsched/level3opencode/xsched/3rdparty/jsoncpp/include/json/json.h"
    "/mnt/data/1file/Desktop/code/code/xsched/level3opencode/xsched/3rdparty/jsoncpp/include/json/json_features.h"
    "/mnt/data/1file/Desktop/code/code/xsched/level3opencode/xsched/3rdparty/jsoncpp/include/json/reader.h"
    "/mnt/data/1file/Desktop/code/code/xsched/level3opencode/xsched/3rdparty/jsoncpp/include/json/value.h"
    "/mnt/data/1file/Desktop/code/code/xsched/level3opencode/xsched/3rdparty/jsoncpp/include/json/version.h"
    "/mnt/data/1file/Desktop/code/code/xsched/level3opencode/xsched/3rdparty/jsoncpp/include/json/writer.h"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/mnt/data/1file/Desktop/code/code/xsched/level3opencode/xsched/build_sm120/3rdparty/jsoncpp/include/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
