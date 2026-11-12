# - Config file for the XSched package
# It defines the following variables:
#  XSched_FOUND - True if XSched has been found
#  XSched_INCLUDE_DIRS - The paths where to find XSched headers
#  XSched_VERSION - The version of XSched
#  XSched_VERSION_MAJOR - The major version of XSched
#  XSched_VERSION_MINOR - The minor version of XSched
#  XSched_VERSION_PATCH - The patch version of XSched


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include("${CMAKE_CURRENT_LIST_DIR}/XSchedTargets.cmake")

set_and_check(XSched_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include")

# versions
set(XSched_VERSION 1.2.0)
set(XSched_VERSION_MAJOR 1)
set(XSched_VERSION_MINOR 2)
set(XSched_VERSION_PATCH 0)

check_required_components(XSched)
