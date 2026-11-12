#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "XSched::preempt" for configuration "Release"
set_property(TARGET XSched::preempt APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(XSched::preempt PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libpreempt.so"
  IMPORTED_SONAME_RELEASE "libpreempt.so"
  )

list(APPEND _cmake_import_check_targets XSched::preempt )
list(APPEND _cmake_import_check_files_for_XSched::preempt "${_IMPORT_PREFIX}/lib/libpreempt.so" )

# Import target "XSched::halcuda" for configuration "Release"
set_property(TARGET XSched::halcuda APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(XSched::halcuda PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhalcuda.so"
  IMPORTED_SONAME_RELEASE "libhalcuda.so"
  )

list(APPEND _cmake_import_check_targets XSched::halcuda )
list(APPEND _cmake_import_check_files_for_XSched::halcuda "${_IMPORT_PREFIX}/lib/libhalcuda.so" )

# Import target "XSched::shimcuda" for configuration "Release"
set_property(TARGET XSched::shimcuda APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(XSched::shimcuda PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "XSched::halcuda"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libshimcuda.so"
  IMPORTED_SONAME_RELEASE "libshimcuda.so"
  )

list(APPEND _cmake_import_check_targets XSched::shimcuda )
list(APPEND _cmake_import_check_files_for_XSched::shimcuda "${_IMPORT_PREFIX}/lib/libshimcuda.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
