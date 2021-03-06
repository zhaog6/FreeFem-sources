INCLUDE(FindPackageHandleStandardArgs)
INCLUDE(PackageManagerPaths)

FIND_LIBRARY(UMFPACK_LIBRARIES NAMES umfpack 
                              PATHS ${PACKMAN_LIBRARIES_PATHS})

IF(UMFPACK_LIBRARIES)
  SET(UMFPACK_FOUND True)
ENDIF(UMFPACK_LIBRARIES)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(UMFPACK DEFAULT_MSG UMFPACK_LIBRARIES)



