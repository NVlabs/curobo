# This macro will install ${PROJECT_NAME} to ${CMAKE_INSTALL_PREFIX} when running make install
#
# FlexivInstallLibrary() will install all subfolders of ${CMAKE_CURRENT_SOURCE_DIR}/include
# FlexivInstallLibrary(install_directories) will install only the specified install_directories
#
# Requirements:
# 1. project structure should resemble:
#    project
#     - README.md
#     - CMakeLists.txt that calls this macro
#     - cmake/${PROJECT_NAME}-config.cmake.in
#     - include/subfolder/*.h or *.hpp
# 2. build the library using cmake target functions
#    - add_library(${PROJECT_NAME} ...) before calling this macro
#    - target_include_directories(${PROJECT_NAME} ...)
#    - target_link_libraries(${PROJECT_NAME} ...)
#    - target_compile_features(${PROJECT_NAME} ...)
#    - target_compile_options(${PROJECT_NAME} ...)
#
# Installed files:
# - include/subfolder/*.h or *.hpp
# - lib/lib{PROJECT_NAME}
# - lib/cmake/{PROJECT_NAME}/

macro(FlexivInstallLibrary)
    # copy the executables and libraries to the CMAKE_INSTALL_PREFIX DIRECTORY
    # GNUInstallDirs will set CMAKE_INSTALL* to be the standard relative paths
    include(GNUInstallDirs)
    install(TARGETS ${PROJECT_NAME}
        EXPORT "${PROJECT_NAME}-targets"
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )

    if(${ARGC} EQUAL 0)
        # install all subfolders of ${CMAKE_CURRENT_SOURCE_DIR}/include
        file(GLOB install_directories ${CMAKE_CURRENT_SOURCE_DIR}/include/*)
        foreach(install_directory ${install_directories})
            if(IS_DIRECTORY ${install_directory})
                install(DIRECTORY ${install_directory}
                        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
                        FILES_MATCHING 
                        PATTERN "*.h"
                        PATTERN "*.hpp"
                )
            endif()
        endforeach()
    elseif(${ARGC} EQUAL 1)
        # install specified directories only
        foreach(install_directory ${ARGV0})
            install(DIRECTORY ${install_directory}
                    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
                    FILES_MATCHING 
                    PATTERN "*.h"
                    PATTERN "*.hpp"
                    )
        endforeach()
    else()
        message(FATAL_ERROR "FlexivInstallLibrary take 0 or 1 argument, but given ${ARGC}")
    endif()

    # Create a *config-version.cmake file so that find_package can have a version specified
    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
        "${PROJECT_NAME}-config-version.cmake"
        VERSION ${PACKAGE_VERSION}
        COMPATIBILITY AnyNewerVersion
        )

    # copy the *-targets.cmake file to the CMAKE_INSTALL_PREFIX directory
    install(EXPORT "${PROJECT_NAME}-targets"
            FILE "${PROJECT_NAME}-targets.cmake"
            NAMESPACE "flexiv::"
            DESTINATION "lib/cmake/${PROJECT_NAME}"
            )

    # copy the *.-config file to the CMAKE_INSTALL_PREFIX directory. This will specify the dependencies.
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}-config.cmake.in" "${PROJECT_NAME}-config.cmake" @ONLY)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
                  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
            DESTINATION "lib/cmake/${PROJECT_NAME}"
            )

    # Use the CPack Package Generator
    set(CPACK_PACKAGE_VENDOR "Flexiv")
    set(CPACK_PACKAGE_CONTACT "support@flexiv.com")
    set(CPACK_PACKAGE_DESCRIPTION "Flexiv Robotic Development Kit (RDK)")
    set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
    set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
    set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
    set(CPACK_RESOURCE_FILE_README  "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
    include(CPack)
endmacro()