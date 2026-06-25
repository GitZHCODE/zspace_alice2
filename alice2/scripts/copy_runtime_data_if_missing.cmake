if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR is required")
endif()

if(NOT DEFINED DEST_DIR)
    message(FATAL_ERROR "DEST_DIR is required")
endif()

file(MAKE_DIRECTORY "${DEST_DIR}")
file(GLOB_RECURSE RUNTIME_DATA_FILES RELATIVE "${SOURCE_DIR}" "${SOURCE_DIR}/*")

foreach(DATA_FILE IN LISTS RUNTIME_DATA_FILES)
    set(SOURCE_FILE "${SOURCE_DIR}/${DATA_FILE}")
    set(DEST_FILE "${DEST_DIR}/${DATA_FILE}")

    if(IS_DIRECTORY "${SOURCE_FILE}")
        file(MAKE_DIRECTORY "${DEST_FILE}")
    elseif(NOT EXISTS "${DEST_FILE}")
        get_filename_component(DEST_PARENT "${DEST_FILE}" DIRECTORY)
        file(MAKE_DIRECTORY "${DEST_PARENT}")
        file(COPY "${SOURCE_FILE}" DESTINATION "${DEST_PARENT}")
    endif()
endforeach()
