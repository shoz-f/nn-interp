cmake_minimum_required(VERSION 3.13)

# project start here
project(nn_interp)

include(FetchContent)
set(THIRD_PARTY ${CMAKE_SOURCE_DIR}/3rd_party)

# get configuration of nn-framework
message("NNINTERP=$ENV{NNINTERP}")
string(TOLOWER $ENV{NNINTERP} NNINTERP)
string(REPLACE "-" ";" CONF_NNINTERP ${NNINTERP})

# set NN_FRAMEWORK
list(GET CONF_NNINTERP 0 NN_FRAMEWORK)
# set NN_CONFIG
list(GET CONF_NNINTERP 1 NN_CONFIG)
# set NN_TARGET
list(LENGTH CONF_NNINTERP LEN_NNINTERP)
if(${LEN_NNINTERP} GREATER 2)
	list(SUBLIST CONF_NNINTERP 2 -1 NN_TARGET)
	list(JOIN NN_TARGET "-" NN_TARGET)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
	message("TARGET: Windows")
	set(NN_TARGET "windows-x86_64")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	message("TARGET: Linux")
	set(NN_TARGET "linux-x86_64")
#elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
else()
	message(FATAL_ERROR "unknown target")
endif()


if(${NN_FRAMEWORK} STREQUAL "libtorch")
	include(TorchInterp.cmake)
elseif(${NN_FRAMEWORK} STREQUAL "onnx")
	include(OnnxInterp.cmake)
elseif(${NN_FRAMEWORK} STREQUAL "tflite")
	include(TflInterp.cmake)
else()
	message(FATAL_ERROR "Unkown NN framework: ${NNINTERP}")
endif()

# add Nlohmann JSON
set(URL_NLOHMANN_JSON "https://github.com/nlohmann/json/releases/download/v3.7.3/include.zip")
set(NLOHMANN_JSON_ROOTDIR ${THIRD_PARTY}/nlohmann_json)

if(NOT EXISTS ${NLOHMANN_JSON_ROOTDIR})
        message("** Download nlohmann_json.")
        FetchContent_Declare(nlohmann_json
                URL ${URL_NLOHMANN_JSON}
                SOURCE_DIR ${NLOHMANN_JSON_ROOTDIR}
                )
        FetchContent_MakeAvailable(nlohmann_json)
endif()

include_directories(${NLOHMANN_JSON_ROOTDIR}/include)

# my own libraries
set(GETOPT 
	src/getopt/getopt.c
	src/getopt/getopt_long.c
	)
add_library(interp
	STATIC
	src/tiny_ml.cpp
	src/tensor_spec.cpp
	src/io_port.cpp
	src/nonmaxsuppression.cpp
	${GETOPT}
	)

# main
add_executable(nn_interp
	src/main.cpp
	${X_INTERP}
	)
target_link_libraries(nn_interp
	interp
	)

# installation
install(TARGETS nn_interp
	RUNTIME
	DESTINATION $ENV{MIX_APP_PATH}/priv
	)
install(TARGETS nn_interp
	RUNTIME
	DESTINATION ${CMAKE_SOURCE_DIR}/priv
	)

if(NNFW_DLLS)
	add_custom_command(TARGET nn_interp
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory $ENV{MIX_APP_PATH}/priv
		COMMAND ${CMAKE_COMMAND} -E copy_if_different ${NNFW_DLLS} $ENV{MIX_APP_PATH}/priv
		)
	add_custom_command(TARGET nn_interp
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/priv
		COMMAND ${CMAKE_COMMAND} -E copy_if_different ${NNFW_DLLS} ${CMAKE_SOURCE_DIR}/priv
		)
endif()
