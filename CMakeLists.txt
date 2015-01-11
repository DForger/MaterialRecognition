project(MaterialRecognition)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

#file(MAKE_DIRECTORY "./Debug/")
#file(MAKE_DIRECTORY "./Release/")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

########### set Debug or Release Mode ##########
set(DEBUG_MODE ON) #flag

if(DEBUG_MODE)
    add_definitions(-DDEBUG_MODE)
    message("Debug Mode")
    set(CMAKE_BUILD_TYPE DEBUG)
#    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/Debug)
else()
    message("Release Mode")
    set(CMAKE_BUILD_TYPE RELEASE)
#    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/Release)
endif()

########## set up qt #########
set(USE_QT5 OFF) #flag

if(USE_QT5)
    set(CMAKE_AUTOMOC ON)
    find_package(Qt5Widgets)
else()
    find_package(Qt4 REQUIRED)
endif()

include_directories(/usr/include/libxml2)

############### set up OpenGM ########
#include_directories(/usr/local/include/opengm/externals/)
#link_directories(/usr/local/lib/opengm/externals)
#include_directories(/usr/local/include/opengm/externals/qpbo)

############## set up OpenCV ########
find_package(OpenCV REQUIRED)
add_definitions(-DUSE_OPENCV_3_0=1)

############## set up Eigen #########
#find_package(Eigen3 REQUIRED)
#include_directories(EIGEN3_INCLUDE_DIR)
include_directories(/usr/include/eigen3)

############# set up PCL #############
#find_package(PCL 1.7 REQUIRED)
#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

############## set up OpenNI ###########
#SET(OPENNI2_DIR ~/App_Home/OpenNI-Linux-x64-2.2)
#INCLUDE_DIRECTORIES(${OPENNI2_DIR}/Include)
#LINK_DIRECTORIES(${OPENNI2_DIR}/Redist/)
#SET(OPENNI2_LIBRARIES ${OPENNI2_DIR}/Redist/libOpenNI2.so)

##############set up VLFeat ###############
set (VLFEAT_DIR /home/shenyunjun/App_Home/vlfeat-master)
include_directories(${VLFEAT_DIR}/)
link_directories(${VLFEAT_DIR}/bin/glnxa64/)
set(VLFEAT_LIBS ${VLFEAT_DIR}/bin/glnxa64/libvl.so)


############## include source code ###########
set(${PROJECT_NAME}_SOURCES
    main.cpp
#    MaterialClassifier/multimaterial_classifier.cpp
    MaterialClassifier/material_classifier.cpp
    Preprocessing/SLICSuperpixel.cpp
    Miscellaneous/data_io.h
    Miscellaneous/interpolation.h
    Miscellaneous/visualization.h
    Miscellaneous/miscellaneous.cpp
)

set(${PROJECT_NAME}_HEADERS
#    MaterialClassifier/multimaterial_classifier.h
    MaterialClassifier/material_classifier.h
    Miscellaneous/miscellaneous.h
    Preprocessing/SLICSuperpixel.h
)

set(${PROJECT_NAME}_FORMS
#    GUI/mainwindow.ui
)

if(USE_QT5)
    qt5_wrap_ui(${PROJECT_NAME}_FORM_HEADERS
        ${${PROJECT_NAME}_FORMS}
    )

    add_definitions(-DQT_NO_KEYWORDS)

    add_executable(${PROJECT_NAME}
        ${${PROJECT_NAME}_SOURCES}
        ${${PROJECT_NAME}_FORM_HEADERS}
    )

    target_link_libraries(${PROJECT_NAME}
        Qt5::Widgets
        ${OpenCV_LIBS}
#        ${PCL_LIBRARIES}
#        ${OPENNI2_LIBRARIES}
#        opengm-external-ad3-shared
#        opengm-external-qpbo-shared
        tbb
#        armadillo
#        mlpack
    )
else()
    qt4_wrap_cpp(${PROJECT_NAME}_HEADERS_MOC
        ${${PROJECT_NAME}_HEADERS}
        )
    qt4_wrap_ui(${PROJECT_NAME}_FORMS_HEADERS
        ${${PROJECT_NAME}_FORMS}
        )

    include(${QT_USE_FILE})
    add_definitions(${QT_DEFINITIONS})
        add_definitions(-DQT_NO_KEYWORDS)

    add_executable(${PROJECT_NAME}
        ${${PROJECT_NAME}_SOURCES}
        ${${PROJECT_NAME}_HEADERS_MOC}
        ${${PROJECT_NAME}_FORMS_HEADERS}
    )

    target_link_libraries(${PROJECT_NAME}
        ${QT_LIBRARIES}
        ${OpenCV_LIBS}
        ${VLFEAT_LIBS}
#        ${PCL_LIBRARIES}
#        ${OPENNI2_LIBRARIES}
#        opengm-external-ad3-shared
#        opengm-external-mrf-shared
#        opengm-external-qpbo-shared
        tbb
#        armadillo
#        mlpack
    )
endif()


#include_directories(${CMAKE_CURRENT_BINARY_DIR})

