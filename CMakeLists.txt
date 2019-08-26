cmake_minimum_required(VERSION 3.1)

project(model_generator)

# Flags
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -ltbb")

# On inclut notre bibliothèque dans le processus de CMake
add_subdirectory(libs)

# Crée des variables avec les fichiers à compiler
set(SRCS
  main.cpp
  # miscellaneous/main.cpp
  # miscellaneous/video_test.cpp
  # miscellaneous/persistent_world.cpp
  )

# Notre exécutable
add_executable(${PROJECT_NAME} ${SRCS})

########################### PCL Library ##########################
find_package (PCL 1.9 REQUIRED)

if (PCL_FOUND)
  # Une fois la dépendance trouvée, nous l'incluons au projet
  target_include_directories(${PROJECT_NAME} PUBLIC ${PCL_INCLUDE_DIRS})
  target_link_libraries (${PROJECT_NAME} ${PCL_LIBRARIES})
else ()
  # Sinon, nous affichons un message
  message(FATAL_ERROR "PCL not found")
endif ()

######################### OpenCV Library #########################

set(OpenCV_DIR "/home/elias-hanna/Documents/projects/opencv.OSID_ubuntu-16.04.installtree/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

########################### Kinect2 ##############################

FIND_LIBRARY(freenect2_LIBRARY freenect2
    PATHS /home/elias-hanna/freenect2/lib
    NO_DEFAULT_PATH
)
SET(freenect2_LIBRARIES ${freenect2_LIBRARY} pthread)
FIND_PATH(freenect2_INCLUDE_DIR libfreenect2/libfreenect2.hpp
    PATHS /home/elias-hanna/freenect2/include
    NO_DEFAULT_PATH
)
SET(freenect2_INCLUDE_DIRS ${freenect2_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC ${freenect2_INCLUDE_DIRS})
target_link_libraries (${PROJECT_NAME} ${freenect2_LIBRARIES})
