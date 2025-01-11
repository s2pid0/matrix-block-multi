# Указываем минимальную версию CMake
cmake_minimum_required(VERSION 3.10)

# Устанавливаем имя проекта
project(MatrixMultiply)

# Указываем стандарт C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Добавляем исполняемый файл
add_executable(matrix_multiply main.cpp)

# Указываем, что проект будет компилироваться с использованием MinGW
if(MINGW)
    message(STATUS "Using MinGW as the compiler")
endif()

