@echo off
setlocal

echo ========================================
echo Alice2 3D Viewer - Run Script
echo ========================================
echo.

:: Default build folder
set "BUILD_DIR=build"

:: Optional CUDA mode
if /I "%~1"=="cuda" (
    set "BUILD_DIR=build_cuda"
)

set "EXE_PATH=%BUILD_DIR%\bin\Release\alice2.exe"
set "DLL_GLEW=%BUILD_DIR%\bin\Release\glew32.dll"
set "DLL_GLFW=%BUILD_DIR%\bin\Release\glfw3.dll"

:: Check if executable exists
if not exist "%EXE_PATH%" (
    echo ERROR: alice2.exe not found!
    echo Please build the project first using build.bat or build.bat cuda.
    pause
    exit /b 1
)

:: Check for required DLLs
echo Checking for required DLLs...
if not exist "%DLL_GLEW%" (
    echo WARNING: glew32.dll not found in %BUILD_DIR%\bin\Release. The program may not run correctly.
)
if not exist "%DLL_GLFW%" (
    echo WARNING: glfw3.dll not found in %BUILD_DIR%\bin\Release. The program may not run correctly.
)

echo.
echo Launching Alice2 3D Viewer...
cd /d "%BUILD_DIR%\bin\Release"
start alice2.exe
cd /d "%~dp0"
echo Alice2 launched successfully!
echo.

endlocal
