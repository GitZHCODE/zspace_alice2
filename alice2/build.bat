@echo off
echo ========================================
echo Alice2 3D Viewer - Build Script
echo ========================================
echo.

:: Check if we're in the right directory
if not exist "CMakeLists.txt" (
    echo ERROR: CMakeLists.txt not found!
    echo Please run this script from the alice2 root directory.
    pause
    exit /b 1
)

:: Configure with CMake
echo [1/3] Configuring project with CMake...
cmake . -B build -G "Visual Studio 17 2022" -A x64
if errorlevel 1 (
    echo ERROR: CMake configuration failed!
    pause
    exit /b 1
)

:: Build the project
echo.
echo [2/3] Building project (Release configuration)...
cmake --build build --config Release
if errorlevel 1 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

:: Build complete
echo.
echo [3/3] Build complete!
echo.
echo Executable location: build\bin\Release\alice2.exe
echo DLLs copied automatically to executable directory.
echo.
echo To create a distribution package, run:
echo   cmake --install build --prefix dist
echo.
echo Build successful! You can now run the alice2 viewer.
