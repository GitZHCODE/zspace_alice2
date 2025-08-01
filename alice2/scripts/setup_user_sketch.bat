@echo off
echo ========================================
echo Alice2 User Sketch Setup
echo ========================================
echo.

:: Check if we're in the right directory
if not exist "userSrc" (
    echo ERROR: userSrc directory not found!
    echo Please run this script from the alice2 root directory.
    pause
    exit /b 1
)

:: Get sketch name from user
set /p SKETCH_NAME="Enter your sketch name (e.g., my_sketch): "
if "%SKETCH_NAME%"=="" (
    echo ERROR: Sketch name cannot be empty!
    pause
    exit /b 1
)

:: Create the sketch filename
set SKETCH_FILE=userSrc\%SKETCH_NAME%.cpp

:: Check if file already exists
if exist "%SKETCH_FILE%" (
    echo WARNING: %SKETCH_FILE% already exists!
    set /p OVERWRITE="Overwrite existing file? (y/N): "
    if /i not "%OVERWRITE%"=="y" (
        echo Cancelled.
        pause
        exit /b 0
    )
)

:: Create the sketch file from template
echo Creating %SKETCH_FILE%...
(
echo // %SKETCH_NAME%.cpp - Custom Alice2 Sketch
echo // Generated by Alice2 User Sketch Setup
echo.
echo #include "sketch_base.h"
echo #include "../include/alice2/Renderer.h"
echo #include "../include/alice2/Math.h"
echo.
echo class %SKETCH_NAME% : public SketchBase {
echo public:
echo     void setup^(^) override {
echo         // Initialize your sketch here
echo         // This is called once when the sketch starts
echo     }
echo.
echo     void update^(^) override {
echo         // Update logic here
echo         // This is called every frame before drawing
echo     }
echo.
echo     void draw^(^) override {
echo         // Rendering logic here
echo         // Use renderer.drawPoint^(^), renderer.drawLine^(^), etc.
echo         
echo         // Example: Draw a simple point at origin
echo         renderer.drawPoint^(Vector3^(0, 0, 0^), Color^(1, 1, 1^), 5.0f^);
echo     }
echo.
echo     void onKeyPress^(int key^) override {
echo         // Handle keyboard input
echo         // Example: Press 'R' to reset
echo         if ^(key == 'R'^) {
echo             // Reset logic here
echo         }
echo     }
echo.
echo     void onMouseMove^(double x, double y^) override {
echo         // Handle mouse movement
echo         // x, y are screen coordinates
echo     }
echo };
echo.
echo // Register the sketch with the system
echo REGISTER_SKETCH^(%SKETCH_NAME%^);
) > "%SKETCH_FILE%"

echo.
echo SUCCESS: Created %SKETCH_FILE%
echo.
echo Next steps:
echo 1. Edit %SKETCH_FILE% to implement your custom 3D content
echo 2. Build the project using build.bat or Visual Studio
echo 3. Run alice2.exe to see your sketch in action
echo.
echo Tips:
echo - Check userSrc\examples\ for more complex examples
echo - See userSrc\README.md for detailed documentation
echo - Use the alice2 math library for 3D calculations
echo.
pause
