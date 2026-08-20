@echo off
setlocal

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 exit /b 1

set "RFSR_RADIOCONDA_PREFIX=D:\mysoft2\radioconda"
set "RFSR_BUILD_DIR=D:\Desktop\proj\gr-lora_sdr\build-radioconda"
set "RFSR_CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

if not exist "%RFSR_RADIOCONDA_PREFIX%\Library\include\boost\version.hpp" (
    echo Missing RadioConda C++ headers: libboost-devel 1.86.
    echo Install that development package before rebuilding gr-lora_sdr.
    exit /b 2
)

set "CONDA_PREFIX=%RFSR_RADIOCONDA_PREFIX%"
set "PATH=%RFSR_RADIOCONDA_PREFIX%;%RFSR_RADIOCONDA_PREFIX%\Library\bin;%RFSR_RADIOCONDA_PREFIX%\Scripts;%PATH%"

if not exist "%RFSR_BUILD_DIR%" mkdir "%RFSR_BUILD_DIR%"
cd /d "%RFSR_BUILD_DIR%"

"%RFSR_CMAKE%" --fresh .. -G "NMake Makefiles" ^
    -DCMAKE_INSTALL_PREFIX="%RFSR_RADIOCONDA_PREFIX%\Library" ^
    -DCMAKE_PREFIX_PATH="%RFSR_RADIOCONDA_PREFIX%\Library" ^
    -DPYTHON_EXECUTABLE="%RFSR_RADIOCONDA_PREFIX%\python.exe" ^
    -DGR_PYTHON_DIR="%RFSR_RADIOCONDA_PREFIX%\Lib\site-packages" ^
    -DCMAKE_CXX_FLAGS="/d2FH4- /permissive- /Zc:__cplusplus" ^
    -DENABLE_DOXYGEN=OFF ^
    -DENABLE_TESTING=OFF
if errorlevel 1 exit /b 1

nmake
if errorlevel 1 exit /b 1

echo RadioConda build completed successfully.
echo The build was not installed into %RFSR_RADIOCONDA_PREFIX%.
endlocal
