@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 exit /b 1

set GR_CONDA_PREFIX=D:\mysoft2\miniconda3\envs\gr-lora

cd /d "D:\Desktop\proj\gr-lora_sdr"
if exist build rmdir /s /q build
mkdir build
cd build

cmake .. -G "NMake Makefiles" ^
    -DCMAKE_INSTALL_PREFIX="%GR_CONDA_PREFIX%\Library" ^
    -DCMAKE_PREFIX_PATH="%GR_CONDA_PREFIX%\Library" ^
    -DPYTHON_EXECUTABLE="%GR_CONDA_PREFIX%\python.exe" ^
    -DGR_PYTHON_DIR="%GR_CONDA_PREFIX%\Lib\site-packages" ^
    -DCMAKE_CXX_FLAGS="/d2FH4- /permissive- /Zc:__cplusplus" ^
    -DENABLE_DOXYGEN=OFF ^
    -DENABLE_TESTING=OFF
if errorlevel 1 exit /b 1

nmake
if errorlevel 1 exit /b 1

nmake install
if errorlevel 1 exit /b 1

echo Build completed successfully!
