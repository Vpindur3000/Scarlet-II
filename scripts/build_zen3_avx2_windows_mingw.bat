@echo off
setlocal
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DSCARLET_ZEN3_AVX2=ON
cmake --build build -j
if errorlevel 1 exit /b 1
echo Built: build\scarlet.exe and build\scarlet-perft.exe
