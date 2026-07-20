@echo off
set ENGINE=%1
if "%ENGINE%"=="" set ENGINE=build\scarlet.exe
%ENGINE% < scripts\smoke_leela_windows.uci
