@echo off
rem ==============================================================================
rem SMARTCACHE AI — Native x86-64 Assembly Build Script
rem 
rem Assembles smartcache_engine.asm with Microsoft Macro Assembler (ml64.exe)
rem and links into smartcache_asm.dll.
rem ==============================================================================

setlocal enabledelayedexpansion

pushd "%~dp0"

set "ML64_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\ml64.exe"
set "LINK_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\link.exe"

if not exist "%ML64_PATH%" (
    echo [ERROR] ml64.exe not found at "%ML64_PATH%"
    popd
    exit /b 1
)

if not exist "%LINK_PATH%" (
    echo [ERROR] link.exe not found at "%LINK_PATH%"
    popd
    exit /b 1
)

echo [1/2] Assembling smartcache_engine.asm with ml64...
"%ML64_PATH%" /c /nologo /Fosmartcache_engine.obj smartcache_engine.asm
if errorlevel 1 (
    echo [ERROR] Assembly compilation failed.
    exit /b 1
)

echo [2/2] Linking smartcache_asm.dll...
"%LINK_PATH%" /nologo /DLL /NOENTRY ^
    /EXPORT:smartcache_init ^
    /EXPORT:smartcache_reset ^
    /EXPORT:smartcache_access ^
    /EXPORT:smartcache_choose_victim ^
    /EXPORT:smartcache_get_weights ^
    /EXPORT:smartcache_run_trace ^
    smartcache_engine.obj ^
    /OUT:smartcache_asm.dll

if errorlevel 1 (
    echo [ERROR] Linking failed.
    exit /b 1
)

echo.
echo [SUCCESS] Built smartcache_asm.dll successfully!
popd
endlocal
