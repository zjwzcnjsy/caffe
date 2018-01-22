set DEPENDECY_PATH=%~1%
set PTTHON_HOME=%~2%
set OUTPUT_DIR=%~3%

echo BinplaceDependencies : Copy *.dll to output.
copy /y "%DEPENDECY_PATH%\*.dll" "%OUTPUT_DIR%"

echo BinplaceDependencies : Copy python36.dll to output.
copy /y "%PTTHON_HOME%\python36.dll" "%OUTPUT_DIR%"