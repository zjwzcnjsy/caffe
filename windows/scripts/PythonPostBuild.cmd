set SOLUTION_DIR=%~1%
set OUTPUT_DIR=%~2%

echo PythonPostBuild.cmd : copy python generated scripts to output.

copy /y "%SOLUTION_DIR%..\python\caffe\*.py" "%OUTPUT_DIR%pycaffe\caffe"
copy /y "%SOLUTION_DIR%..\python\caffe\utils\*.py" "%OUTPUT_DIR%pycaffe\caffe\utils"
copy /y "%SOLUTION_DIR%..\python\caffe\imagenet\ilsvrc_2012_mean.npy" "%OUTPUT_DIR%pycaffe\caffe\imagenet"
copy /y "%SOLUTION_DIR%..\python\caffe\model\*.py" "%OUTPUT_DIR%pycaffe\caffe\model"
move /y "%OUTPUT_DIR%_caffe.*" "%OUTPUT_DIR%pycaffe\caffe"

echo copy "%OUTPUT_DIR%*.dll" "%OUTPUT_DIR%pycaffe\caffe"
copy /y "%OUTPUT_DIR%*.dll" "%OUTPUT_DIR%pycaffe\caffe"