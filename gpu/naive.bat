setlocal enabledelayedexpansion

set VALUE=1
set COUNT=10  :: How many times to run

for /L %%i in (1,1,%COUNT%) do (
    raytracer.exe !VALUE! 1 > metrics/naive/data_!VALUE!.txt
    set /a VALUE=!VALUE!*2
)

endlocal
