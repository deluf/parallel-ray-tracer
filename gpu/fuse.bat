setlocal enabledelayedexpansion

set VALUE=1
set COUNT=5 :: How many times to run

for /L %%i in (1,1,%COUNT%) do (
    raytracer.exe !VALUE! !VALUE! > metrics/fuse/data_!VALUE!x!VALUE!.txt
    set /a VALUE=!VALUE!*2
)

endlocal
