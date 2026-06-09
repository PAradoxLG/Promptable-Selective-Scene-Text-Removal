@echo off
pushd %~dp0
powershell.exe -command "Start-Process -WindowStyle hidden -FilePath 'C:\Users\lg\Hi-SAM\zscript.bat' -WorkingDirectory 'C:\Users\lg\Hi-SAM'"
popd

@REM @echo off
@REM pushd %~dp0
@REM powershell.exe -command "& {Register-ScheduledTask -TaskName 'xxxRunProject' -Action (New-ScheduledTaskAction -WorkingDirectory C:\Users\lg\Hi-SAM -Execute 'powershell' -Argument '-NoProfile Start-Process -WindowStyle hidden -FilePath zscript.bat'); Start-ScheduledTask -TaskName 'xxxRunProject'; Unregister-ScheduledTask -TaskName 'xxxRunProject' -Confirm:$false}"
@REM popd