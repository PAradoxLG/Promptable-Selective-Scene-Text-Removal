# 保存当前工作目录
$initialDir = Get-Location
# 切换到当前脚本所在目录
Set-Location -Path $PSScriptRoot

# settings
$wd="C:\Users\gran\Hi-SAM"
$cmd="zscript.bat"
$redirect_std_out="nohup.out.log"
$redirect_std_err="nohup.err.log"

# running
$taskname="NohupTask-$(Get-Random)"
Register-ScheduledTask -TaskName "$taskname" -Action (New-ScheduledTaskAction -WorkingDirectory $wd -Execute "powershell" -Argument "-NoProfile ""Start-Process -WindowStyle hidden $cmd""")
Start-ScheduledTask -TaskName "$taskname"
Unregister-ScheduledTask -TaskName "$taskname" -Confirm:$false

# 恢复到之前的工作目录
Set-Location -Path $initialDir