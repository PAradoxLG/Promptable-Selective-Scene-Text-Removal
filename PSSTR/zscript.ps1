# 激活 conda 环境
conda activate str

# 获取当前日期时间，并按 "yyyyMMdd_HHmmss" 格式进行格式化
$currentDateTime = Get-Date -Format "yyyyMMdd_HHmmss"

# 定义训练模式和日志目录
$train_mode = '_point_and_box_for_segment'
$logdir = 'C:\Users\gran\Hi-SAM\logs'

# 拼接日志文件名
$log_file = "${currentDateTime}${train_mode}.log"

# 拼接日志文件的完整路径
$log_path = Join-Path -Path $logdir -ChildPath $log_file

$env:PYTHONUNBUFFERED = "1"
# 运行 Python 脚本，并将输出和错误输出重定向到日志文件
python train.py --promptable *> $log_path
