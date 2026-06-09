@echo off

:: 激活 conda 环境
call conda activate str

for /f "tokens=2 delims==" %%I in ('"wmic os get localdatetime /value"') do set datetime=%%I
set year=%datetime:~0,4%
set month=%datetime:~4,2%
set day=%datetime:~6,2%
set hour=%datetime:~8,2%
set min=%datetime:~10,2%
set sec=%datetime:~12,2%
set currentDateTime=%year%%month%%day%_%hour%%min%%sec%

:: 定义训练模式和日志目录
@REM set train_mode=_point_and_box_and_mask_for_erase
@REM set train_mode=_text_for_segment
set train_mode=_resume_text_and_point_and_box_and_mask_for_erase
@REM set train_mode=_text_for_test
@REM set logdir=C:\Users\gran\Hi-SAM\logs
set logdir=C:\Users\lg\Hi-SAM\logs

:: 拼接日志文件名
set log_file=%currentDateTime%%train_mode%.log

:: 拼接日志文件的完整路径
set log_path=%logdir%\%log_file%

set PYTHONUNBUFFERED=1
:: 运行 Python 脚本，并将输出和错误输出重定向到日志文件
@REM python train.py --promptable > %log_path% 2>&1
@REM python train.py --promptable --word_prompt --device cuda:2 --valid_period 50 --batch_size_train 4 --max_epoch_num 200 --lr_drop_epoch 80 > %log_path% 2>&1
@REM python train.py --promptable --word_prompt --device cuda:2 --valid_period 50 --batch_size_train 4 --max_epoch_num 200 --lr_drop_epoch 100 --unimask_decoder_weight C:\Users\lg\Hi-SAM\work_dirs\2025-03-10__182437\199.pth > %log_path% 2>&1

@REM python train.py --promptable --erase_mode --unimask_decoder_weight C:/Users/gran/Hi-SAM/work_dirs/2025-01-06__220108/176.pth --batch_size_train 4 > %log_path% 2>&1
@REM python train.py --promptable --word_prompt --erase_mode --unimask_decoder_weight C:/Users/gran/Hi-SAM/work_dirs/2025-01-06__220108/176.pth --batch_size_train 4 > %log_path% 2>&1
python train.py --promptable --word_prompt --erase_mode --device cuda:2 --valid_period 10 --lr 1e-6 --batch_size_train 2 --batch_size_valid 1 --max_epoch_num 200 --lr_drop_epoch 50 --unimask_decoder_weight C:\Users\lg\Hi-SAM\work_dirs\2025-03-15__185422\199.pth --erase_decoder_weight C:\Users\lg\Hi-SAM\work_dirs\2025-08-18__093311\best.pth > %log_path% 2>&1