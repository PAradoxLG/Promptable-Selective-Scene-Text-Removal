@REM only segment text
python .\demo_promptable.py --promptable --input datasets\FlickrST\train\image\FIS_1.jpg

@REM erase text
python .\demo_erase.py --promptable --erase_mode --input datasets\FlickrST\train\image\FIS_1.jpg