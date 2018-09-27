@echo off
FOR %%I IN (*.json) DO @python "D:\Program Files\Anaconda3_64\Scripts\labelme_json_to_dataset.exe" "%%I">nul 2>nul&&echo %%I converted.||(echo %%I Failed.&@python "D:\Program Files\Anaconda3_64\Scripts\labelme_json_to_dataset.exe" "%%I")
echo Finished!
pause