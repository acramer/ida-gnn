kill -9 $(ps -fu $USER | grep 'cloud_script' | grep -v grep | awk '{print $2}')
kill -9 $(ps -fu $USER | grep 'auto_run' | grep -v grep | awk '{print $2}')
kill -9 $(ps -fu $USER | grep 'python main.py' | grep -v grep | awk '{print $2}')
