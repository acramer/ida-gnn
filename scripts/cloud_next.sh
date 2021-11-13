kill -9 $(ps -fu $USER | grep 'python main.py' | grep -v grep | awk '{print $2}')
