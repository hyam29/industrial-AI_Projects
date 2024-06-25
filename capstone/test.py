# detect_video.py 파일에서 동영상파일이 만약 안열릴 때(콘솔에 비디오를 못 연다는 오류 메세지가 뜰 것임)
# 해당 영상을 이 코드를 실행해서 새로 추출 후 detect_video 실행

import subprocess
import os

input_path = r'C:\Users\User\Desktop\capstone\finalProject\datasets\VIS_Onboard\Videos\MVI_0792_VIS_OB.avi'
output_path = r'C:\Users\User\Desktop\capstone\finalProject\datasets\VIS_Onboard\Videos\MVI_0792_VIS_OB_converted.mp4'
ffmpeg_path = r'C:\utility\ffmpeg-2024-05-29-git-fa3b153cb1-full_build\bin\ffmpeg.exe'  # 여기에 ffmpeg.exe 파일의 실제 경로를 입력하세요

# FFmpeg를 사용하여 비디오 파일 변환
subprocess.run([ffmpeg_path, '-i', input_path, '-c:v', 'libx264', '-c:a', 'aac', output_path])

# 변환된 파일을 확인
if os.path.exists(output_path):
    print("File converted successfully.")
else:
    print("File conversion failed.")
