from video_editor import VideoEditor
import os

input_path = "test.mp4"
output_path = "output_test.mp4"

# 确保使用绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_dir, input_path)
output_path = os.path.join(current_dir, output_path)

editor = VideoEditor(input_path, output_path)
editor.process_video()
