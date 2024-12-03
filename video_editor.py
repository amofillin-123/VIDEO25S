import os
import random
import subprocess
import json
import tempfile
import numpy as np
from scenedetect import detect, ContentDetector
import shutil
import cv2

class VideoEditor:
    def __init__(self, input_path, output_path, target_duration=25):
        self.input_path = input_path
        self.output_path = output_path
        self.target_duration = target_duration
        self.temp_dir = tempfile.mkdtemp()
        self.progress_callback = None

    def _run_ffmpeg(self, command):
        """运行 ffmpeg 命令"""
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                print(f"FFmpeg 错误: {result.stderr}")
                return False
            return True
        except Exception as e:
            print(f"运行 FFmpeg 失败: {str(e)}")
            return False

    def _extract_video_segment(self, start, duration, output_path):
        """提取视频片段"""
        command = [
            'ffmpeg', '-y',
            '-i', self.input_path,
            '-ss', str(start),
            '-t', str(duration),
            '-map', '0:v:0',  # 只提取第一个视频流
            '-c:v', 'libx264',  # 使用 H.264 编码
            '-preset', 'fast',  # 快速编码
            '-crf', '23',  # 保持较好的质量
            output_path
        ]
        return self._run_ffmpeg(command)

    def _concat_videos(self, video_files, output_path):
        """合并视频文件"""
        list_file = os.path.join(self.temp_dir, 'file_list.txt')
        with open(list_file, 'w') as f:
            for video_file in video_files:
                f.write(f"file '{video_file}'\n")

        command = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file,
            '-c', 'copy',  # 直接复制，不重新编码
            output_path
        ]
        return self._run_ffmpeg(command)

    def _detect_scenes(self):
        """使用 PySceneDetect 检测场景"""
        try:
            # 使用内容检测器，阈值设置为较高以避免过度分割
            scenes = detect(self.input_path, ContentDetector(threshold=30))
            # 转换为时间戳列表
            scene_list = []
            for scene in scenes:
                start_time = float(scene[0].get_seconds())
                end_time = float(scene[1].get_seconds())
                duration = end_time - start_time
                # 过滤掉太短的场景（小于0.5秒）
                if duration >= 0.5:
                    scene_list.append((start_time, end_time))
            return scene_list
        except Exception as e:
            print(f"场景检测失败: {str(e)}")
            return None

    def _select_scenes(self, scenes, total_duration):
        """智能选择场景，优先选择有字幕的场景，同时保留开头和结尾"""
        if not scenes or len(scenes) < 4:  # 至少需要4个场景
            return scenes

        # 计算每个场景的时长并检测字幕
        scene_info = []
        print("\n检测场景字幕...")
        for i, (start, end) in enumerate(scenes):
            duration = end - start
            print(f"\n分析场景 {i+1}/{len(scenes)} ({start:.1f}s - {end:.1f}s):")
            subtitle_info = self._detect_scene_with_subtitle(start, end)
            # 计算场景得分：有字幕得3分，无字幕得1分
            score = 3 if subtitle_info['has_subtitle'] else 1
            scene_info.append({
                'duration': duration,
                'start': start,
                'end': end,
                'score': score,
                'has_subtitle': subtitle_info['has_subtitle'],
                'subtitle_text': subtitle_info['text'] if subtitle_info['has_subtitle'] else None
            })
            
            # 打印场景信息
            status = "有字幕" if subtitle_info['has_subtitle'] else "无字幕"
            if subtitle_info['has_subtitle']:
                print(f"状态: {status}, 文本: {subtitle_info['text']}")
            else:
                print(f"状态: {status}")
        
        # 保留开头的两个场景
        start_scenes = scene_info[:2]
        for scene in start_scenes:
            scene['score'] += 1  # 开头场景额外加1分
        start_duration = sum(scene['duration'] for scene in start_scenes)
        
        # 保留结尾的两个场景
        end_scenes = scene_info[-2:]
        for scene in end_scenes:
            scene['score'] += 1  # 结尾场景额外加1分
        end_duration = sum(scene['duration'] for scene in end_scenes)
        
        # 中间场景
        middle_scenes = scene_info[2:-2]
        
        # 计算中间部分需要的时长
        target_middle_duration = self.target_duration - (start_duration + end_duration)
        
        if target_middle_duration <= 0:
            # 如果开头和结尾已经超过目标时长，只保留它们的一部分
            selected_scenes = [(scene['start'], scene['end']) for scene in (start_scenes + end_scenes)]
            return selected_scenes[:int(self.target_duration)]
        
        # 按分数排序中间场景，分数相同时随机排序
        middle_scenes.sort(key=lambda x: (x['score'], random.random()), reverse=True)
        
        # 选择中间场景
        selected_middle_scenes = []
        current_duration = 0
        
        for scene in middle_scenes:
            if current_duration + scene['duration'] <= target_middle_duration:
                selected_middle_scenes.append((scene['start'], scene['end']))
                current_duration += scene['duration']
                if current_duration >= target_middle_duration:
                    break
        
        # 按时间顺序合并所有选中的场景
        selected_scenes = (
            [(scene['start'], scene['end']) for scene in start_scenes] +  # 开头场景
            sorted(selected_middle_scenes) +                              # 中间场景（保持时间顺序）
            [(scene['start'], scene['end']) for scene in end_scenes]     # 结尾场景
        )
        
        return selected_scenes

    def _extract_frame(self, time_point, output_path):
        """在指定时间点提取一帧"""
        command = [
            'ffmpeg', '-y',
            '-ss', str(time_point),
            '-i', self.input_path,
            '-vframes', '1',
            '-q:v', '2',
            output_path
        ]
        return self._run_ffmpeg(command)

    def detect_text_in_frame(self, frame):
        """
        使用OpenCV检测帧中是否包含文字
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 应用自适应阈值
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 分析轮廓以检测文本
        text_like_contours = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            area = cv2.contourArea(contour)
            
            # 文本通常具有特定的宽高比和面积
            if 0.1 < aspect_ratio < 15 and 50 < area < 10000:
                text_like_contours += 1
                
            if text_like_contours > 5:  # 如果检测到足够多的可能文本区域
                return True
                
        return False

    def has_text_in_video(self, video_path, sample_interval=1):
        """
        检查视频中是否包含文字
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 每秒采样一帧
        for i in range(0, total_frames, fps * sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
                
            if self.detect_text_in_frame(frame):
                cap.release()
                return True
                
        cap.release()
        return False

    def _detect_scene_with_subtitle(self, start_time, end_time):
        """检测场景是否包含字幕，返回字幕信息"""
        duration = end_time - start_time
        # 在场景中提取更多帧进行检查
        check_points = [
            start_time + (duration * ratio)
            for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ]
        
        detected_texts = []
        consecutive_frames_with_text = 0
        last_valid_text = None
        
        for time_point in check_points:
            frame_path = os.path.join(self.temp_dir, f"frame_{time_point}.jpg")
            if self._extract_frame(time_point, frame_path):
                frame = cv2.imread(frame_path)
                if self.detect_text_in_frame(frame):
                    current_text = "Detected text"
                    # 如果当前文本与上一个有效文本相似度高，增加连续计数
                    if last_valid_text is None or current_text == last_valid_text:
                        consecutive_frames_with_text += 1
                        last_valid_text = current_text
                    else:
                        consecutive_frames_with_text = 1
                        last_valid_text = current_text
                    detected_texts.append(current_text)
                else:
                    consecutive_frames_with_text = 0
                
                # 清理临时文件
                if os.path.exists(frame_path):
                    os.remove(frame_path)
                    
                # 如果连续3帧检测到相似的文本，确定是字幕场景
                if consecutive_frames_with_text >= 3:
                    break
        
        # 如果至少有3帧检测到文本，就认为场景中有字幕
        if len(detected_texts) >= 3:
            # 选择最常见的文本作为场景的字幕文本
            from collections import Counter
            text_counter = Counter(detected_texts)
            most_common_text = text_counter.most_common(1)[0][0]
            return {
                'has_subtitle': True,
                'text': most_common_text
            }
        return {
            'has_subtitle': False,
            'text': None
        }

    def process_video(self, progress_callback=None):
        """处理视频的主要方法"""
        self.progress_callback = progress_callback
        try:
            if self.progress_callback:
                self.progress_callback(0)  # 开始处理
                
            print(f"正在加载视频: {self.input_path}")
            
            # 获取视频信息
            probe_command = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=duration',
                '-of', 'json',
                self.input_path
            ]
            
            result = subprocess.run(probe_command, capture_output=True, text=True)
            video_info = json.loads(result.stdout)
            total_duration = float(video_info['streams'][0]['duration'])
            
            print(f"视频总时长: {total_duration}秒")

            if total_duration < self.target_duration:
                print("错误：视频时长不足25秒")
                return False

            # 提取音频（仅前25秒）
            print("提取音频...")
            temp_audio = os.path.join(self.temp_dir, "temp_audio.mp3")
            command = [
                'ffmpeg', '-y',
                '-i', self.input_path,
                '-t', str(self.target_duration),
                '-vn',
                '-acodec', 'libmp3lame',
                '-q:a', '0',  # 最高质量
                temp_audio
            ]
            if not self._run_ffmpeg(command):
                print("警告：提取音频失败")
                temp_audio = None

            # 检测场景
            print("检测场景变化...")
            if self.progress_callback:
                self.progress_callback(20)  # 20% 进度
            
            scenes = self._detect_scenes()
            if not scenes:
                print("场景检测失败，使用备用方案...")
                return False

            # 选择场景
            print("选择场景...")
            if self.progress_callback:
                self.progress_callback(40)  # 40% 进度
                
            selected_scenes = self._select_scenes(scenes, total_duration)

            # 提取选中的场景
            print("提取选中的场景...")
            if self.progress_callback:
                self.progress_callback(60)  # 60% 进度
            video_segments = []
            for i, (start, end) in enumerate(selected_scenes):
                segment_file = os.path.join(self.temp_dir, f"segment_{i}.mp4")
                if self._extract_video_segment(start, end - start, segment_file):
                    video_segments.append(segment_file)
                    print(f"提取场景: {start:.1f}s - {end:.1f}s {'(开头)' if i < 2 else '(结尾)' if i >= len(selected_scenes)-2 else '(中间)'}")

            if not video_segments:
                print("错误：无法提取有效场景")
                return False

            # 合并视频片段
            print("合并场景...")
            if self.progress_callback:
                self.progress_callback(80)  # 80% 进度
            temp_video = os.path.join(self.temp_dir, "temp_video.mp4")
            if not self._concat_videos(video_segments, temp_video):
                print("错误：合并视频片段失败")
                return False

            # 添加音频（如果有）
            if temp_audio:
                print("添加音频...")
                command = [
                    'ffmpeg', '-y',
                    '-i', temp_video,
                    '-i', temp_audio,
                    '-c:v', 'copy',  # 保持视频流不变
                    '-c:a', 'aac',
                    '-b:a', '320k',  # 使用最高音频比特率
                    '-shortest',  # 使用最短的流的长度
                    self.output_path
                ]
                if not self._run_ffmpeg(command):
                    print("错误：添加音频失败")
                    return False
            else:
                # 如果没有音频，直接复制视频
                shutil.copy2(temp_video, self.output_path)

            # 清理临时文件
            print("清理资源...")
            shutil.rmtree(self.temp_dir)
            
            print(f"处理完成，输出文件：{self.output_path}")
            if self.progress_callback:
                self.progress_callback(100)  # 完成
            return True

        except Exception as e:
            print(f"处理失败: {str(e)}")
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            return False

    def create_final_video(self):
        """为了保持兼容性的包装方法"""
        return self.process_video()
