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
        self.analysis_video_path = None
        self.max_analysis_width = 640  # 分析用的最大宽度
        self.sample_interval = 2  # 采样间隔（秒）

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

    def _select_scenes(self, scene_list):
        """选择要使用的场景"""
        if not scene_list:
            return []

        # 按时间排序场景
        sorted_scenes = sorted(scene_list, key=lambda x: x[0])
        
        # 选择场景
        selected_scenes = []
        total_duration = 0
        target_duration = self.target_duration
        
        # 确保至少选择一个场景
        if sorted_scenes:
            first_scene = sorted_scenes[0]
            duration = first_scene[1] - first_scene[0]
            selected_scenes.append((first_scene[0], first_scene[1]))
            total_duration += duration

        # 随机选择其他场景
        remaining_scenes = sorted_scenes[1:]
        random.shuffle(remaining_scenes)
        
        for scene in remaining_scenes:
            duration = scene[1] - scene[0]
            if total_duration + duration <= target_duration:
                selected_scenes.append((scene[0], scene[1]))
                total_duration += duration
            
            if total_duration >= target_duration:
                break

        # 按时间顺序排序选中的场景
        selected_scenes.sort()
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
        使用OpenCV检测帧中是否包含文字（优化版本）
        """
        # 缩小图像以加快处理
        height, width = frame.shape[:2]
        if width > 400:
            scale = 400 / width
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
            
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用MSER检测器
        mser = cv2.MSER_create()
        mser.setMinArea(100)
        mser.setMaxArea(2000)
        mser.setDelta(5)
        
        # 检测区域
        regions, _ = mser.detectRegions(gray)
        
        # 如果检测到足够多的区域，认为存在文字
        if len(regions) > 10:
            return True
            
        return False

    def has_text_in_video(self, video_path, sample_interval=3):
        """
        检查视频中是否包含文字（优化版本）
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 每3秒采样一帧
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

    def _create_analysis_version(self):
        """创建用于分析的低分辨率版本"""
        analysis_path = os.path.join(self.temp_dir, "analysis_version.mp4")
        
        # 获取原始视频信息
        probe = self._get_video_info(self.input_path)
        if not probe:
            return None
            
        # 计算新的分辨率
        width = int(probe.get('width', 1920))
        height = int(probe.get('height', 1080))
        if width > self.max_analysis_width:
            scale_factor = self.max_analysis_width / width
            new_width = self.max_analysis_width
            new_height = int(height * scale_factor)
        else:
            new_width = width
            new_height = height
            
        # 创建低分辨率版本，降低帧率以加快处理
        command = [
            "ffmpeg", "-i", self.input_path,
            "-vf", f"scale={new_width}:{new_height}",
            "-r", "10",  # 降低帧率到10fps
            "-c:v", "libx264", "-crf", "28",  # 使用较高压缩率
            "-y", analysis_path
        ]
        
        if self._run_ffmpeg(command):
            self.analysis_video_path = analysis_path
            return analysis_path
        return None

    def _get_video_info(self, video_path):
        """获取视频信息"""
        command = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            video_path
        ]
        
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        return {
                            'width': int(stream.get('width', 0)),
                            'height': int(stream.get('height', 0)),
                            'duration': float(stream.get('duration', 0))
                        }
        except Exception as e:
            print(f"获取视频信息失败: {str(e)}")
        return None

    def _create_video_from_scenes(self, scenes):
        """使用原始视频创建最终输出"""
        video_segments = []
        for i, (start, end) in enumerate(scenes):
            segment_file = os.path.join(self.temp_dir, f"segment_{i}.mp4")
            if self._extract_video_segment(start, end - start, segment_file):
                video_segments.append(segment_file)
                print(f"提取场景: {start:.1f}s - {end:.1f}s {'(开头)' if i < 2 else '(结尾)' if i >= len(scenes)-2 else '(中间)'}")

        if not video_segments:
            print("错误：无法提取有效场景")
            return False

        # 合并视频片段
        print("合并场景...")
        temp_video = os.path.join(self.temp_dir, "temp_video.mp4")
        if not self._concat_videos(video_segments, temp_video):
            print("错误：合并视频片段失败")
            return False

        # 添加音频（如果有）
        print("添加音频...")
        command = [
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', self.input_path,
            '-c:v', 'copy',  # 保持视频流不变
            '-c:a', 'aac',
            '-b:a', '320k',  # 使用最高音频比特率
            '-shortest',  # 使用最短的流的长度
            self.output_path
        ]
        if not self._run_ffmpeg(command):
            print("错误：添加音频失败")
            return False

        return True

    def create_final_video(self):
        """创建最终视频的主方法"""
        try:
            # 1. 创建分析用的低分辨率版本
            print("创建分析版本...")
            if not self._create_analysis_version():
                print("创建分析版本失败")
                return False
                
            # 2. 在低分辨率版本上进行分析
            print("检测场景...")
            scenes = detect(self.analysis_video_path, 
                          ContentDetector(threshold=30.0, min_scene_len=15))  # 调整场景检测参数
            if not scenes:
                print("场景检测失败")
                return False
                
            # 3. 处理场景信息
            print("分析场景...")
            scene_list = []
            for scene in scenes:
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                
                # 使用低分辨率版本检测字幕
                subtitle_info = self._detect_scene_with_subtitle(start_time, end_time)
                scene_list.append({
                    'start': start_time,
                    'end': end_time,
                    'has_subtitle': subtitle_info['has_subtitle'],
                    'subtitle_text': subtitle_info['text']
                })
                
            # 4. 选择场景
            print("选择场景...")
            selected_scenes = self._select_scenes(scene_list)
            if not selected_scenes:
                print("场景选择失败")
                return False
                
            # 5. 使用原始高质量视频进行剪辑
            print("创建最终视频...")
            success = self._create_video_from_scenes(selected_scenes)
            
            if success:
                print("处理完成！")
            return success
            
        except Exception as e:
            print(f"处理失败: {str(e)}")
            return False
            
        finally:
            # 清理临时文件
            if self.analysis_video_path and os.path.exists(self.analysis_video_path):
                os.remove(self.analysis_video_path)
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)

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
                
            selected_scenes = self._select_scenes(scenes)

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
