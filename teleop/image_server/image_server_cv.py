import cv2
import zmq
import time
import struct
from collections import deque
import numpy as np
import pyrealsense2 as rs


class RealSenseCamera(object):
    def __init__(self, img_shape, fps, serial_number=None, enable_depth=False) -> None:
        """
        RealSense 相机封装类。
        Args:
            img_shape (list or tuple): 图像尺寸 [height, width]。
            fps (int): 帧率。
            serial_number (str or None): RealSense 设备序列号；若为 None，则使用默认设备。
            enable_depth (bool): 是否启用深度流。
        """
        self.img_shape = img_shape  # [height, width]
        self.fps = fps
        self.serial_number = serial_number
        self.enable_depth = enable_depth

        # 用于对齐深度到彩色帧
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        # 初始化 RealSense 管道
        self.init_realsense()

    def init_realsense(self):
        """
        配置并启动 RealSense 管道，设置彩色和深度流（可选）。
        """
        self.pipeline = rs.pipeline()
        config = rs.config()
        # 如果提供序列号，绑定到特定设备
        if self.serial_number is not None:
            config.enable_device(self.serial_number)

        # 启用彩色流，指定宽高和帧率，格式 BGR8
        config.enable_stream(rs.stream.color, self.img_shape[1], self.img_shape[0], rs.format.bgr8, self.fps)

        # 若启用深度，配置深度流
        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.img_shape[1], self.img_shape[0], rs.format.z16, self.fps)

        # 启动管道并获取 profile
        profile = self.pipeline.start(config)
        self._device = profile.get_device()
        if self._device is None:
            print('[Image Server] pipe_profile.get_device() is None .')
        # 若启用深度，获取深度刻度
        if self.enable_depth:
            assert self._device is not None
            depth_sensor = self._device.first_depth_sensor()
            self.g_depth_scale = depth_sensor.get_depth_scale()

        # 获取彩色流内参，用于后续对齐或处理
        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    def get_frame(self):
        """
        获取一帧对齐后的彩色和深度图。
        Returns:
            tuple: (color_image, depth_image) 或仅 color_image (若未启用深度)。
            若无法获取彩色帧则返回 None。
        """
        # 等待一组帧（包含彩色和深度）
        frames = self.pipeline.wait_for_frames()
        # 对齐深度到彩色视角
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()

        # 若启用深度，获取深度帧
        if self.enable_depth:
            depth_frame = aligned_frames.get_depth_frame()

        # 未获取到彩色帧则返回 None
        if not color_frame:
            return None

        # 转换为 numpy 数组
        color_image = np.asanyarray(color_frame.get_data())
        # 若需要 RGB，可做 cvtColor，但此处保持 BGR 用于 OpenCV
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        depth_image = np.asanyarray(depth_frame.get_data()) if self.enable_depth else None
        return color_image, depth_image

    def release(self):
        """
        停止 RealSense 管道。
        """
        self.pipeline.stop()


class OpenCVCamera():
    def __init__(self, device_id, img_shape, fps):
        """
        OpenCV VideoCapture 封装类，用于 USB 摄像头或 V4L2 设备。
        Args:
            device_id: 设备标识，例如整数索引或设备路径 '/dev/video0'。
            img_shape (list or tuple): 图像尺寸 [height, width]。
            fps (int): 期望的帧率。
        """
        self.id = device_id
        self.fps = fps
        self.img_shape = img_shape  # [height, width]
        # 使用 V4L2 后端打开设备
        self.cap = cv2.VideoCapture(self.id, cv2.CAP_V4L2)
        # 设置编码格式 MJPG，可提高性能并降低带宽
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_shape[0])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.img_shape[1])
        # 设置帧率
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # 测试能否读取一帧
        if not self._can_read_frame():
            print(f"[Image Server] Camera {self.id} Error: Failed to initialize the camera or read frames. Exiting...")
            self.release()

    def _can_read_frame(self):
        """
        测试读取一帧是否成功。
        Returns:
            bool: True 表示可读。
        """
        success, _ = self.cap.read()
        return success

    def release(self):
        """
        释放 VideoCapture 资源。
        """
        self.cap.release()

    def get_frame(self):
        """
        获取一帧图像。
        Returns:
            numpy.ndarray 或 None: BGR 图像；若读取失败则返回 None。
        """
        ret, color_image = self.cap.read()
        if not ret:
            return None
        return color_image


class ImageServer:
    def __init__(self, config, port=5555, Unit_Test=False):
        """
        图像服务器，采集多个摄像头画面后通过 ZeroMQ 发布。
        Args:
            config (dict): 配置，包括 fps、摄像头类型、分辨率、设备列表等。
            port (int): ZeroMQ 发布端口。
            Unit_Test (bool): 单元测试模式，启用性能统计头部。
        Config 示例:
            config = {
                'fps': 30,
                'head_camera_type': 'opencv' 或 'realsense',
                'head_camera_image_shape': [480, 1280],
                'head_camera_id_numbers': [0] 或 ['serial1', ...],
                'wrist_camera_type': 'opencv' 或 'realsense',
                'wrist_camera_image_shape': [480, 640],
                'wrist_camera_id_numbers': [0,1] 或 ['serial2', ...],
            }
        若不使用手腕摄像头，可忽略对应字段。
        """
        print(config)
        # 基本属性
        self.fps = config.get('fps', 30)
        self.head_camera_type = config.get('head_camera_type', 'opencv')
        self.head_image_shape = config.get('head_camera_image_shape', [480, 640])
        self.head_camera_id_numbers = config.get('head_camera_id_numbers', [0])

        self.wrist_camera_type = config.get('wrist_camera_type', None)
        self.wrist_image_shape = config.get('wrist_camera_image_shape', [480, 640])
        self.wrist_camera_id_numbers = config.get('wrist_camera_id_numbers', None)

        self.port = port
        self.Unit_Test = Unit_Test

        # 初始化头部摄像头列表
        self.head_cameras = []
        if self.head_camera_type == 'opencv':
            for device_id in self.head_camera_id_numbers:
                camera = OpenCVCamera(device_id=device_id, img_shape=self.head_image_shape, fps=self.fps)
                self.head_cameras.append(camera)
        elif self.head_camera_type == 'realsense':
            for serial_number in self.head_camera_id_numbers:
                camera = RealSenseCamera(img_shape=self.head_image_shape, fps=self.fps, serial_number=serial_number)
                self.head_cameras.append(camera)
        else:
            print(f"[Image Server] Unsupported head_camera_type: {self.head_camera_type}")

        # 初始化手腕摄像头列表（可选）
        self.wrist_cameras = []
        if self.wrist_camera_type and self.wrist_camera_id_numbers:
            if self.wrist_camera_type == 'opencv':
                for device_id in self.wrist_camera_id_numbers:
                    camera = OpenCVCamera(device_id=device_id, img_shape=self.wrist_image_shape, fps=self.fps)
                    self.wrist_cameras.append(camera)
            elif self.wrist_camera_type == 'realsense':
                for serial_number in self.wrist_camera_id_numbers:
                    camera = RealSenseCamera(img_shape=self.wrist_image_shape, fps=self.fps, serial_number=serial_number)
                    self.wrist_cameras.append(camera)
            else:
                print(f"[Image Server] Unsupported wrist_camera_type: {self.wrist_camera_type}")

        # 初始化 ZeroMQ 发布者
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")

        # 单元测试模式下初始化性能指标
        if self.Unit_Test:
            self._init_performance_metrics()

        # 打印已初始化的摄像头信息
        for cam in self.head_cameras:
            if isinstance(cam, OpenCVCamera):
                h = int(cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w = int(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                print(f"[Image Server] Head camera {cam.id} resolution: {h} x {w}")
            elif isinstance(cam, RealSenseCamera):
                print(f"[Image Server] Head camera {cam.serial_number} resolution: {cam.img_shape[0]} x {cam.img_shape[1]}")
            else:
                print("[Image Server] Unknown camera type in head_cameras.")

        for cam in self.wrist_cameras:
            if isinstance(cam, OpenCVCamera):
                h = int(cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w = int(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                print(f"[Image Server] Wrist camera {cam.id} resolution: {h} x {w}")
            elif isinstance(cam, RealSenseCamera):
                print(f"[Image Server] Wrist camera {cam.serial_number} resolution: {cam.img_shape[0]} x {cam.img_shape[1]}")
            else:
                print("[Image Server] Unknown camera type in wrist_cameras.")

        print("[Image Server] Image server has started, waiting for client connections...")

    def _init_performance_metrics(self):
        """
        单元测试模式：初始化用于计算和打印实际发送 FPS 的性能指标。
        """
        self.frame_count = 0  # 总发送帧数
        self.time_window = 1.0  # 时间窗口 (秒)
        self.frame_times = deque()  # 存储最近 time_window 内的时间戳
        self.start_time = time.time()  # 记录开始时间

    def _update_performance_metrics(self, current_time):
        """
        更新性能指标：将当前时间添加到 deque，移除过期时间戳，并累加帧计数。
        Args:
            current_time (float): 当前时间戳。
        """
        # 添加新时间戳
        self.frame_times.append(current_time)
        # 移除超过 time_window 范围的旧时间戳
        while self.frame_times and self.frame_times[0] < current_time - self.time_window:
            self.frame_times.popleft()
        # 增加总帧计数
        self.frame_count += 1

    def _print_performance_metrics(self, current_time):
        """
        每发送若干帧打印一次实际 FPS、总帧数和已运行时间。
        Args:
            current_time (float): 当前时间戳。
        """
        # 每 30 帧打印一次
        if self.frame_count % 30 == 0:
            elapsed_time = current_time - self.start_time
            real_time_fps = len(self.frame_times) / self.time_window
            print(f"[Image Server] Real-time FPS: {real_time_fps:.2f}, Total frames sent: {self.frame_count}, Elapsed time: {elapsed_time:.2f} sec")

    def _close(self):
        """
        关闭所有摄像头并关闭 ZeroMQ 连接。
        """
        for cam in self.head_cameras:
            cam.release()
        for cam in self.wrist_cameras:
            cam.release()
        self.socket.close()
        self.context.term()
        print("[Image Server] The server has been closed.")

    def send_process(self):
        """
        主循环：不断从摄像头获取图像，将头部和手腕图像拼接，编码为 JPEG，并通过 ZMQ 发送。
        在 Unit_Test 模式下，前置打包时间戳和帧 ID。
        当捕获 KeyboardInterrupt 时退出并调用 _close。
        """
        try:
            while True:
                # 获取头部所有摄像头帧
                head_frames = []
                for cam in self.head_cameras:
                    if self.head_camera_type == 'opencv':
                        color_image = cam.get_frame()
                        if color_image is None:
                            print("[Image Server] Head camera frame read is error.")
                            break
                    elif self.head_camera_type == 'realsense':
                        color_image, depth_image = cam.get_frame()
                        if color_image is None:
                            print("[Image Server] Head camera frame read is error.")
                            break
                    head_frames.append(color_image)
                # 若读取失败导致帧数不符，则退出循环
                if len(head_frames) != len(self.head_cameras):
                    break
                # 水平拼接头部多相机画面
                head_color = cv2.hconcat(head_frames)
                
                # 若配置了手腕摄像头，获取并拼接
                if self.wrist_cameras:
                    wrist_frames = []
                    for cam in self.wrist_cameras:
                        if self.wrist_camera_type == 'opencv':
                            color_image = cam.get_frame()
                            if color_image is None:
                                print("[Image Server] Wrist camera frame read is error.")
                                break
                        elif self.wrist_camera_type == 'realsense':
                            color_image, depth_image = cam.get_frame()
                            if color_image is None:
                                print("[Image Server] Wrist camera frame read is error.")
                                break
                        wrist_frames.append(color_image)
                    # 若读取失败帧数不符，则退出循环
                    if len(wrist_frames) != len(self.wrist_cameras):
                        break
                    wrist_color = cv2.hconcat(wrist_frames)

                    # 将头部和手腕画面拼接
                    full_color = cv2.hconcat([head_color, wrist_color])
                else:
                    full_color = head_color

                # 编码为 JPEG
                ret, buffer = cv2.imencode('.jpg', full_color)
                if not ret:
                    print("[Image Server] Frame imencode is failed.")
                    continue

                jpg_bytes = buffer.tobytes()

                # 单元测试模式下，在前面打包时间戳和帧 ID
                if self.Unit_Test:
                    timestamp = time.time()
                    frame_id = self.frame_count
                    # 8 字节 double + 4 字节 unsigned int
                    header = struct.pack('dI', timestamp, frame_id)
                    message = header + jpg_bytes
                else:
                    message = jpg_bytes

                # 发送消息
                self.socket.send(message)

                # 更新并打印性能指标（若 Unit_Test）
                if self.Unit_Test:
                    current_time = time.time()
                    self._update_performance_metrics(current_time)
                    self._print_performance_metrics(current_time)

        except KeyboardInterrupt:
            print("[Image Server] Interrupted by user.")
        finally:
            # 关闭资源
            self._close()


if __name__ == "__main__":
    # 示例配置，用于本地测试
    config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [480, 1280],  # Head camera resolution
        'head_camera_id_numbers': [0],
        'wrist_camera_type': 'opencv',
        'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
        'wrist_camera_id_numbers': [2, 4],
    }

    server = ImageServer(config, Unit_Test=False)
    server.send_process()
