import time
from vuer import Vuer
from vuer.schemas import ImageBackground, Hands
from multiprocessing import Array, Process, shared_memory  # 用于进程间共享数据
import numpy as np
import asyncio
import cv2

from multiprocessing import context
Value = context._default_context.Value  # 获取默认上下文的值类型


class TeleVision:
    """主视觉处理类，负责3D场景渲染和图像传输"""
    
    def __init__(self, binocular, img_shape, img_shm_name, cert_file="cert.pem", key_file="key.pem", ngrok=False):
        """初始化3D显示系统
        Args:
            binocular: 是否双目模式(VR模式)
            img_shape: 图像尺寸 (height, width, channels)
            img_shm_name: 共享内存名称，用于获取图像数据
            cert_file: SSL证书文件路径
            key_file: SSL密钥文件路径
            ngrok: 是否使用ngrok进行内网穿透
        """
        self.binocular = binocular  # 是否为双目显示模式
        try:
            width = int(img_shape[1])
            height = int(img_shape[0])
        except (ValueError, TypeError):
            # Assume img_shape is already a tuple of ints
            height, width = img_shape

        # Compute image width depending on binocular mode
        self.img_width = width // 2 if binocular else width
        self.img_height = height
        # 根据显示模式计算单目宽度(双目模式下图像会被水平分割)
        self.img_width = img_shape[1] // 2 if binocular else img_shape[1]
        
        # 初始化Vuer服务器(3D场景服务器)
        if ngrok:
            self.vuer = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3)  # 无加密模式
        else:
            self.vuer = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)

        # 注册事件处理函数
        self.vuer.add_handler("HAND_MOVE")(self.on_hand_move)  # 手部移动事件
        self.vuer.add_handler("CAMERA_MOVE")(self.on_cam_move)  # 相机移动事件

        # 初始化共享内存访问(用于读取图像数据)
        existing_shm = shared_memory.SharedMemory(name=img_shm_name)
        self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=existing_shm.buf)

        # 根据显示模式启动对应的图像处理协程
        if binocular:
            self.vuer.spawn(start=False)(self.main_image_binocular)  # 双目模式
        else:
            self.vuer.spawn(start=False)(self.main_image_monocular)  # 单目模式

        # 初始化共享数据结构(用于进程间通信)
        self.left_hand_shared = Array('d', 16, lock=True)    # 左手变换矩阵(4x4)
        self.right_hand_shared = Array('d', 16, lock=True)   # 右手变换矩阵(4x4)
        self.left_landmarks_shared = Array('d', 75, lock=True)  # 左手关节点坐标(25x3)
        self.right_landmarks_shared = Array('d', 75, lock=True) # 右手关节点坐标
        
        self.head_matrix_shared = Array('d', 16, lock=True)  # 头部变换矩阵(4x4)
        self.aspect_shared = Value('d', 1.0, lock=True)      # 屏幕宽高比
        import ctypes
        self.is_pinching= Value(ctypes.c_bool,False, lock=False)
        # 启动Vuer服务器进程
        self.process = Process(target=self.vuer_run)
        self.process.daemon = True  # 设置为守护进程
        self.process.start()

    def vuer_run(self):
        """运行Vuer服务器的主函数"""
        self.vuer.run()

    async def on_cam_move(self, event, session, fps=60):
        """处理相机移动事件的回调函数
        Args:
            event: 包含相机参数的事件数据
            session: Vuer会话对象
            fps: 帧率限制
        """
        try:
            # 更新头部姿态矩阵和屏幕宽高比
            self.head_matrix_shared[:] = event.value["camera"]["matrix"]  # 4x4变换矩阵
            self.aspect_shared.value = event.value['camera']['aspect']    # 宽高比
        except Exception as e:
            print(f"Camera move error: {e}")

    async def on_hand_move(self, event, session, fps=60):
        """处理手部移动事件的回调函数
        Args:
            event: 包含手部数据的事件对象
            session: Vuer会话对象
            fps: 帧率限制
        """
        try:
            # 更新手部变换矩阵和关节点数据
            # print("更新手部变换矩阵和关节点数据",event.value)
            self.is_pinching.value = event.value["rightState"]["pinching"]
            self.left_hand_shared[:] = event.value["leftHand"]            # 左手矩阵
            self.right_hand_shared[:] = event.value["rightHand"]           # 右手矩阵
            self.left_landmarks_shared[:] = np.array(event.value["leftLandmarks"]).flatten()  # 左手关节点
            self.right_landmarks_shared[:] = np.array(event.value["rightLandmarks"]).flatten()# 右手关节点
        except Exception as e:
            print(f"Hand move error: {e}")
    
    
    async def main_image_binocular(self, session, fps=60):
        """双目模式下的主图像处理循环"""
        # 初始化手部显示组件
        session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False)
        
        while True:
            # 从共享内存获取并处理图像
            display_image = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2RGB)  # 转换颜色空间
            
            # 更新左右眼图像到3D场景
            session.upsert(
                [
                    # 左眼图像(使用层掩码1)
                    ImageBackground(
                        display_image[:, :self.img_width],  # 左半部分图像
                        aspect=1.778,     # 预设宽高比(16:9)
                        height=1,         # 虚拟高度
                        distanceToCamera=1,  # 与相机距离
                        layers=1,         # 渲染层掩码(对应左眼相机)
                        format="jpeg",    # 压缩格式
                        quality=50,       # 压缩质量
                        key="background-left",  # 组件唯一标识
                        interpolate=True, # 启用插值
                    ),
                    # 右眼图像(使用层掩码2)
                    ImageBackground(
                        display_image[:, self.img_width:],  # 右半部分图像
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        layers=2,         # 对应右眼相机
                        format="jpeg",
                        quality=50,
                        key="background-right",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",  # 指定更新到背景子节点
            )
            # 保持约30fps的更新率(0.016秒*2 ≈ 30Hz)
            await asyncio.sleep(0.016 * 2)

    async def main_image_monocular(self, session, fps=60):
        """单目模式下的主图像处理循环"""
        session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=True, showRight=True)
        while True:
            # 处理单目图像
            display_image = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2RGB)
            session.upsert(
                [
                    ImageBackground(
                        display_image,  # 完整图像
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        format="jpeg",
                        quality=50,
                        key="background-mono",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            await asyncio.sleep(0.016)  # 约60fps

    @property
    def left_hand(self):
        """获取左手变换矩阵(4x4 column-major格式)"""
        return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")
    
    @property
    def right_hand(self):
        """获取右手变换矩阵(4x4 column-major格式)"""
        return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")
        
    @property
    def left_landmarks(self):
        """获取左手关节点坐标(25个点，每个点xyz坐标)"""
        return np.array(self.left_landmarks_shared[:]).reshape(25, 3)
    
    @property
    def right_landmarks(self):
        """获取右手关节点坐标(25个点，每个点xyz坐标)"""
        return np.array(self.right_landmarks_shared[:]).reshape(25, 3)

    @property
    def head_matrix(self):
        """获取头部变换矩阵(4x4 column-major格式)"""
        return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")

    @property
    def aspect(self):
        """获取当前屏幕宽高比"""
        return float(self.aspect_shared.value)
    
if __name__ == '__main__':
    """测试代码"""
    import os 
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)  # 添加父目录到系统路径
    
    import threading
    from image_server.image_client import ImageClient  # 自定义图像客户端

    # 初始化共享内存(用于图像传输)
    img_shape = (480, 640 * 2, 3)  # 双目图像尺寸(480p，水平双拼)
    img_shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
    img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=img_shm.buf)
    
    # 启动图像接收客户端
    img_client = ImageClient(tv_img_shape=img_shape, tv_img_shm_name=img_shm.name)
    image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)
    image_receive_thread.start()

    # 初始化3D显示系统(双目模式)
    tv = TeleVision(True, img_shape, img_shm.name)
    print("VR显示系统已启动，按Ctrl+C终止程序...")
    
    # 主循环(保持程序运行)
    while True:
        time.sleep(0.03)  # 降低CPU占用