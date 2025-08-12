# -*- coding: utf-8 -*-  # 声明UTF-8编码格式，确保中文字符正常解析[3,4](@ref)
import numpy as np
import time
import argparse
import cv2  # OpenCV库用于图像处理和显示
from multiprocessing import shared_memory, Array, Lock  # 多进程共享内存和同步机制
import threading
import os 
import sys
import panda_py
from panda_py import libfranka
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)  # 添加父目录到系统路径，用于模块导入

# 导入自定义机器人控制模块
from teleop.open_television.franka.tv_wrapper import TeleVisionWrapper
from teleop.robot_control.robot_arm import Franka_Panda_ArmController
from teleop.robot_control.robot_arm_ik import Franka_Panda_ARMIK
from teleop.image_server.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter

if __name__ == '__main__':
    # ================== 命令行参数解析 ==================
    """ 参数说明：
    --task_dir: 数据保存路径，默认./utils/data
    --frequency: 数据保存频率（Hz），默认30
    --record: 是否启用数据记录模式
    --arm: 机械臂型号选择（franka_panda）
    --hand: 手部控制器类型选择 """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type = str, default = './utils/data', help = '数据保存路径')
    parser.add_argument('--frequency', type = int, default = 30.0, help = '数据保存频率（Hz）')
    parser.add_argument('--record', action = 'store_true', help = '启用数据记录')
    parser.add_argument('--no-record', dest = 'record', action = 'store_false', help = '禁用数据记录')
    parser.set_defaults(record = False)  # 默认不启用数据记录
    parser.add_argument('--arm', type=str, choices=['franka_panda'], default='franka_panda', help='机械臂型号选择')
    parser.add_argument('--hand', type=str, choices=['gripper', 'inspire1'], default='gripper', help='手部控制器类型')
    args = parser.parse_args()
    print(f"运行参数:{args}\n")

    # ================== 图像采集配置 ==================
    """ 摄像头参数配置：
    - 头部摄像头：分辨率1280x480，ID0
    - 手腕摄像头：分辨率640x480，ID2/4
    - 双目判断逻辑：摄像头数量>1 或宽高比>2.0 """
    img_config = {
        'fps': 30,  # 图像采集帧率
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [480, 640],  # 图像高度x宽度
        'head_camera_id_numbers': [0],  # 头部摄像头设备ID
        # 'wrist_camera_type': 'opencv',
        'wrist_camera_image_shape': [480, 640],  # 手腕摄像头分辨率
        'wrist_camera_id_numbers': [2],  # 手腕摄像头设备ID
    }
    ASPECT_RATIO_THRESHOLD = 2.0  # 双目判断宽高比阈值
    BINOCULAR = len(img_config['head_camera_id_numbers']) > 1 or (
        img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD)
    WRIST = 'wrist_camera_type' in img_config  # 手腕摄像头启用标志

    # ================== 共享内存初始化 ==================
    """ 创建共享内存用于跨进程图像传输：
    - 头部摄像头：双目模式下宽度扩展为2倍
    - 手腕摄像头：双摄像头水平拼接 """
    if BINOCULAR:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)
    
    # 头部摄像头共享内存
    tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=tv_img_shm.buf)

    # 手腕摄像头共享内存（若启用）

    if WRIST:  
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=wrist_img_shm.buf)
        img_client = ImageClient(tv_img_shape=tv_img_shape, tv_img_shm_name=tv_img_shm.name,
                                wrist_img_shape=wrist_img_shape, wrist_img_shm_name=wrist_img_shm.name)
    else:
        img_client = ImageClient(tv_img_shape=tv_img_shape, tv_img_shm_name=tv_img_shm.name)

    # ================== 多线程初始化 ==================
    """ 启动图像接收线程：
    - 独立守护线程运行
    - 通过共享内存实时获取摄像头数据 """
    image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)
    image_receive_thread.start()

    # ================== 设备控制器初始化 ==================
    # XR设备通信接口
    tv_wrapper = TeleVisionWrapper(BINOCULAR, tv_img_shape, tv_img_shm.name)
    
    # 机械臂控制器选择
    if args.arm == 'franka_panda':
        arm_ctrl = Franka_Panda_ArmController()  # franka panda型号机械臂控制器
        arm_ik = Franka_Panda_ARMIK()          # 对应的逆运动学求解器
    else:
        print("无机械臂控制器")

    # 手部控制器初始化                         
    if args.hand == "inspire1":
        """ 灵巧手控制器：
        - 共享数组存储左右手75维骨骼数据
        - 互斥锁保证数据同步安全
        - 状态数组存储14维关节状态 """
        print("Inspire1控制器开发中...")
        left_hand_array = Array('d', 75, lock=True)         # 左手输入数据缓冲区
        right_hand_array = Array('d', 75, lock=True)        # 右手输入数据缓冲区
        dual_hand_data_lock = Lock()                        # 数据访问互斥锁
        dual_hand_state_array = Array('d', 12, lock=False)  # 关节状态输出
        dual_hand_action_array = Array('d', 12, lock=False) # 控制指令输出
        # hand_ctrl = Inspire_Controller(left_hand_array, right_hand_array, 
        #                             dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)
    else:
        pass
    
    # ================== 数据记录模块 ==================
    if args.record:
        """ 数据记录器：
        - 按指定频率保存操作数据
        - 支持断点续录功能 """
        recorder = EpisodeWriter(task_dir=args.task_dir, frequency=args.frequency, rerun_log=True)
        recording = False  # 录制状态标志
        
    # ================== 主控制循环 ==================
    try:
        user_input = input("请输入启动指令（输入'r'开始运行）：\n")
        if user_input.lower() == 'r':
            arm_ctrl.speed_gradual_max()  # 机械臂缓启动

            running = True
            while running:
                start_time = time.time()
                
                # 从XR设备获取实时数据
                """ 获取数据包含：
                - 头部旋转矩阵
                - 左右手腕空间坐标
                - 左右手部骨骼数据 """
                head_rmat, left_wrist, right_wrist, left_hand, right_hand,is_pinching,_ = tv_wrapper.step()

                # 机械臂状态获取
                current_lr_arm_q = arm_ctrl.get_current_dual_arm_q()   # 当前关节角度
                current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq() # 当前关节角速度

                # 逆运动学求解
                """ 求解过程：
                1. 根据手腕目标位姿计算关节角度
                2. 考虑当前运动状态进行动态调整 """
                time_ik_start = time.time()
                sol_q = panda_py.ik(right_wrist)
                time_ik_end = time.time()
                if np.isnan(sol_q[0]):
                    print("逆运动学求解失败，关节角度包含NaN值，跳过本次控制")
                    continue
                arm_ctrl.ctrl_dual_arm(sol_q, None,is_pinching)  # 发送控制指令
                # 图像显示处理
                tv_resized_image = cv2.resize(tv_img_array, (tv_img_shape[1]//2, tv_img_shape[0]//2))
                tv_resized_image = cv2.resize(tv_img_array, (tv_img_shape[1] // 2, tv_img_shape[0] // 2))
                cv2.imshow("record image", tv_resized_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                elif key == ord('s') and args.record:
                    recording = not recording # state flipping
                    if recording:
                        if not recorder.create_episode():
                            recording = False
                    else:
                        recorder.save_episode()

                # record data
                if args.record:
                    # dex hand or gripper
                    if args.hand == "gripper":
                        pass
                    elif args.hand == "inspire1":
                        with dual_hand_data_lock:
                            left_hand_state = dual_hand_state_array[:6]
                            right_hand_state = dual_hand_state_array[-6:]
                            left_hand_action = dual_hand_action_array[:6]
                            right_hand_action = dual_hand_action_array[-6:]
                    else:
                        print("No dexterous hand set.")
                        pass
                    # head image
                    current_tv_image = tv_img_array.copy()
                    # wrist image
                    if WRIST:
                        current_wrist_image = wrist_img_array.copy()
                    # arm state and action
                    left_arm_state  = current_lr_arm_q[:7]
                    right_arm_state = current_lr_arm_q[-7:]
                    left_arm_action = sol_q[:7]
                    right_arm_action = sol_q[-7:]

                    if recording:
                        colors = {}
                        depths = {}
                        if BINOCULAR:
                            colors[f"color_{0}"] = current_tv_image[:, :tv_img_shape[1]//2]
                            colors[f"color_{1}"] = current_tv_image[:, tv_img_shape[1]//2:]
                            if WRIST:
                                colors[f"color_{2}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                                colors[f"color_{3}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                        else:
                            colors[f"color_{0}"] = current_tv_image
                            if WRIST:
                                colors[f"color_{1}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                                # colors[f"color_{2}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                        states = {
                            "left_arm": {                                                                    
                                "qpos":  [],    # numpy.array -> list
                                "qvel":   [],                          
                                "torque": [],                        
                            }, 
                            "right_arm": {                                                                    
                                "qpos":   right_arm_state.tolist(),       
                                "qvel":   [],                          
                                "torque": [],                         
                            },                        
                            "left_hand": {                                                                    
                                "qpos":   [],           
                                "qvel":   [],                           
                                "torque": [],                          
                            }, 
                            "right_hand": {                                                                    
                                "qpos":   [],       
                                "qvel":   [],                           
                                "torque": [],  
                            }, 
                            "body": None, 
                        }
                        actions = {
                            "left_arm": {                                   
                                "qpos":   [],       
                                "qvel":   [],       
                                "torque": [],      
                            }, 
                            "right_arm": {                                   
                                "qpos":   right_arm_action.tolist(),       
                                "qvel":   [],       
                                "torque": [],       
                            },                         
                            "left_hand": {                                   
                                "qpos":   [],       
                                "qvel":   [],       
                                "torque": [],       
                            }, 
                            "right_hand": {                                   
                                "qpos":   [],       
                                "qvel":   [],       
                                "torque": [], 
                            }, 
                            "body": None, 
                        }
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / float(args.frequency)) - time_elapsed)
                time.sleep(sleep_time)
                print(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        print("检测到中断信号，正在关闭 TeleVision...")
        if tv_wrapper.process.is_alive():
            tv_wrapper.process.terminate()   # 杀掉子进程
            tv_wrapper.process.join()        # 等待子进程退出
        print("已成功关闭进程并释放端口。")

        print("检测到键盘中断信号...")
        """ 资源清理：
        1. 机械臂归位
        2. 释放共享内存
        3. 关闭数据记录器 """
        arm_ctrl.ctrl_dual_arm_go_home()  # 机械臂安全归位
        tv_img_shm.unlink()  # 释放共享内存
        tv_img_shm.close()
        if WRIST:
            wrist_img_shm.unlink()
            wrist_img_shm.close()
        if args.record:
            recorder.close()  # 关闭数据记录文件
        print("程序安全退出")
        print("检测到键盘中断信号...")
    finally:
        """ 资源清理：
        1. 机械臂归位
        2. 释放共享内存
        3. 关闭数据记录器 """
        # arm_ctrl.ctrl_dual_arm_go_home()  # 机械臂安全归位
        tv_img_shm.unlink()  # 释放共享内存
        tv_img_shm.close()
        if WRIST:
            wrist_img_shm.unlink()
            wrist_img_shm.close()
        if args.record:
            recorder.close()  # 关闭数据记录文件
        print("程序安全退出")