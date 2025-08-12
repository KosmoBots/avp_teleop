import numpy as np
import threading
import time,json,math
from enum import IntEnum
import os 
import sys
import panda_py
from panda_py import libfranka

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)  # 添加父目录到系统路径，用于模块导入
kTopicLowCommand = "rt/lowcmd"
kTopicLowState = "rt/lowstate"
armTopicLowCommand = "rt/arm_Command"
armTopicLowState = "rt/arm_Feedback"
Franka_Num_Motors= 9
import logging
logging.basicConfig(level=logging.INFO)
class MotorState:
    def __init__(self):
        self.q = None #弧度
        self.dq = None #速度  弧度/s
class Franka_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(Franka_Num_Motors)]
class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data
class Franka_Panda_ArmController:
    """Franka_Panda型机器人双臂控制器，用于实现双臂关节的实时控制"""
    def __init__(self,hostname="192.168.3.100",username='user',password = 'password'):
        """控制器初始化，设置控制参数，初始化通信和线程"""
        print("Initialize Franka_Panda_ArmController...")
        
        # 初始化目标位置和力矩数组（14个关节）
        self.q_target = np.zeros(6)      # 目标关节角度数组
        self.tauff_target = np.zeros(6)  # 目标关节力矩数组

        # PID控制参数配置（分不同关节类型）
        self.kp_high = 300.0   # 高刚度关节的比例增益（如腿部主要关节）
        self.kd_high = 5.0     # 高刚度关节的微分增益
        self.kp_low = 140.0    # 低刚度关节的比例增益（如肩部等脆弱关节）
        self.kd_low = 3.0      # 低刚度关节的微分增益
        self.kp_wrist = 50.0   # 手腕关节的比例增益
        self.kd_wrist = 2.0    # 手腕关节的微分增益
        self.is_pinching = False
        # 运动控制参数
        self.all_motor_q = None           # 存储所有关节当前角度
        self.arm_velocity_limit = 20.0    # 关节运动速度限制（rad/s）
        self.control_dt = 1.0 / 250.0    # 控制周期（250Hz）

        # 渐进加速控制相关参数
        self._speed_gradual_max = False    # 是否启用渐进加速模式
        self._gradual_start_time = None    # 渐进加速开始时间
        self._gradual_time = None          # 渐进加速总时长

        # 系统初始化
        desk = panda_py.Desk(hostname, username, password)
        desk.unlock()
        desk.activate_fci()
        # 创建机械臂控制类
        self.robot = panda_py.Panda(hostname)
        gripper = libfranka.Gripper(hostname)
        # 等待直到成功连接机械臂
        while self.robot is None:
            time.sleep(0.01)
            self.robot = panda_py.Panda(hostname)
            self.gripper = libfranka.Gripper(hostname)
            print("[Franka_ArmController] Waiting to connect ...")
        self.lowstate_buffer = DataBuffer()  # 用于缓存关节状态数据
        # 启动状态订阅线程（实时获取关节状态）
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True  # 设置为守护线程
        self.subscribe_thread.start()
        # 获取并显示当前所有关节角度
        self.all_motor_q = self.get_current_motor_q()
        print(f"Current all body motor state q:\n{self.all_motor_q} \n")
        print(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        print("Lock all joints except two arms...\n")
        print("Lock OK!\n")
        # 启动控制命令发布线程
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()  # 创建线程锁确保数据同步
        self.publish_thread.daemon = True
        self.publish_thread.start()

        print("Initialize Franka_Panda_ArmController OK!\n")
    def is_valid_data(self,msg):
        data = json.loads(msg.data_)
        if data['seq'] == 10 and data['address'] == 2 and data['funcode'] == 1:
            return True
        else:
            return False
    def _subscribe_motor_state(self):
        """订阅线程函数：持续从机械臂获取电机状态并更新缓冲区"""
        while True:
            state = self.robot.get_state()
            if state is not None and state['q'] is not None :
                lowstate = Franka_LowState()  # 创建状态存储对象
                # 更新所有电机的角度和角速度
                current_pos=state['q']
                current_dq=state['dq']
                for id in range(Franka_Num_Motors):
                    lowstate.motor_state[id].q = current_pos[id]
                    lowstate.motor_state[id].dq = current_dq[id]
                self.lowstate_buffer.SetData(lowstate)  # 存储到缓冲区
            time.sleep(0.002)  # 控制订阅频率

    def clip_arm_q_target(self, target_q, velocity_limit):
        """限制目标角度变化速度，防止突变
        Args:
            target_q: 目标角度数组
            velocity_limit: 最大允许速度（rad/s）
        Returns:
            cliped_arm_q_target: 经过速度限制后的目标角度
        """
        current_q = self.get_current_dual_arm_q()  # 当前实际角度
        delta = target_q[0-6] - current_q  # 计算角度差
        # 计算速度缩放比例（基于控制周期）
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        # 应用速度限制后的目标角度
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target
    
    def _ctrl_motor_state(self):
        """控制线程函数：周期性地发送控制命令到电机"""
        i = 0
        while True:
            start_time = time.time()

            # 使用线程锁确保数据同步
            with self.ctrl_lock:
                arm_q_target = self.q_target
                arm_tauff_target = self.tauff_target

            # # 应用速度限制后的目标角度
            cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            # # 遍历所有手臂关节，设置控制命令
            # 发布控制命令
            joint_radian = tuple(cliped_arm_q_target)
            logging.info("move joint to {0}".format(cliped_arm_q_target))
            self.robot.move_to_joint_position(joint_radian)

            # 渐进加速逻辑处理
            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                # 在设定时间内线性增加速度限制
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            # 控制频率维持
            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)

    def ctrl_dual_arm(self, q_target, tauff_target,is_pinching):
        """设置双臂控制目标
        Args:
            q_target: 目标角度数组（14个元素）
            tauff_target: 目标力矩数组（14个元素）
        """
        with self.ctrl_lock:  # 线程安全操作
            self.q_target = q_target
            self.tauff_target = tauff_target
            self.is_pinching =is_pinching

    def get_mode_machine(self):
        """获取当前机器运行模式"""
        return self.lowstate_subscriber.Read().mode_machine

    def get_current_motor_q(self):
        """获取所有关节的当前角度"""
        print("get_current_motor_q",self.lowstate_buffer.GetData())
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in Franka_JointIndex])

    def get_current_dual_arm_q(self):
        """获取双臂关节的当前角度"""
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in Franka_JointArmIndex])

    def get_current_dual_arm_dq(self):
        """获取双臂关节的当前角速度"""
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in Franka_JointArmIndex])

    def ctrl_dual_arm_go_home(self):
        """控制双臂回到零位（初始位置）"""
        print("[Franka_Panda_ArmController] ctrl_dual_arm_go_home start...")
        with self.ctrl_lock:
            self.q_target = np.zeros(7)  # 设置目标角度为零
        tolerance = 0.05  # 位置收敛阈值
        # 持续检测直到所有关节接近零位
        while True:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                print("[Franka_Panda_ArmController] both arms have reached the home position.")
                break
            # time.sleep(0.05)

    def speed_gradual_max(self, t=5.0):
        """启用渐进加速模式
        Args:
            t: 加速总时间（秒），默认为5秒
        """
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        """立即设置最大速度限制（30 rad/s）"""
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        """判断是否为脆弱关节（需要较低刚度）
        Args:
            motor_index: 关节索引枚举值
        Returns:
            bool: 是否是脆弱关节
        """
        # 定义脆弱关节索引列表（如踝关节、肩部关节）
        weak_motors = [
            Franka_JointIndex.shoulder_joint.value,
            Franka_JointIndex.upperArm_joint.value,
            Franka_JointIndex.foreArm_joint.value,
            Franka_JointIndex.wrist1_joint.value,
            Franka_JointIndex.wrist2_joint.value,
            Franka_JointIndex.wrist3_joint.value,
        ]
        return motor_index.value in weak_motors

    def _Is_wrist_motor(self, motor_index):
        """判断是否为手腕关节
        Args:
            motor_index: 关节索引枚举值
        Returns:
            bool: 是否是手腕关节
        """
        # 定义手腕关节索引列表
        wrist_motors = [
            Franka_JointIndex.wrist1_joint.value,
            Franka_JointIndex.wrist2_joint.value,
            Franka_JointIndex.wrist3_joint.value,
        ]
        return motor_index.value in wrist_motors

class Franka_JointArmIndex(IntEnum):
    """双臂关节索引枚举定义"""
    shoulder_joint = 0
    upperArm_joint = 1
    foreArm_joint = 2
    wrist1_joint = 3
    wrist2_joint = 4
    wrist3_joint = 5


class Franka_JointIndex(IntEnum):
    """全身关节索引枚举定义"""
    shoulder_joint = 0
    upperArm_joint = 1
    foreArm_joint = 2
    wrist1_joint = 3
    wrist2_joint = 4
    wrist3_joint = 5
    # world_joint = 6 固定节点不算


if __name__ == "__main__":
    from robot_arm_ik import G1_29_ArmIK, H1_2_ArmIK
    import pinocchio as pin 

    # arm_ik = G1_29_ArmIK(Unit_Test = True, Visualization = False)
    # arm = G1_29_ArmController()
    arm_ik = H1_2_ArmIK(Unit_Test = True, Visualization = False)
    arm = H1_2_ArmController()

    # initial positon
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, +0.2, 0.1]),
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, -0.2, 0.1]),
    )

    rotation_speed = 0.005  # Rotation speed in radians per iteration
    q_target = np.zeros(35)
    tauff_target = np.zeros(35)

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program): \n")
    if user_input.lower() == 's':
        step = 0
        arm.speed_gradual_max()
        while True:
            if step <= 120:
                angle = rotation_speed * step
                L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
                R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

                L_tf_target.translation += np.array([0.001,  0.001, 0.001])
                R_tf_target.translation += np.array([0.001, -0.001, 0.001])
            else:
                angle = rotation_speed * (240 - step)
                L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
                R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

                L_tf_target.translation -= np.array([0.001,  0.001, 0.001])
                R_tf_target.translation -= np.array([0.001, -0.001, 0.001])

            L_tf_target.rotation = L_quat.toRotationMatrix()
            R_tf_target.rotation = R_quat.toRotationMatrix()

            current_lr_arm_q  = arm.get_current_dual_arm_q()
            current_lr_arm_dq = arm.get_current_dual_arm_dq()

            sol_q, sol_tauff = arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous, current_lr_arm_q, current_lr_arm_dq)

            arm.ctrl_dual_arm(sol_q, sol_tauff)

            step += 1
            if step > 240:
                step = 0
            time.sleep(0.01)