import numpy as np
from typing import List, Dict
import numpy as np
class FingerAngleProcessor():
    def __init__(self):
        self.finger_order = ["pinky", "ring", "middle", "index", "thumb"]

    def compute_plane_normal(self,p1, p2, p3):
        """通过3点拟合掌心平面的法向量"""
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p1)
        normal = np.cross(v1, v2)
        return normal / np.linalg.norm(normal)

    def compute_flexion_angles(self,landmarks: List[List[float]]) -> Dict[str, float]:
        fingers = {
            'thumb':  (1, 4),    # MCP: 1, Tip: 4
            'index':  (5, 9),
            'middle': (9 + 1, 9 + 5),
            'ring':   (13 + 1, 13 + 5),
            'pinky':  (17 + 1, 17 + 5),
        }
        # print("landmarks",landmarks[0])
        # 掌心平面拟合点（wrist、middle、ring MCP）
        palm_normal = self.compute_plane_normal(
            landmarks[0],     # wrist
            landmarks[9],     # middle-finger-metacarpal
            landmarks[13],    # ring-finger-metacarpal
        )
        # print("palm_normal",palm_normal[0])
        angles = {}
        if np.isnan(palm_normal[0]):
            return {
            'thumb':  -200,    # MCP: 1, Tip: 4
            'index':  -200,
            'middle': -200,
            'ring':   -200,
            'pinky':  -200,
        }
        for finger, (mcp_idx, tip_idx) in fingers.items():
            v = np.array(landmarks[tip_idx]) - np.array(landmarks[mcp_idx])
            v = v / np.linalg.norm(v)
            dot = np.dot(v, palm_normal)
            angle_rad = np.arccos(np.clip(dot, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)
            flexion_angle = np.clip(angle_deg,-13,51.6) if finger == 'thumb' else np.clip(180-(90-angle_deg),19,176.7)  # 屈伸运动: 90表示垂直手掌，越小越弯曲
            angles[finger] = round(flexion_angle, 2)
        return angles
    def map_angles_to_scores(self,angle_dict, score_max=1000, score_min=0):
        """
        将输入的手指角度字典线性映射到给定的分数范围（默认为 [score_max, score_min] 反向映射）。

        参数:
            angle_dict (dict): 键为手指名称，值为对应的角度（float，单位度）。
            score_max (float): 最大分数，对应最小角度。
            score_min (float): 最小分数，对应最大角度。

        返回:
            dict: 键为手指名称，值为映射后的分数（float）。
        """
        mapped = []
        # print("angle_dict",angle_dict)
        for finger, angle in angle_dict.items():
            min_finger = 19.0
            max_finger = 176.0
            # 确定各手指的角度范围
            if finger == "thumb":
                min_finger = -13.0
                max_finger = 53.6
            if angle == -200:
                angle_dict[finger] = -1
            else:
                # 线性映射
                angle_dict[finger] = int(round(((angle - min_finger) / (max_finger - min_finger)) * (score_max - score_min) + score_min))
                if angle_dict[finger] > 1000:
                    angle_dict[finger] = 1000
                if angle_dict[finger] < 0:
                    angle_dict[finger] = 200
        mapped = [angle_dict[finger] for finger in self.finger_order]
        mapped.append(0)
        # print('mapped',mapped)
        return mapped
import time
import struct
import numpy as np
import serial
from typing import Optional, List, Union

class InspireHandException(Exception):
    """自定义异常：手部控制器错误"""
    pass

class InspireHand:
    def __init__(self,
                 port: str = "/dev/ttyUSB0",
                 baudrate: int = 115200,
                 device_id: int = 1,
                 timeout: float = 0.1):
        """
        初始化 Inspire Hand 控制器

        :param port: 串口端口，例如 '/dev/ttyUSB0' 或 'COM3'
        :param baudrate: 波特率，默认 115200
        :param device_id: 设备 ID，默认 1
        :param timeout: 串口超时时间（秒）
        """
        # 打开串口
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout
        )
        self.device_id = device_id
        # 接收缓存
        self._recv_buffer = bytearray(1024)

    def _calculate_checksum(self, data: bytes, start: int = 2) -> int:
        """
        计算校验和，从 data[start] 开始到末尾字节前一位，并取低 8 位
        :param data: 输入字节序列
        :param start: 校验起始位置（默认从第三个字节开始）
        :return: 1 字节校验值
        """
        return sum(data[start:-1]) & 0xFF

    def change_id(self, old_id: int, new_id: int) -> bool:
        """
        更改设备 ID
        :param old_id: 当前设备 ID
        :param new_id: 新的设备 ID
        :return: 成功返回 True，失败抛出异常
        """
        # 构造命令头部，不含校验
        cmd = bytearray([0xEB, 0x90, old_id, 0x04, 0x12, 0xE8, 0x03, new_id, 0x00])
        # 计算并填充校验
        cmd[-1] = self._calculate_checksum(cmd)

        success = self._send_command(bytes(cmd), expected_response_size=9)
        if not success:
            raise InspireHandException("更改设备 ID 失败")
        self.device_id = new_id
        return True

    def set_position(self, positions: Union[np.ndarray, List[float]]) -> bool:
        """
        设置六个手指目标位置，范围 [0,1]
        :param positions: 6 元素列表或数组，每个值在 [0,1]
        :return: 成功返回 True
        """
        # 转为 numpy 数组
        arr = np.array(positions, dtype=np.float32)
        if arr.shape != (6,):
            raise ValueError("Positions must be a 6-element array")

        # 缩放到 0-1000 并转为无符号 16 位整数
        scaled = np.clip(arr * 1000, 0, 1000).astype(np.uint16)

        # 命令头（含 device_id）
        header = bytearray([0xEB, 0x90, self.device_id, 0x0F, 0x12, 0xCE, 0x05])
        # 按小端打包位置数据
        for val in scaled:
            header += struct.pack('<H', int(val))
        # 占位校验
        header += b'\x00'
        # 计算并填充校验
        header[-1] = self._calculate_checksum(header)

        return self._send_command(bytes(header), expected_response_size=9)

    def get_position(self) -> Optional[np.ndarray]:
        """
        获取当前手指位置值
        :return: 6 元素数组（归一化到 [0,1]），失败返回 None
        """
        cmd = bytearray([0xEB, 0x90, self.device_id, 0x04, 0x11, 0x0A, 0x06, 0x0C, 0x00])
        cmd[-1] = self._calculate_checksum(cmd)

        if not self._send_command(bytes(cmd), expected_response_size=20):
            return None
        try:
            # 从接收缓存解析每个手指位置
            return np.array([
                struct.unpack_from('<H', self._recv_buffer, 7 + 2*i)[0] / 1000.0
                for i in range(6)
            ], dtype=np.float32)
        except struct.error as e:
            raise InspireHandException(f"解析位置数据失败: {e}")

    def set_velocity(self, velocities: List[int]) -> bool:
        """
        设置六个手指的运动速度（Signed Short）
        :param velocities: 6 个速度值
        :return: 成功返回 True
        """
        if len(velocities) != 6:
            raise ValueError("Exactly 6 velocity values required")

        header = bytearray([0xEB, 0x90, self.device_id, 0x0F, 0x12, 0xF2, 0x05])
        for v in velocities:
            header += struct.pack('<h', int(v))
        header += b'\x00'
        header[-1] = self._calculate_checksum(header)

        return self._send_command(bytes(header), expected_response_size=9)

    def set_force_threshold(self, thresholds: List[int]) -> bool:
        """
        设置力控阈值，单位：克
        :param thresholds: 6 个阈值，每个在 0-1000
        :return: 成功返回 True
        """
        if any(t < 0 or t > 1000 for t in thresholds):
            raise ValueError("Thresholds must be between 0 and 1000")

        header = bytearray([0xEB, 0x90, self.device_id, 0x0F, 0x12, 0xDA, 0x05])
        for t in thresholds:
            header += struct.pack('<H', int(t))
        header += b'\x00'
        header[-1] = self._calculate_checksum(header)

        return self._send_command(bytes(header), expected_response_size=9)

    def get_force(self) -> Optional[np.ndarray]:
        """
        获取当前力传感器测量值，单位：牛顿
        :return: 6 元素数组（单位 N），失败返回 None
        """
        cmd = bytearray([0xEB, 0x90, self.device_id, 0x04, 0x11, 0x2E, 0x06, 0x0C, 0x00])
        cmd[-1] = self._calculate_checksum(cmd)

        if not self._send_command(bytes(cmd), expected_response_size=20):
            return None
        try:
            # 克 转换为 牛顿： g*9.8/1000
            return np.array([
                struct.unpack_from('<H', self._recv_buffer, 7 + 2*i)[0] * 9.8 / 1000.0
                for i in range(6)
            ], dtype=np.float32)
        except struct.error as e:
            raise InspireHandException(f"解析力数据失败: {e}")

    def clear_errors(self) -> bool:
        """
        清除错误状态
        :return: 成功返回 True
        """
        cmd = bytearray([0xEB, 0x90, self.device_id, 0x04, 0x12, 0xEC, 0x03, 0x01, 0x00])
        cmd[-1] = self._calculate_checksum(cmd)
        return self._send_command(bytes(cmd), expected_response_size=9)

    def calibrate_force_sensors(self) -> bool:
        """
        校准力传感器，内部会等待 10 秒
        :return: 校准完成返回 True，否则 False
        """
        cmd = bytearray([0xEB, 0x90, self.device_id, 0x04, 0x12, 0x2F, 0x06, 0x01, 0x00])
        cmd[-1] = self._calculate_checksum(cmd)
        if not self._send_command(bytes(cmd), expected_response_size=9):
            return False
        time.sleep(10)
        return self._read_response(expected_size=9)

    def _send_command(self,
                     command: bytes,
                     expected_response_size: int) -> bool:
        """
        私有：发送命令并读取响应
        :param command: 完整命令字节
        :param expected_response_size: 期望响应长度
        :return: 响应校验通过返回 True
        """
        try:
            self.ser.write(command)
            time.sleep(0.005)
            return self._read_response(expected_size=expected_response_size)
        except serial.SerialException as e:
            raise InspireHandException(f"串口通信错误: {e}")

    def _read_response(self, expected_size: int) -> bool:
        """
        私有：读取并校验响应
        :param expected_size: 期望长度
        :return: 校验通过返回 True
        """
        data = self.ser.read(expected_size)
        if len(data) != expected_size:
            return False
        # 校验并缓存数据
        if data[-1] != self._calculate_checksum(data):
            return False
        self._recv_buffer[:expected_size] = data
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """
        关闭串口连接
        """
        if self.ser.is_open:
            self.ser.close()

# 使用示例
if __name__ == "__main__":
    try:
        with InspireHand(port="/dev/ttyUSB0") as hand:
            hand.set_position([1.3]*6)
            pos = hand.get_position()
            if pos is not None:
                print(f"当前位置: {pos}")
            hand.set_force_threshold([300]*6)
            forces = hand.get_force()
            if forces is not None:
                print(f"当前力值: {forces} ")
    except InspireHandException as e:
        print(f"操作失败: {e}")
    except KeyboardInterrupt:
        print("用户中断操作")
    # hand_points = [[0.0, 0.0, 0.0], [-0.016240413292689215, -0.03457382009450649, 0.030125786842884472], [-0.016348741381851273, -0.061186558754578346, 0.04880306401488088], [-0.015360146724679336, -0.09240504792260174, 0.06170289945286178], [-0.01654478877079102, -0.114554063127283, 0.0723906049394224], [-0.009583497931303864, -0.036486168413091935, 0.01977302435214956], [-0.007316473492816389, -0.09599626260536098, 0.023550670663128104], [-0.04345916896802793, -0.10663779456944944, 0.019197915985110003], [-0.04798850081540662, -0.08293452346146246, 0.022080634587005266], [-0.031370310785068245, -0.06825006105439502, 0.025156683755760856], [-0.007686656951013671, -0.03432367679042936, 0.003563558501876063], [-0.0025431739097165895, -0.09564665134682082, 0.0017258660941896764], [-0.04499704463142207, -0.10190333285954845, 0.0006068392605104167], [-0.050302976332276206, -0.07554925129747891, 0.006630888198930762], [-0.0302390980385554, -0.06205254815873751, 0.012948899751295206], [-0.0060157978613539775, -0.034775526477476504, -0.014992402376417413], [-0.006529314424900345, -0.08869385194755308, -0.01746527366041073], [-0.04515368824859278, -0.09350225524683342, -0.015070369235489878], [-0.04902577404487962, -0.06854516929135324, -0.006806086776547926], [-0.027895122886168355, -0.05701862604185126, -0.002926907981693705], [-0.00941984332168877, -0.03407357042101089, -0.02299856733801353], [-0.013691194786129568, -0.07789588827074834, -0.03505407694712015], [-0.04256913408075702, -0.08595749942471453, -0.028359395026556045], [-0.04765462901504813, -0.06787808902677517, -0.020624695877706767], [-0.02908633401539129, -0.05641395859571641, -0.018190916615104413]]
    # processor = FingerAngleProcessor()
    # angles = processor.map_angles_to_scores(processor.compute_flexion_angles(hand_points))
    # print("各手指关节角度（单位：度）：",angles)
