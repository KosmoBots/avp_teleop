import numpy as np
import math

class FrankaEmikaIK:
    """Franka Emika Panda机械臂逆运动学求解器"""
    
    def __init__(self):
        # 机械臂DH参数（单位：米）
        self.d1 = 0.3330   # 基座到第一关节的垂直距离
        self.d3 = 0.3160   # 第二到第三关节的垂直距离
        self.d5 = 0.3840   # 第四到第五关节的垂直距离
        self.d7e = 0.2104  # 第七关节到末端执行器的距离
        self.a4 = 0.0825   # 第三到第四关节的水平偏移
        self.a7 = 0.0880   # 第六到第七关节的水平偏移
        
        # 预计算连杆长度平方和实际长度
        self.LL24 = self.a4**2 + self.d3**2
        self.LL46 = self.a4**2 + self.d5**2
        self.L24 = math.sqrt(self.LL24)
        self.L46 = math.sqrt(self.LL46)
        
        # 预计算关键角度（单位：弧度）
        self.thetaH46 = math.atan(self.d5/self.a4)    # 第四到虚拟点的角度
        self.theta342 = math.atan(self.d3/self.a4)    # 第三到第四关节的角度
        self.theta46H = math.atan(self.a4/self.d5)    # 虚拟点到第六关节角度
        
        # 关节角度限制（单位：弧度）
        self.q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        
        # 默认初始关节角度
        self.kQDefault = np.array([0.0, -math.pi/4, 0.0, -3*math.pi/4, 0.0, math.pi/2, math.pi/4])
    
    def solve(self, O_T_EE, q_actual=None, q7=None):
        """
        逆运动学求解
        
        参数:
            O_T_EE: 4x4齐次变换矩阵，表示末端执行器在基坐标系中的位姿
            q_actual: 当前关节角度，用于解的选择（可选）
            q7: 第七关节的期望角度（可选）
            
        返回:
            7维关节角度数组，如果无解则返回NaN数组
        """
        # 初始化返回值为全NaN，表示无效解
        q_NAN = np.full(7, fill_value=float('nan'))
        q = np.zeros(7)
        
        # 设置默认参数
        if q_actual is None:
            q_actual = self.kQDefault
        if q7 is None:
            q7 = math.pi/4
        
        # 检查输入的q7是否在合法范围内
        if q7 <= self.q_min[6] or q7 >= self.q_max[6]:
            return q_NAN
        else:
            q[6] = q7  # 使用输入的q7值
        
        # 提取末端执行器旋转矩阵和位置
        R_EE = O_T_EE[:3, :3]
        z_EE = O_T_EE[:3, 2]
        p_EE = O_T_EE[:3, 3]
        
        # 计算第七关节位置（p7 = 末端位置 - 末端延长量）
        p_7 = p_EE - self.d7e * z_EE
        
        # 根据q7计算第六关节坐标系X轴方向
        x_EE_6 = np.array([math.cos(q7 - math.pi/4), -math.sin(q7 - math.pi/4), 0.0])
        x_6 = R_EE @ x_EE_6  # 转换到基坐标系
        x_6 /= np.linalg.norm(x_6)  # 归一化提高计算精度
        p_6 = p_7 - self.a7 * x_6  # 第六关节位置
        
        # 计算第四关节角度q4
        p_2 = np.array([0.0, 0.0, self.d1])  # 第二关节位置
        V26 = p_6 - p_2  # 第二到第六关节向量
        LL26 = np.dot(V26, V26)  # 向量长度的平方
        L26 = math.sqrt(LL26)  # 实际长度
        
        # 三角形不等式检查，确保解存在
        if (self.L24 + self.L46 < L26) or (self.L24 + L26 < self.L46) or (L26 + self.L46 < self.L24):
            return q_NAN
        
        # 使用余弦定理计算q4
        theta246 = math.acos((self.LL24 + self.LL46 - LL26) / (2 * self.L24 * self.L46))
        q[3] = theta246 + self.thetaH46 + self.theta342 - 2 * math.pi  # 组合角度
        
        # 检查q4合法性
        if q[3] <= self.q_min[3] or q[3] >= self.q_max[3]:
            return q_NAN
        
        # 计算第六关节角度q6
        theta462 = math.acos((LL26 + self.LL46 - self.LL24) / (2 * L26 * self.L46))
        theta26H = self.theta46H + theta462  # 组合角度
        D26 = -L26 * math.cos(theta26H)  # 投影长度
        
        # 构建第六关节坐标系
        Z_6 = np.cross(z_EE, x_6)  # Z轴由X和末端Z轴确定
        Y_6 = np.cross(Z_6, x_6)  # Y轴正交补全
        R_6 = np.column_stack((x_6, Y_6/np.linalg.norm(Y_6), Z_6/np.linalg.norm(Z_6)))  # 第六关节旋转矩阵
        
        # 将V26转换到第六关节坐标系
        V_6_62 = R_6.T @ (-V26)
        
        # 计算角度分量
        Phi6 = math.atan2(V_6_62[1], V_6_62[0])  # 平面旋转角
        Theta6 = math.asin(D26 / math.hypot(V_6_62[0], V_6_62[1]))
        
        # 根据当前关节角度选择解分支
        is_case6_0 = False  # 这个值应该根据实际情况计算
        is_case1_1 = (q_actual[1] < 0)
        
        # 根据配置选择解
        if is_case6_0:
            q[5] = math.pi - Theta6 - Phi6  # 情况0
        else:
            q[5] = Theta6 - Phi6  # 情况1
        
        # 角度归一化到[-π, π]
        if q[5] <= self.q_min[5]:
            q[5] += 2 * math.pi
        elif q[5] >= self.q_max[5]:
            q[5] -= 2 * math.pi
        
        # 最终合法性检查
        if q[5] <= self.q_min[5] or q[5] >= self.q_max[5]:
            return q_NAN
        
        # 计算第一、二关节角度q1,q2
        thetaP26 = 3 * math.pi/2 - theta462 - theta246 - self.theta342
        thetaP = math.pi - thetaP26 - theta26H
        LP6 = L26 * math.sin(thetaP26) / math.sin(thetaP)  # 几何投影长度
        
        # 计算第五关节方向
        z_6_5 = np.array([math.sin(q[5]), math.cos(q[5]), 0.0])
        z_5 = R_6 @ z_6_5  # 转换到基坐标系
        
        # 计算第二关节到虚拟点P的向量
        V2P = p_6 - LP6 * z_5 - p_2
        L2P = np.linalg.norm(V2P)  # 向量长度
        
        # 处理奇异位置（Z轴对齐情况）
        if abs(V2P[2]/L2P) > 0.999:
            q[0] = q_actual[0]  # 保持当前q1
            q[1] = 0.0  # q2设为0
        else:
            # 常规情况计算角度
            q[0] = math.atan2(V2P[1], V2P[0])  # q1的方位角
            q[1] = math.acos(V2P[2]/L2P)  # q2的俯仰角
            
            # 根据第二关节配置调整解
            if is_case1_1:
                q[0] += math.pi if q[0] < 0 else -math.pi  # 反向解
                q[1] = -q[1]  # 镜像解
        
        # 合法性检查
        if q[0] < self.q_min[0] or q[0] > self.q_max[0] or q[1] < self.q_min[1] or q[1] > self.q_max[1]:
            return q_NAN
        
        # 计算第三关节角度q3
        z_3 = V2P / L2P  # 第三关节Z轴
        Y_3 = np.cross(-V26, V2P)  # 临时Y轴方向
        y_3 = Y_3 / np.linalg.norm(Y_3)  # 归一化Y轴
        x_3 = np.cross(y_3, z_3)  # 正交补全X轴
        
        # 构建从基座到第二关节的旋转矩阵
        R_1 = np.array([[math.cos(q[0]), -math.sin(q[0]), 0],
                        [math.sin(q[0]), math.cos(q[0]), 0],
                        [0, 0, 1]])
        R_1_2 = np.array([[math.cos(q[1]), 0, math.sin(q[1])],
                          [0, 1, 0],
                          [-math.sin(q[1]), 0, math.cos(q[1])]])
        R_2 = R_1 @ R_1_2  # 组合旋转
        
        # 将x_3转换到第二关节坐标系
        x_2_3 = R_2.T @ x_3
        q[2] = math.atan2(x_2_3[2], x_2_3[0])  # 计算q3角度
        
        if q[2] <= self.q_min[2] or q[2] >= self.q_max[2]:
            return q_NAN
        
        # 计算第五关节角度q5
        # 计算从第四关节到第五关节的向量
        VH4 = p_2 + self.d3 * z_3 + self.a4 * x_3 - p_6 + self.d5 * z_5
        
        # 构建第五关节旋转矩阵
        c6 = math.cos(q[5])
        s6 = math.sin(q[5])
        R_5_6 = np.array([[c6, -s6, 0],
                          [0, 0, -1],
                          [s6, c6, 0]])
        R_5 = R_6 @ R_5_6.T
        
        # 将向量转换到第五关节坐标系
        V_5_H4 = R_5.T @ VH4
        q[4] = -math.atan2(V_5_H4[1], V_5_H4[0])  # 计算q5角度
        
        if q[4] <= self.q_min[4] or q[4] >= self.q_max[4]:
            return q_NAN
        
        return q  # 返回所有关节角度解

    def is_valid(self, q):
        """检查关节角度是否在合法范围内"""
        return np.all((q >= self.q_min) & (q <= self.q_max))