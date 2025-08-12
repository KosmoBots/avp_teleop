import numpy as np


T_to_franka_left_wrist = np.array([[1, 0, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 1]])

T_to_franka_right_wrist = np.array([[1, 0, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, -1, 0, 0],
                                     [0, 0, 0, 1]])
                                     

T_to_franka_hand = np.array([[0, 0, 1, 0],
                              [-1,0, 0, 0],
                              [0, -1,0, 0],
                              [0, 0, 0, 1]])


T_robot_openxr = np.array([[0, 0, -1, 0],
                           [-1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]])


T_franka_openxr = np.array([[0, 0, -1, 0],
                            [-1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]])

const_head_vuer_mat = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 1.5],
                                [0, 0, 1, -0.2],
                                [0, 0, 0, 1]])


#initial position
const_right_wrist_vuer_mat = np.array(
                            [[-1,  0,  0,  0.05],
                            [ 0,  -1, 0,  1.5],
                            [0, 0, 1, -0.1],  
                            [0, 0, 0, 1.00]]
)
# initial position
const_left_wrist_vuer_mat = np.array([[1, 0, 0, -0.15],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, -0.3],
                                      [0, 0, 0, 1]])
vuer_right_wrist_mat = np.array( 
                            [[1,  0,  0,  0.35],
                            [ 0,  -1, 0,  1.5],
                            [0, 0, 1, -0.35],  
                            [0, 0, 0, 1]]
)
vuer_left_wrist_mat = np.array([[1, 0, 0, -0.5],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0.5],
                                [0, 0, 0, 1]])
# 在常量定义区域添加

T_to_franka_wrist = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])
