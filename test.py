import panda_py
# Panda hostname/IP and Desk login information of your robot
hostname = '192.168.3.100'
username = 'user'
password = 'password'
desk = panda_py.Desk(hostname, username, password)
desk.unlock()
desk.activate_fci()
from panda_py import libfranka
panda = panda_py.Panda(hostname)
gripper = libfranka.Gripper(hostname)

#获取状态
state = panda.get_state()
print("机械臂的关节状态：",state["q"])
#写入位置
panda.move_to_start()
pose = panda.get_pose()
pose[2,3] -= .1
q = panda_py.ik(pose)
print("写入的位置：",q)
panda.move_to_joint_position(q)
