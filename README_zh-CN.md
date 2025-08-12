#安装ros2
#安装gpu驱动
#安装cuda 12.04
#创建avp环境
conda create -n avp python=3.8
conda activate avp
#conda 安装 ik解算包
conda install pinocchio==3.1.0 -c conda-forge
#安装依赖包
pip install meshcat
pip install casadi
pip install -r requirements.txt
#配置网络
ifconfig | grep inet #获取本地IP地址
quest3配置wifi网络，需要和本地PC在同一IP范围下
#创建安全证书
sudo apt install libnss3-tools
mkcert -install && mkcert -cert-file cert.pem -key-file key.pem 192.168.*.* localhost 127.0.0.1
cp -r cert.pem key.pem  avp_teleoperate_franka_teleop/
#开放端口
sudo ufw allow 8012