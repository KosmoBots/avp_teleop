# AVP Franka 遥操作项目

这是一个基于 ROS2 的 Franka 机器人遥操作项目，支持通过 Quest3 头显进行远程操作。

## 📋 系统要求

- Ubuntu 20.04 或更高版本
- Python 3.8
- ROS2
- NVIDIA GPU 驱动
- CUDA 12.04
- Quest3 头显

## 🚀 快速开始

### 1. 环境准备

#### 安装 ROS2
```bash
# 安装 ROS2 (请根据您的 Ubuntu 版本选择对应的安装命令)
# Ubuntu 20.04: ROS2 Foxy
# Ubuntu 22.04: ROS2 Humble
```

#### 安装 GPU 驱动和 CUDA
```bash
# 安装 NVIDIA GPU 驱动
sudo apt update
sudo apt install nvidia-driver-xxx  # 替换为适合您GPU的驱动版本

# 安装 CUDA 12.04
# 请从 NVIDIA 官网下载并安装 CUDA Toolkit 12.04
```

### 2. 创建 Python 环境

```bash
# 创建 conda 环境
conda create -n avp python=3.8
conda activate avp

# 安装 IK 解算包
conda install pinocchio==3.1.0 -c conda-forge

# 安装其他依赖包
pip install meshcat
pip install casadi
pip install -r requirements.txt
```

### 3. 网络配置

#### 获取本地 IP 地址
```bash
ifconfig | grep inet
```

#### Quest3 网络配置
- 确保 Quest3 和本地 PC 在同一 IP 网段下
- 配置 Quest3 的 WiFi 网络连接

### 4. 安全证书配置

```bash
# 安装证书工具
sudo apt install libnss3-tools

# 创建安全证书
mkcert -install && mkcert -cert-file cert.pem -key-file key.pem 192.168.*.* localhost 127.0.0.1

# 复制证书文件到项目目录
cp -r cert.pem key.pem avp_teleoperate_franka/teleop/
```

### 5. 防火墙配置

```bash
# 开放必要端口
sudo ufw allow 8012
```

## 📁 项目结构

```
avp_teleoperate_franka/
├── assets/                 # Franka 机器人模型文件
├── teleop/                 # 遥操作核心代码
│   ├── robot_control/      # 机器人控制模块
│   ├── open_television/    # 电视模块
│   ├── image_server/       # 图像服务器
│   └── utils/              # 工具函数
├── utils/                  # 工具和数据
└── requirements.txt        # Python 依赖
```

## 🔧 主要功能

- **机器人遥操作**: 通过 Quest3 头显远程控制 Franka 机器人
- **实时图像传输**: 支持实时视频流传输
- **IK 解算**: 集成 Pinocchio 进行逆运动学计算
- **安全通信**: 支持 HTTPS 安全通信

## 📦 依赖包

主要依赖包括：
- `pinocchio==3.1.0`: 机器人动力学和运动学库
- `torch==2.3.0`: PyTorch 深度学习框架
- `opencv_python==4.9.0.80`: 计算机视觉库
- `vuer==0.0.32rc7`: 3D 可视化库
- `aiohttp==3.9.5`: 异步 HTTP 服务器

完整依赖列表请参考 `requirements.txt`。

## 🚨 注意事项

1. 确保 Quest3 和 PC 在同一网络环境下
2. 正确配置防火墙端口
3. 安装正确版本的 CUDA 和 GPU 驱动
4. 使用 Python 3.8 环境

## 🚨 常见报错修复

### 1. 模块导入错误

#### ImportError: No module named 'pinocchio'
```bash
# 解决方案：确保先安装 pinocchio
conda install pinocchio==3.1.0 -c conda-forge
```

#### ImportError: No module named 'isaacgym'
```bash
# 解决方案：github下载isaacgym
# 安装顺序：
cd isaacgym/python
pip install -e .
```
#### IAttributeError: partially initialized module 'charset_normalizer' has no attribute 'md__mypyc' (most likely due to a circular import)
```bash
# 重装charset_normalizer
# 进入同一 conda 环境
pip uninstall -y charset_normalizer
pip install --no-cache-dir --force-reinstall charset_normalizer==3.2.0  # 或 requests 指定的兼容版本
```


## 📞 技术支持

如有问题，请按以下顺序检查：

1. **环境检查**
   - Python 版本是否为 3.8
   - conda 环境是否正确激活
   - 依赖包是否完整安装

2. **网络检查**
   - Quest3 和 PC 是否在同一网络
   - 防火墙端口是否开放
   - 证书文件是否正确配置

3. **硬件检查**
   - GPU 驱动是否正确安装
   - CUDA 版本是否匹配
   - 机器人连接是否正常

4. **日志检查**
   - 查看控制台错误信息
   - 检查系统日志
   - 确认文件路径正确性

## 📄 许可证

请查看 LICENSE 文件了解详细信息。