## 环境配置
CUDA 测试过11.3/11.7/11.8

transforms3d

非夕RDK已包含在项目内

根目录下运行'pip install -e . --no-build-isolation' ([curobo官方安装教程](https://curobo.org/source/getting_started/1_install_instructions.html))

---------------------------------
---------------------------------
## 运行
python examples/bi-teleoperation/receiver.py

构建BiFlexivControl类对象，设置各个ip

构建Receiver类对象，传入BiFlexivControl类对象

运行Receiver类对象的run函数

详见receiver.py main函数

---------------------------------
---------------------------------
## 操作指南

手柄左键各自控制对应机械臂是否开启跟随

任意手柄右键让双臂系统go home

扳机线性控制夹爪(响应很慢)

---------------------------------
---------------------------------
## 文档
---------------------------------
### receiver.py
Receiver类：(基于UDP通信)

1. 初始化：

    ```python
    def __init__(self, controller, local_ip, port):
    ```

    参数：

    - controller：双臂控制的类(BiFlexivController)
    - local_ip：接收IP
    - port：接收端口


2. 运行：(Thread类固有函数)

    ```python
    def run(self):
    ```

    参数：无

    返回：无
---------------------------------
### bi_flexiv_controller.py
get_custom_world_model方法:

对双臂桌面以及铁笼碰撞体声明

BiFlexivController类：

1. 初始化：
    ```python
    def __init__(self, local_ip, left_robot_ip, right_robot_ip, left_origin_offset,right_origin_offset):
    ```

    参数：

    - local_ip：控制端PC的IP
    - left_robot_ip：左臂机械臂IP
    - right_robot_ip：右臂机械臂IP
    - left_origin_offset： 左臂机械臂相对原点的偏移量
    - right_origin_offset： 右臂机械臂相对原点的偏移量


所需curobo配置文件：

"src/curobo/content/configs/robot/dual_flexiv.yml"

"src/curobo/content/configs/robot/dual_flexiv_bigger.yml"

---------------------------------
### flexiv_controller.py
FlexivController类：

1. 初始化：
    ```python
    def __init__(self, world_model, local_ip, robot_ip, origin_offset):
    ```

    参数：

    - world_model：世界碰撞体(即bi_flexiv_controller.py中的get_custom_world_model方法)
    - local_ip：控制端PC的IP
    - robot_ip：机械臂IP
    - origin_offset： 机械臂相对原点的偏移量


所需curobo配置文件：

"src/curobo/content/configs/robot/flexiv.yml"

---------------------------------
### UniController.py
弃用

---------------------------------

## Unity(VR端)
向接收端ip端口发送数据，格式如下：

```json
{
    "leftHand":{
        "pos":[x,y,z],
        "quat":[w,x,y,z],
        "squeeze":float, #扳机值
    },
    "rightHand":{
        "pos":[x,y,z],
        "quat":[w,x,y,z],
        "squeeze":float, #扳机值
    },
}
```
传输数据参考系为unity左手坐标系。

VR房间标定时，因为机械臂面朝东，因此VR头显面朝东标定，此时Unity左手坐标系z为东，y为上，curobo右手坐标系x为东，z为上。

