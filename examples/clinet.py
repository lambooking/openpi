import numpy as np
from openpi_client import websocket_client_policy
from openpi_client import image_tools

# 连接到服务器
client = websocket_client_policy.WebsocketClientPolicy(
    host="你的服务器IP",
    port=8000
)

# 假设这些是从你的数据源获取的
def load_my_data():
    # 这里替换为你自己的数据加载代码
    exterior_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # 示例图像
    wrist_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)    # 示例图像
    joint_positions = np.random.rand(7)  # 示例关节位置
    gripper_position = np.random.rand(1)  # 示例夹爪位置
    return exterior_image, wrist_image, joint_positions, gripper_position

# 加载数据
exterior_image, wrist_image, joint_positions, gripper_position = load_my_data()

# 处理图像尺寸
exterior_image_resized = image_tools.resize_with_pad(exterior_image, 224, 224)
wrist_image_resized = image_tools.resize_with_pad(wrist_image, 224, 224)

# 创建观测数据
observation = {
    "observation/exterior_image_1_left": exterior_image_resized,
    "observation/wrist_image_left": wrist_image_resized,
    "observation/joint_position": joint_positions,
    "observation/gripper_position": gripper_position,
    "prompt": "pick up the object"
}

# 获取推理结果
action_chunk = client.infer(observation)["actions"]
print(f"Action chunk shape: {action_chunk.shape}")
print(f"First action: {action_chunk[0]}")