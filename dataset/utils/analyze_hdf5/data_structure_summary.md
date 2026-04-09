# 机器人操作数据结构概览（通用版）

## 基本信息
- **机器人类型**: X7S双臂机器人
- **演示数量**: 通常为1个演示 (demo_0)，可能有多个
- **时间步数**: **可变** (典型范围: 600-1000步，取决于任务复杂度)
- **文件大小**: 通常1-2 MB，取决于时间步数和场景复杂度
- **Action采样率 (FPS)**: **约13-14 fps** (基于实际数据分析)
  - 视频帧率: 30 fps
  - Action采样率约为视频帧率的 **44-47%**
  - 计算方法: `action_fps = action_steps / video_duration`

## 数据结构树（通用结构）

```
dataset_success.hdf5
└── data/
    └── demo_0/                    # 演示数据 (T步，T可变)
        │
        ├── actions                # 动作数据 (T, 21) - 固定21维
        │   └── 21维动作向量，范围通常: [-1.5, 1.0]
        │
        ├── checkpoints/           # [可选] 检查点数据
        │   ├── frame_index        # 检查点帧索引 (1, 1)
        │   └── frame_index_triggered  # 触发的检查点 (1, 1)
        │
        ├── eef/                   # 末端执行器姿态 (End Effector)
        │   ├── left_pose          # 左臂姿态 (T, 7) - [x,y,z,qx,qy,qz,qw]
        │   ├── right_pose        # 右臂姿态 (T, 7)
        │   ├── relative_left_pose  # 相对左臂姿态 (T, 7)
        │   └── relative_right_pose # 相对右臂姿态 (T, 7)
        │
        ├── initial_state/         # 初始状态 (t=0时刻，形状为(1, ...))
        │   ├── articulation/     # 关节物体（场景中的可动物体）
        │   │   ├── robot/        # 机器人初始状态 [固定存在]
        │   │   │   ├── joint_position (1, 25)  # 25个关节位置
        │   │   │   ├── joint_velocity (1, 25)  # 25个关节速度
        │   │   │   ├── root_pose (1, 7)        # 根位置和姿态
        │   │   │   └── root_velocity (1, 6)    # 根线速度和角速度
        │   │   └── [场景物体]/   # [可变] 场景中的关节物体
        │   │       └── 例如: television_main_group_1, cabinet_1_main_group_1
        │   │           ├── joint_position (1, N)  # N取决于物体类型
        │   │           ├── joint_velocity (1, N)
        │   │           ├── root_pose (1, 7)
        │   │           └── root_velocity (1, 6)
        │   │
        │   └── rigid_object/      # 刚体物体（可操作的物体）
        │       └── [物体名称]/    # [可变] 物体列表取决于任务
        │           └── 例如: plate, porcelain_mug, akita_black_bowl等
        │               ├── root_pose (1, 7)
        │               └── root_velocity (1, 6)
        │
        ├── joint_targets/        # 关节目标值
        │   ├── joint_pos_target  # 关节位置目标 (T, 50) - 50维
        │   ├── joint_vel_target  # 关节速度目标 (T, 50)
        │   └── joint_effort_target # 关节力矩目标 (T, 50)
        │
        ├── obs/                   # 观测数据
        │   ├── actions            # 上一时刻动作 (T, 21)
        │   ├── ee_pose            # 末端执行器姿态 (T, 2, 7) - [左臂, 右臂]
        │   ├── joint_pos          # 关节位置 (T, 25)
        │   ├── joint_pos_rel      # 相对关节位置 (T, 25)
        │   ├── joint_vel          # 关节速度 (T, 25)
        │   ├── joint_vel_rel      # 相对关节速度 (T, 25)
        │   │
        │   ├── raw_action/        # 原始动作数据
        │   │   ├── left_arm_abs    # 左臂绝对位置 (T, 7)
        │   │   ├── left_arm_delta  # 左臂增量 (T, 6)
        │   │   ├── left_gripper    # 左夹爪 (T,) - 范围[-1, 1]
        │   │   ├── right_arm_abs   # 右臂绝对位置 (T, 7)
        │   │   ├── right_arm_delta # 右臂增量 (T, 6)
        │   │   ├── right_gripper   # 右夹爪 (T,) - 通常为-1（关闭）
        │   │   ├── lpose_abs       # 左臂变换矩阵 (T, 4, 4)
        │   │   ├── rpose_abs       # 右臂变换矩阵 (T, 4, 4)
        │   │   ├── lpose_delta     # 左臂增量变换矩阵 (T, 4, 4)
        │   │   ├── rpose_delta     # 右臂增量变换矩阵 (T, 4, 4)
        │   │   ├── base_mode       # 基座模式 (T,) - 通常全为1
        │   │   ├── lgrasp          # 左抓取 (T,) - 与left_gripper相同
        │   │   ├── rgrasp          # 右抓取 (T,) - 与right_gripper相同
        │   │   └── [其他按钮/状态] # x_button, y_button等
        │   │
        │   └── raw_input/         # 原始输入数据 (T-1, ...) - 比actions少1步
        │       ├── abs_left_wrist_mat   # 绝对左腕变换矩阵 (T-1, 4, 4)
        │       ├── abs_right_wrist_mat  # 绝对右腕变换矩阵 (T-1, 4, 4)
        │       ├── head_mat             # 头部变换矩阵 (T-1, 4, 4)
        │       ├── rel_left_wrist_mat   # 相对左腕变换矩阵 (T-1, 4, 4)
        │       ├── rel_right_wrist_mat  # 相对右腕变换矩阵 (T-1, 4, 4)
        │       │
        │       ├── internal_state/      # 内部状态 (T-1, 1)
        │       │   ├── base_mode_flag
        │       │   ├── has_started
        │       │   ├── started
        │       │   ├── last_checkpoint_frame_idx
        │       │   └── [其他内部状态]
        │       │
        │       ├── left_controller_state/  # 左手控制器状态 (T-1, 1)
        │       │   ├── trigger              # 触发器 [0, 1]
        │       │   ├── thumbstick
        │       │   ├── thumbstick_x
        │       │   ├── thumbstick_y
        │       │   ├── squeeze
        │       │   ├── a_button
        │       │   └── b_button
        │       │
        │       └── right_controller_state/ # 右手控制器状态 (T-1, 1)
        │           └── [同left_controller_state结构]
        │
        ├── processed_actions       # 处理后的动作 (T, 23) - 23维
        │
        └── states/                # 状态数据 (每步的状态，形状为(T, ...))
            ├── articulation/
            │   ├── robot/        # 机器人状态 [固定存在]
            │   │   ├── joint_position (T, 25)
            │   │   ├── joint_velocity (T, 25)
            │   │   ├── root_pose (T, 7)
            │   │   └── root_velocity (T, 6)
            │   └── [场景物体]/   # [可变] 与initial_state中的articulation对应
            │       └── 结构同initial_state，但时间维度为T
            │
            └── rigid_object/      # 物体状态 [可变]
                └── [物体名称]/   # 与initial_state中的rigid_object对应
                    ├── root_pose (T, 7)
                    └── root_velocity (T, 6)
```

## 关键数据维度（固定维度）

| 数据类型 | 形状模式 | 说明 |
|---------|---------|------|
| **动作** | (T, 21) | 21维动作向量，T为时间步数 |
| **关节位置/速度** | (T, 25) | 25个关节（机器人固定） |
| **关节目标** | (T, 50) | 50维（包括机器人和其他关节） |
| **末端执行器姿态** | (T, 2, 7) | 左右双臂的姿态 [x,y,z,qx,qy,qz,qw] |
| **处理后的动作** | (T, 23) | 23维处理后的动作 |
| **原始输入** | (T-1, ...) | 比actions少1步 |
| **物体状态** | (T, 7) 或 (1, 7) | 7维姿态（位置3+四元数4） |
| **变换矩阵** | (T, 4, 4) 或 (T-1, 4, 4) | 4x4齐次变换矩阵 |

## 可变部分说明

### 1. 时间步数 (T)
- **变化范围**: 通常600-1000步
- **获取方法**: `actions.shape[0]` 或 `obs/joint_pos.shape[0]`
- **注意**: `raw_input`下的数据为 `T-1` 步
- **Action FPS分析**:
  - 基于实际样本分析（644步，视频时长48.2秒）:
    - Action FPS ≈ **13.36 fps**
    - 视频FPS = 30 fps
    - Action采样间隔 ≈ 0.075秒 (75毫秒)
    - Action与视频帧率比 ≈ 0.445 (约每2.25个视频帧对应1个action)
  - **说明**: Action数据以较低频率采样，可能用于控制循环或减少数据量

### 2. 场景物体 (articulation)
- **robot**: 固定存在
- **其他物体**: 取决于场景配置
  - 示例: `television_main_group_1`, `cabinet_1_main_group_1`
  - **获取方法**: 遍历 `initial_state/articulation/` 或 `states/articulation/`

### 3. 刚体物体 (rigid_object)
- **物体列表**: 完全取决于任务
  - 示例: `plate`, `porcelain_mug`, `akita_black_bowl`, `chocolate_pudding`等
  - **获取方法**: 遍历 `initial_state/rigid_object/` 或 `states/rigid_object/`

### 4. 检查点 (checkpoints)
- **存在性**: 可选，不是所有文件都有
- **检查方法**: `'checkpoints' in f['data/demo_0'].keys()`

### 5. 控制器状态
- **使用情况**: 不同演示中控制器使用情况不同
- **trigger均值**: 通常在0.4-0.5之间
- **按钮使用**: 有些演示使用y_button，有些不用

## 数据读取建议

### 1. 获取时间步数
```python
with h5py.File(filepath, 'r') as f:
    T = f['data/demo_0/actions'].shape[0]
```

### 2. 获取物体列表
```python
# 获取场景物体
scene_objects = list(f['data/demo_0/initial_state/articulation'].keys())
# 排除'robot'，得到场景物体

# 获取刚体物体
rigid_objects = list(f['data/demo_0/initial_state/rigid_object'].keys())
```

### 3. 检查可选数据
```python
demo_group = f['data/demo_0']
has_checkpoints = 'checkpoints' in demo_group.keys()
```

### 4. 读取状态数据
```python
# 机器人状态（固定）
robot_joint_pos = f['data/demo_0/states/articulation/robot/joint_position'][:]

# 物体状态（需要遍历）
for obj_name in rigid_objects:
    obj_pose = f[f'data/demo_0/states/rigid_object/{obj_name}/root_pose'][:]
```

## Action FPS详细分析

### 采样率说明
- **Action FPS**: 约 **13-14 fps** (每秒13-14个动作样本)
- **视频 FPS**: 30 fps
- **采样间隔**: 约75毫秒/步

### 计算方法
```python
import h5py
import subprocess

# 获取action步数
with h5py.File('dataset.hdf5', 'r') as f:
    action_steps = f['data/demo_0/actions'].shape[0]

# 获取视频时长（秒）
video_duration = float(subprocess.check_output([
    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
    '-of', 'default=noprint_wrappers=1:nokey=1', 'video.mp4'
]).strip())

# 计算action fps
action_fps = action_steps / video_duration
print(f"Action FPS: {action_fps:.2f}")
```

### 实际样本数据
| 样本 | Action步数 | 视频时长(秒) | Action FPS | 视频FPS |
|------|-----------|-------------|-----------|---------|
| 样本1 | 644 | 48.2 | 13.36 | 30 |
| 样本2 | 724 | ~54.3* | ~13.33* | 30 |

*注: 样本2的视频时长基于相同fps比例估算

### 为什么Action FPS低于视频FPS？
1. **控制循环频率**: 机器人控制循环可能以较低频率运行（如10-15 Hz）
2. **数据压缩**: 降低采样率可以减少数据存储量
3. **动作平滑性**: 较低频率的动作采样可能已经足够描述操作轨迹
4. **系统限制**: 可能是数据采集系统的实际采样能力

## 数据特点
1. **时间序列数据**: T个时间步，记录完整操作过程
2. **双臂操作**: 包含左右双臂的完整状态和动作
3. **多模态观测**: 包含关节状态、末端执行器姿态、原始输入等
4. **物体跟踪**: 记录了场景中所有物体的位置和速度变化
5. **原始输入**: 保留了VR控制器的原始输入数据
6. **场景可变**: 场景物体和任务物体根据任务不同而变化
7. **采样频率**: Action以约13-14 fps采样，低于视频的30 fps

