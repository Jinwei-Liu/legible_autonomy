# 实验系统使用说明

## 修改内容总结

### 1. PS5手柄控制
- **demo_shared.py** 已修改为使用PS5 DualSense手柄控制
- 使用左摇杆控制机器人
- 三角键(Triangle)重置位置和信念
- 删除了所有键盘控制代码

### 2. 可视化改进
- **高亮显示**：信念最大的目标会有特殊高亮效果
  - 黄色填充
  - 青色脉冲边框
  - 发光效果
- 信念柱状图中最高的也会高亮显示

### 3. 实验数据收集系统

新增 `experiment_data_collection.py` 文件，功能包括：

#### 参与者信息收集
- 开始时弹出对话框
- 输入：参与者代码、性别（下拉选择）、年龄

#### 实验流程
- 左上角显示参与者的目标意图
- **不显示**机器人的信念柱状图（保持盲测）
- 在路径中点（50%进度）时自动暂停
- 弹出问卷询问：
  1. 机器人是否理解了你的意图？（是/否）
  2. 你认为机器人的目标是什么？（选择）

#### 数据记录
每帧记录以下信息：
- 机器人位置
- 用户控制输入
- 机器人动作
- 执行的混合动作
- 机器人的信念分布
- β值（控制权重）

#### 实验设计
- 每个TASK_WEIGHT条件测试5次
- TASK_WEIGHT列表可在config.py中配置
- 默认：`[0.0, 0.03, 0.06, 0.1, 0.15]`
- 试次顺序随机化
- 目标也随机分配

## 配置参数

在 `config.py` 中新增：

```python
# 实验用的TASK_WEIGHT列表
TASK_WEIGHT_LIST = [0.0, 0.03, 0.06, 0.1, 0.15]

# 每个条件的重复次数
TRIALS_PER_CONDITION = 5

# 中点问卷触发阈值（0.5 = 50%进度）
MIDPOINT_THRESHOLD = 0.5
```

## 运行方法

### 1. 运行演示（Demo）
```bash
cd legible_autonomy
python demo_shared.py
```

**要求**：
- 连接PS5手柄
- 使用左摇杆控制
- 三角键重置

### 2. 运行实验数据收集
```bash
cd legible_autonomy
python experiment_data_collection.py
```

**流程**：
1. 输入参与者信息
2. 每个试次显示目标
3. 三角键开始试次
4. 左摇杆控制
5. 中点自动暂停并弹出问卷
6. 到达目标后自动进入下一试次

## 数据文件

数据保存在 `experiment_data/` 目录：

```
participant_P001_20260202_143025.json
```

### 数据结构

```json
{
  "participant_id": "P001",
  "gender": "Male",
  "age": 25,
  "experiment_date": "2026-02-02T14:30:25",
  "task_weight_list": [0.0, 0.03, 0.06, 0.1, 0.15],
  "trials_per_condition": 5,
  "trials": [
    {
      "trial_num": 0,
      "task_weight": 0.03,
      "target_goal_idx": 1,
      "target_goal": [650.0, 400.0],
      "start_time": "...",
      "end_time": "...",
      "questionnaire": {
        "understood": "Yes",
        "predicted_goal": "Goal 1 (Bottom)",
        "actual_goal": "Goal 1 (Bottom)",
        "progress": 0.52
      },
      "trajectory": [
        {
          "frame": 1,
          "time": 0.016,
          "robot_pos": [100.5, 300.2],
          "user_input": [0.5, 0.1],
          "robot_action": [1.0, 0.0],
          "executed": [0.8, 0.05],
          "beliefs": [0.45, 0.55],
          "beta": 0.6
        },
        ...
      ],
      "final_position": [648.2, 398.5]
    },
    ...
  ]
}
```

## PS5手柄按键映射

| 按键 | 功能 |
|------|------|
| 左摇杆 | 控制机器人移动 |
| 三角键 (Triangle) | 重置/开始试次 |

## 注意事项

1. **手柄连接**：实验前确保PS5手柄已连接
   - 有线：USB-C连接
   - 无线：蓝牙配对

2. **死区设置**：摇杆死区为0.15，避免漂移

3. **数据备份**：实验数据在每个试次后自动保存

4. **实验时长**：
   - 5个条件 × 5次重复 = 25个试次
   - 预计时长：20-30分钟

## 代码修改内容

### modified: config.py
- 添加 `TASK_WEIGHT_LIST`
- 添加 `TRIALS_PER_CONDITION`
- 添加 `MIDPOINT_THRESHOLD`

### modified: demo_shared.py
- 删除键盘控制代码
- 添加PS5手柄支持
- 高亮显示信念最大的目标
- 高亮显示最大信念柱状图

### modified: core/shared_autonomy.py
- 支持动态设置 `task_weight`
- `__init__` 新增 `task_weight` 参数

### new: experiment_data_collection.py
- 完整的实验数据收集系统
- 参与者信息对话框
- 中点问卷
- 自动数据记录和保存

## 疑难解答

### Q: 手柄无法识别
A: 
- 检查手柄是否充电
- 尝试重新连接
- 运行 `python -c "import pygame; pygame.init(); pygame.joystick.init(); print(pygame.joystick.get_count())"`

### Q: 对话框没有显示
A: 
- 确保安装了tkinter: `sudo apt-get install python3-tk` (Linux)
- macOS/Windows通常默认安装

### Q: 数据没有保存
A: 
- 检查 `experiment_data/` 目录权限
- 查看控制台错误信息
