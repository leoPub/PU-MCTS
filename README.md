# PU-MCTS: Parallel Update Monte Carlo Tree Search

本仓库提供了**并行更新蒙特卡洛树搜索（PU-MCTS）**算法的核心实现代码。

需要说明的是，本仓库并未包含原项目庞大且复杂的全部工程代码，而是剥离出了**最关键的方法复现逻辑**。如果您对基础的 MCTS 算法有一定了解，本仓库提供的核心代码已完全足够支撑您对并行更新机制的理解与复现。

## 📁 核心文件说明

* `search_tree.py` & `mcts_search.py`
    **这是本项目的核心。** 这两个文件包含了 PU-MCTS 并行更新机制的完整实现逻辑。建议从这两个文件入手理解算法的底层逻辑。
* `exp_case_mcts_multi.py`
    主仿真循环文件。
    ⚠️ **注意**：该文件在历史版本迭代中保留了一些用于特定调试的参数和设置，在直接运行或借鉴时，请务必注意甄别并根据您的实际需求进行修改。

## 🔗 关联环境与奖励函数

本算法对应的仿真环境基础配置以及奖励函数（Reward Function）的公开实现，请参考配套仓库：
👉 [leoPub/diff_rew](https://github.com/leoPub/diff_rew)

## 🚗 双层仿真循环与环境重置 (Double-layer Simulation Loop)

在双层仿真循环的实现中，MCTS 的 Rollout 步骤需要频繁对仿真环境进行状态重置。在公开代码库的环境（基于 `env_highway_ctn.py`）基础上，本仓库对于环境重定位（`repos`）的具体实现如下。

该函数主要通过底层 API（如 SUMO/Flow 接口）直接控制车辆的位置和速度，以匹配 MCTS 树搜索节点的状态：

```python
    def repos(self, rollout_pos, rollout_lane, rollout_speed, veh_attr, infos):
        # 备选方案: self.k.simulation.kernel_api.simulation.loadState(state_file_path)

        for i, veh_id in enumerate(self.k.vehicle.get_ids()):
            self.k.vehicle.kernel_api.vehicle.moveTo(
                veh_id, 
                f'highway_0_{rollout_lane[i]}',
                max(0, rollout_pos[i] - 0.1)
            )
            self.k.vehicle.kernel_api.vehicle.setSpeed(veh_id, rollout_speed[i])
            # self.k.vehicle.kernel_api.vehicle.setAccel(veh_id, rollout_acc[i])

        self.k.simulation.simulation_step()
        self.k.update(reset=True)
        
        if self.sim_params.render:
            self.k.vehicle.update_vehicle_colors()
            
        # self.k.simulation.simulation_step()
        self.render()
        self.time_counter = 0
        self.exited_vehicles = []
        self.rl_infos = infos
        return None
```

## 📝 声明

本仓库目前公开的内容为原项目剥离出的核心逻辑。由于精力有限，目前提供的代码及上述说明即为所能提供的全部开源内容，希望对相关领域的研究者有所帮助。
