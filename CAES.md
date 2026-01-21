
import math
import matplotlib.pyplot as plt

import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题

class FlywheelModel:
    """飞轮储能系统模型"""

    def __init__(self,
                 radius=0.5,  # 飞轮半径 (m)
                 mass=500,  # 飞轮质量 (kg)
                 max_angular_vel=1000,  # 最大角速度 (rad/s)
                 moment_of_inertia=None,  # 转动惯量，如不提供则自动计算
                 efficiency=0.9,  # 整体效率
                 friction_coeff=0.01):  # 摩擦系数

        self.radius = radius
        self.mass = mass
        self.max_angular_vel = max_angular_vel
        self.efficiency = efficiency
        self.friction_coeff = friction_coeff

        # 如果未提供转动惯量，则假设为实心圆柱体计算
        if moment_of_inertia is None:
            self.moment_of_inertia = 0.5 * mass * radius ** 2  # 实心圆柱体转动惯量
        else:
            self.moment_of_inertia = moment_of_inertia

        self.current_angular_vel = 0  # 当前角速度
        self.energy_history = []  # 能量历史记录
        self.time_history = []  # 时间历史记录

    def calculate_kinetic_energy(self, angular_vel=None):
        """计算飞轮的动能 (J)"""
        if angular_vel is None:
            angular_vel = self.current_angular_vel
        return 0.5 * self.moment_of_inertia * angular_vel ** 2

    def calculate_max_energy(self):
        """计算最大存储能量 (J)"""
        return self.calculate_kinetic_energy(self.max_angular_vel)

    def charge(self, energy, time):
        """
        充电过程
        energy: 输入能量 (J)
        time: 充电时间 (s)
        """
        # 考虑效率后的实际可用能量
        effective_energy = energy * self.efficiency

        # 计算充电后的动能
        current_energy = self.calculate_kinetic_energy()
        new_energy = current_energy + effective_energy

        # 计算对应的角速度
        new_angular_vel = math.sqrt(2 * new_energy / self.moment_of_inertia)

        # 确保不超过最大角速度
        if new_angular_vel > self.max_angular_vel:
            new_angular_vel = self.max_angular_vel
            print(f"警告：已达到最大转速，实际充入能量少于预期")

        self.current_angular_vel = new_angular_vel

        # 记录历史数据
        if self.time_history:
            current_time = self.time_history[-1] + time
        else:
            current_time = time

        self.time_history.append(current_time)
        self.energy_history.append(self.calculate_kinetic_energy())

        return new_energy - current_energy  # 实际充入的能量

    def discharge(self, power, time):
        """
        放电过程
        power: 输出功率 (W)
        time: 放电时间 (s)
        """
        # 计算需要输出的能量
        required_energy = power * time

        # 考虑效率后的实际消耗能量
        energy_to_consume = required_energy / self.efficiency

        current_energy = self.calculate_kinetic_energy()

        # 检查是否有足够的能量
        if energy_to_consume > current_energy:
            # 只能释放所有可用能量
            energy_to_consume = current_energy
            required_energy = energy_to_consume * self.efficiency
            time = required_energy / power if power > 0 else 0
            print(f"警告：能量不足，只能维持 {time:.2f} 秒")

        # 计算放电后的能量和角速度
        new_energy = current_energy - energy_to_consume
        new_angular_vel = math.sqrt(2 * new_energy / self.moment_of_inertia) if new_energy > 0 else 0

        self.current_angular_vel = new_angular_vel

        # 记录历史数据
        if self.time_history:
            current_time = self.time_history[-1] + time
        else:
            current_time = time

        self.time_history.append(current_time)
        self.energy_history.append(new_energy)

        return required_energy  # 实际输出的能量

    def idle_loss(self, time):
        """
        计算闲置时的能量损耗
        time: 闲置时间 (s)
        """
        # 简化的摩擦损耗模型
        loss_factor = math.exp(-self.friction_coeff * time)
        self.current_angular_vel *= loss_factor

        current_energy = self.calculate_kinetic_energy()

        # 记录历史数据
        if self.time_history:
            current_time = self.time_history[-1] + time
        else:
            current_time = time

        self.time_history.append(current_time)
        self.energy_history.append(current_energy)

        return current_energy

    def get_state_of_charge(self):
        """获取当前荷电状态 (0-1)"""
        return self.calculate_kinetic_energy() / self.calculate_max_energy()

    def plot_energy_history(self):
        """绘制能量随时间变化的图表"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_history, self.energy_history)
        plt.title('飞轮储能系统能量变化')
        plt.xlabel('时间 (s)')
        plt.ylabel('能量 (J)')
        plt.grid(True)
        plt.show()


def simulate_flywheel_operation():
    """模拟飞轮储能系统的运行"""
    # 创建飞轮模型
    flywheel = FlywheelModel(
        radius=0.6,  # 半径0.6米
        mass=800,  # 质量800千克
        max_angular_vel=1200  # 最大角速度1200 rad/s
    )

    print(f"飞轮参数:")
    print(f"转动惯量: {flywheel.moment_of_inertia:.2f} kg·m²")
    print(f"最大储能: {flywheel.calculate_max_energy() / 1000:.2f} kJ")

    # 模拟充电过程 - 10秒内充入100kJ能量
    charged_energy = flywheel.charge(100000, 10)
    print(f"\n充电后能量: {flywheel.calculate_kinetic_energy() / 1000:.2f} kJ")
    print(f"实际充入能量: {charged_energy / 1000:.2f} kJ")

    # 模拟闲置5秒
    flywheel.idle_loss(5)
    print(f"闲置5秒后能量: {flywheel.calculate_kinetic_energy() / 1000:.2f} kJ")

    # 模拟放电过程 - 以5kW功率放电10秒
    discharged_energy = flywheel.discharge(5000, 10)
    print(f"放电10秒后能量: {flywheel.calculate_kinetic_energy() / 1000:.2f} kJ")
    print(f"实际释放能量: {discharged_energy / 1000:.2f} kJ")

    # 模拟再次充电
    flywheel.charge(200000, 20)
    print(f"再次充电后能量: {flywheel.calculate_kinetic_energy() / 1000:.2f} kJ")

    # 模拟长时间闲置
    flywheel.idle_loss(60)
    print(f"闲置60秒后能量: {flywheel.calculate_kinetic_energy() / 1000:.2f} kJ")

    # 绘制能量变化曲线
    flywheel.plot_energy_history()


if __name__ == "__main__":
    simulate_flywheel_operation()
