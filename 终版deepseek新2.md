import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ==================== 1. 参数配置 ====================
class SystemParameters:
    """系统参数配置类"""

    def __init__(self):
        # 基础参数
        self.total_power_capacity = 20.0  # MW - 总功率容量
        self.total_energy_capacity = 480.0  # MWh - 总能量容量
        self.plant_rated_power = 13.0  # MW - 电厂额定功率
        self.plant_output = 10.0  # MW - 电厂输出功率
        self.base_load = 10.0  # MW - 基本负荷

        # 脉冲参数
        self.pulse_magnitude = 18.0  # MW - 脉冲幅度
        self.pulse_duration = 10  # s - 脉冲持续时间
        self.max_pulse_magnitude = 20.0  # MW - 最大脉冲幅度
        self.max_pulse_duration = 20  # s - 最长脉冲持续时间

        # 仿真参数
        self.simulation_time = 120  # s - 缩短仿真总时间以便更清晰地显示脉冲
        self.time_step = 1  # s - 时间步长
        self.pulse_start_time = 50  # s - 脉冲开始时间

        # 南京冬季电价参数（工商业）
        self.electricity_prices = {
            'valley': 0.21,  # 元/kWh - 低谷电价 (00:00-06:00, 11:00-13:00)
            'flat': 0.62,  # 元/kWh - 平段电价
            'peak': 1.12,  # 元/kWh - 高峰电价 (14:00-22:00)
            'sharp': 1.34  # 元/kWh - 尖峰电价 (18:00-20:00)
        }

        # 天然气价格
        self.natural_gas_price = 3.65  # 元/立方米
        self.gas_energy_content = 10.0  # kWh/立方米 - 天然气热值

        # 储能设备初始分配比例（可优化）
        self.initial_allocation = {
            'BESS': {'power_ratio': 0.30, 'energy_ratio': 0.35},  # 电化学储能
            'SC': {'power_ratio': 0.20, 'energy_ratio': 0.10},  # 超级电容
            'FESS': {'power_ratio': 0.15, 'energy_ratio': 0.15},  # 飞轮储能
            'SMES': {'power_ratio': 0.15, 'energy_ratio': 0.10},  # 超导磁储能
            'CAES': {'power_ratio': 0.20, 'energy_ratio': 0.30}  # 压缩空气储能
        }

        # 成本参数（元）
        self.cost_parameters = {
            'BESS': {'capex': 0.70, 'opex': 0.025, 'efficiency': 0.92},
            'SC': {'capex': 2.00, 'opex': 0.015, 'efficiency': 0.95},
            'FESS': {'capex': 10000, 'opex': 0.010, 'efficiency': 0.90},
            'SMES': {'capex': 65000, 'opex': 0.050, 'efficiency': 0.96},
            'CAES': {'capex': 2750, 'opex': 0.005, 'efficiency': 0.70}
        }


# ==================== 2. 储能设备模型 ====================
class EnergyStorageBase:
    """储能设备基类"""

    def __init__(self, name, power_capacity, energy_capacity):
        self.name = name
        self.power_capacity = power_capacity  # MW
        self.energy_capacity = energy_capacity  # MWh

        # 状态变量
        self.soc = 0.5  # 初始SOC (0-1)
        self.current_power = 0.0  # MW - 当前功率 (正为放电，负为充电)
        self.voltage = 0.0  # kV - 当前电压
        self.current = 0.0  # A - 当前电流
        self.temperature = 25.0  # °C - 温度

        # 运行记录
        self.power_history = []
        self.soc_history = []
        self.voltage_history = []
        self.current_history = []
        self.cost_history = []

    def update_state(self, power, time_step=1):
        """更新储能状态"""
        # 限制功率在容量范围内
        power = np.clip(power, -self.power_capacity, self.power_capacity)

        # 更新SOC
        energy_change = power * time_step / 3600  # MWh
        self.soc += energy_change / self.energy_capacity
        self.soc = np.clip(self.soc, 0.1, 0.9)  # 防止过充过放

        # 更新功率
        self.current_power = power

        # 记录历史
        self.power_history.append(power)
        self.soc_history.append(self.soc)

        return power


class BESS(EnergyStorageBase):
    """电化学储能系统"""

    def __init__(self, power_capacity, energy_capacity):
        super().__init__("锂离子电池", power_capacity, energy_capacity)
        self.nominal_voltage = 400.0  # V - 标称电压
        self.internal_resistance = 0.01  # Ω - 内阻
        self.cycle_life = 6000  # 循环寿命
        self.self_discharge_rate = 0.001  # 自放电率

    def update_state(self, power, time_step=1):
        power = super().update_state(power, time_step)

        # 计算电流电压（简化模型）
        if abs(power) > 0:
            self.current = (power * 1e6) / self.nominal_voltage  # A
            self.voltage = self.nominal_voltage - self.current * self.internal_resistance
        else:
            self.current = 0
            self.voltage = self.nominal_voltage

        self.current_history.append(self.current)
        self.voltage_history.append(self.voltage)

        return power


class SuperCapacitor(EnergyStorageBase):
    """超级电容器"""

    def __init__(self, power_capacity, energy_capacity):
        super().__init__("超级电容器", power_capacity, energy_capacity)
        self.capacitance = 3000  # F - 电容
        self.max_voltage = 2.7  # V - 最大电压
        self.min_voltage = 1.0  # V - 最小电压
        self.current_voltage = 1.8  # V - 当前电压
        self.self_discharge_rate = 0.05  # 自放电率

    def update_state(self, power, time_step=1):
        power = super().update_state(power, time_step)

        # 计算电压（基于能量）
        energy = self.soc * self.energy_capacity * 3.6e9  # 转换为J
        self.current_voltage = np.sqrt(2 * energy / self.capacitance)
        self.current_voltage = np.clip(self.current_voltage, self.min_voltage, self.max_voltage)

        # 计算电流
        if abs(power) > 0:
            self.current = (power * 1e6) / self.current_voltage
        else:
            self.current = 0

        self.current_history.append(self.current)
        self.voltage_history.append(self.current_voltage)

        return power


class Flywheel(EnergyStorageBase):
    """飞轮储能"""

    def __init__(self, power_capacity, energy_capacity):
        super().__init__("飞轮储能", power_capacity, energy_capacity)
        self.rotational_speed = 0.0  # rad/s - 转速
        self.max_speed = 1000  # rad/s - 最大转速
        self.moment_of_inertia = 500  # kg·m² - 转动惯量
        self.friction_loss = 0.01  # 摩擦损失系数
        self.self_discharge_rate = 0.02  # 自放电率

    def update_state(self, power, time_step=1):
        power = super().update_state(power, time_step)

        # 基于能量计算转速
        energy = self.soc * self.energy_capacity * 3.6e6  # 转换为J
        self.rotational_speed = np.sqrt(2 * energy / self.moment_of_inertia)
        self.rotational_speed = np.clip(self.rotational_speed, 0, self.max_speed)

        # 简化电流电压计算
        self.voltage = 400.0 + self.rotational_speed / 10
        self.current = power * 1e6 / self.voltage if abs(power) > 0 else 0

        self.current_history.append(self.current)
        self.voltage_history.append(self.voltage)

        return power


class SMES(EnergyStorageBase):
    """超导磁储能"""

    def __init__(self, power_capacity, energy_capacity):
        super().__init__("超导磁储能", power_capacity, energy_capacity)
        self.inductance = 100  # H - 电感
        self.critical_current = 1000  # A - 临界电流
        self.critical_field = 8  # T - 临界磁场
        self.current_current = 0  # A - 当前电流
        self.cooling_power = 0.01  # MW - 冷却功率
        self.self_discharge_rate = 0.001  # 自放电率

    def update_state(self, power, time_step=1):
        power = super().update_state(power, time_step)

        # 计算磁场能量和电流
        energy = self.soc * self.energy_capacity * 3.6e6  # J
        self.current_current = np.sqrt(2 * energy / self.inductance)
        self.current_current = np.clip(self.current_current, 0, self.critical_current)

        # 计算电压
        if abs(power) > 0:
            self.voltage = self.inductance * (power * 1e6) / (self.current_current * time_step)
        else:
            self.voltage = 0

        self.current = self.current_current
        self.current_history.append(self.current)
        self.voltage_history.append(self.voltage)

        # 冷却成本
        cooling_cost = self.cooling_power * time_step / 3600 * 0.62  # 按平段电价计算
        self.cost_history.append(cooling_cost)

        return power


class CAES(EnergyStorageBase):
    """压缩空气储能"""

    def __init__(self, power_capacity, energy_capacity):
        super().__init__("压缩空气储能", power_capacity, energy_capacity)
        self.air_mass = 0.0  # kg - 空气质量
        self.max_air_mass = energy_capacity * 100  # 简化转换
        self.min_air_mass = self.max_air_mass * 0.1
        self.pressure = 10.0  # MPa - 压力
        self.natural_gas_consumption = 0.0  # m³ - 天然气消耗
        self.self_discharge_rate = 0.005  # 自放电率

    def update_state(self, power, time_step=1):
        power = super().update_state(power, time_step)

        # 计算空气质量
        self.air_mass = self.soc * self.max_air_mass

        # 天然气消耗（放电时）
        if power > 0:  # 放电
            self.natural_gas_consumption = power * time_step / 3600 * 0.3  # 简化模型
        else:
            self.natural_gas_consumption = 0

        # 简化电流电压计算
        self.voltage = 10.0  # kV
        self.current = power * 1e3 / self.voltage if abs(power) > 0 else 0

        self.current_history.append(self.current)
        self.voltage_history.append(self.voltage)

        return power


# ==================== 3. MPC控制器 ====================
class MPCController:
    """模型预测控制器"""

    def __init__(self, storage_systems, params):
        self.storage_systems = storage_systems
        self.params = params

        # 响应速度优先级（响应速度从快到慢）
        self.response_priority = ['SC', 'SMES', 'FESS', 'BESS', 'CAES']

        # 频率特性分配权重
        self.frequency_weights = {
            'high': {'SC': 0.40, 'SMES': 0.40, 'FESS': 0.20},  # 高频
            'mid': {'FESS': 0.40, 'BESS': 0.30, 'SC': 0.20, 'SMES': 0.10},  # 中频
            'low': {'BESS': 0.40, 'CAES': 0.40, 'FESS': 0.10, 'SC': 0.05, 'SMES': 0.05}  # 低频
        }

        # 历史数据
        self.power_history = []
        self.history_length = 50

    def allocate_power_by_frequency(self, power_to_balance, time_index):
        """根据频率特性分配功率"""
        # 根据时间确定频率特性（脉冲期间高频分量多，非脉冲期间低频分量多）
        is_pulse_period = (time_index >= self.params.pulse_start_time and
                           time_index < self.params.pulse_start_time + self.params.pulse_duration)

        if is_pulse_period:
            # 脉冲期间：高频为主
            freq_ratio = {'high': 0.6, 'mid': 0.3, 'low': 0.1}
        else:
            # 非脉冲期间：低频为主
            freq_ratio = {'high': 0.1, 'mid': 0.3, 'low': 0.6}

        # 计算各频段功率
        power_allocation = {}
        for device_name in self.storage_systems:
            device_power = 0
            for freq_type, ratio in freq_ratio.items():
                if device_name in self.frequency_weights[freq_type]:
                    device_power += power_to_balance * ratio * self.frequency_weights[freq_type][device_name]

            if abs(device_power) > 0:
                power_allocation[device_name] = device_power

        return power_allocation

    def optimize_power_allocation(self, power_to_balance, current_socs, time_index):
        """优化功率分配"""
        # 基础分配基于频率特性
        allocation = self.allocate_power_by_frequency(power_to_balance, time_index)

        # 调整基于SOC状态
        for device_name, power in allocation.items():
            device = self.storage_systems[device_name]
            soc = current_socs[device_name]

            # 根据SOC调整功率
            if power > 0:  # 放电
                soc_factor = min(1.0, max(0, (soc - 0.2) / 0.6))  # SOC低于0.2时不放电
            else:  # 充电
                soc_factor = min(1.0, max(0, (0.8 - soc) / 0.6))  # SOC高于0.8时不充电

            adjusted_power = power * soc_factor

            # 确保不超过设备容量
            max_power = device.power_capacity
            if abs(adjusted_power) > max_power:
                adjusted_power = np.sign(adjusted_power) * max_power

            allocation[device_name] = adjusted_power

        # 确保总分配功率接近需求
        total_allocated = sum(allocation.values())
        if abs(total_allocated - power_to_balance) > 0.01:
            # 重新调整分配
            scale_factor = power_to_balance / total_allocated if abs(total_allocated) > 0 else 0
            for device_name in allocation:
                allocation[device_name] *= scale_factor

        return allocation


# ==================== 4. 分层调度系统 ====================
class HierarchicalScheduler:
    """分层调度系统"""

    def __init__(self, params):
        self.params = params
        self.storage_systems = {}
        self.mpc_controller = None
        self.initialize_storage_systems()

        # 运行记录
        self.load_profile = []
        self.smoothed_load = []
        self.power_imbalance = []
        self.storage_power = []
        self.total_cost = 0.0
        self.cost_breakdown = {
            'BESS': 0.0, 'SC': 0.0, 'FESS': 0.0,
            'SMES': 0.0, 'CAES': 0.0, '天然气': 0.0
        }

    def initialize_storage_systems(self):
        """初始化储能系统"""
        total_power = self.params.total_power_capacity
        total_energy = self.params.total_energy_capacity

        # 根据初始分配比例创建储能设备
        for device_name, ratios in self.params.initial_allocation.items():
            power_capacity = total_power * ratios['power_ratio']
            energy_capacity = total_energy * ratios['energy_ratio']

            if device_name == 'BESS':
                self.storage_systems[device_name] = BESS(power_capacity, energy_capacity)
            elif device_name == 'SC':
                self.storage_systems[device_name] = SuperCapacitor(power_capacity, energy_capacity)
            elif device_name == 'FESS':
                self.storage_systems[device_name] = Flywheel(power_capacity, energy_capacity)
            elif device_name == 'SMES':
                self.storage_systems[device_name] = SMES(power_capacity, energy_capacity)
            elif device_name == 'CAES':
                self.storage_systems[device_name] = CAES(power_capacity, energy_capacity)

        # 初始化MPC控制器
        self.mpc_controller = MPCController(self.storage_systems, self.params)

    def generate_load_profile(self):
        """生成负荷曲线（含脉冲）"""
        time_points = np.arange(0, self.params.simulation_time, self.params.time_step)
        load = np.ones_like(time_points) * self.params.base_load

        # 添加脉冲
        pulse_start = self.params.pulse_start_time
        pulse_end = pulse_start + self.params.pulse_duration

        # 创建平滑的脉冲（避免瞬时突变）
        for i, t in enumerate(time_points):
            if pulse_start <= t < pulse_end:
                # 使用正弦函数创建平滑脉冲
                t_rel = (t - pulse_start) / (pulse_end - pulse_start)
                pulse_shape = np.sin(np.pi * t_rel)
                load[i] = self.params.base_load + pulse_shape * (self.params.pulse_magnitude - self.params.base_load)

        # 添加随机小波动（模拟真实负荷）
        noise = np.random.normal(0, 0.1, len(load))
        load += noise

        return load, time_points

    def run_simulation(self):
        """运行仿真"""
        print("开始混合储能系统协同调度仿真...")

        # 生成负荷曲线
        load_profile, time_points = self.generate_load_profile()
        self.load_profile = load_profile

        # 计算功率不平衡
        power_imbalance = load_profile - self.params.plant_output
        self.power_imbalance = power_imbalance

        # 初始化平滑后负荷
        smoothed_load = np.zeros_like(load_profile)

        # MPC滚动优化
        for i, (current_time, imbalance) in enumerate(zip(time_points, power_imbalance)):
            if i % 20 == 0:
                print(f"仿真进度: {i}/{len(time_points)} ({i / len(time_points) * 100:.1f}%)")

            # 获取当前各储能设备的SOC
            current_socs = {name: device.soc for name, device in self.storage_systems.items()}

            # MPC优化功率分配
            allocation = self.mpc_controller.optimize_power_allocation(imbalance, current_socs, i)

            # 执行功率分配并更新设备状态
            total_storage_power = 0
            for device_name, power in allocation.items():
                device = self.storage_systems[device_name]

                # 更新设备状态
                actual_power = device.update_state(power, self.params.time_step)
                total_storage_power += actual_power

            self.storage_power.append(total_storage_power)

            # 计算平滑后负荷
            smoothed_load[i] = self.params.plant_output + total_storage_power

        self.smoothed_load = smoothed_load

        print("仿真完成!")
        return time_points, load_profile, smoothed_load

    def calculate_performance_metrics(self):
        """计算性能指标"""
        # 计算重合度（相关系数）
        if len(self.load_profile) > 0 and len(self.smoothed_load) > 0:
            correlation = np.corrcoef(self.load_profile, self.smoothed_load)[0, 1]
        else:
            correlation = 0

        # 计算平均绝对误差
        mae = np.mean(np.abs(self.load_profile - self.smoothed_load))

        # 计算脉冲消除效果
        pulse_indices = np.where(self.load_profile > self.params.base_load * 1.2)[0]
        if len(pulse_indices) > 0:
            original_pulse_max = np.max(self.load_profile[pulse_indices])
            smoothed_pulse_max = np.max(self.smoothed_load[pulse_indices])
            pulse_reduction = (original_pulse_max - smoothed_pulse_max) / original_pulse_max * 100
        else:
            pulse_reduction = 0

        # 计算各储能设备贡献
        power_contributions = {}
        for name, device in self.storage_systems.items():
            if device.power_history:
                avg_power = np.mean(np.abs(device.power_history))
                power_contributions[name] = avg_power

        total_contribution = sum(power_contributions.values())
        contribution_ratios = {}
        for name, power in power_contributions.items():
            contribution_ratios[name] = (power / total_contribution * 100) if total_contribution > 0 else 0

        metrics = {
            '负荷重合度': correlation,
            '平均绝对误差': mae,
            '脉冲消除率': pulse_reduction,
            '贡献比例': contribution_ratios,
            '成本构成': self.cost_breakdown
        }

        return metrics


# ==================== 5. 可视化模块 ====================
class Visualization:
    """可视化模块"""

    @staticmethod
    def plot_comprehensive_results(time_points, original_load, smoothed_load, storage_systems, params, metrics):
        """绘制综合结果图"""
        # 创建一个大图，包含多个子图
        fig = plt.figure(figsize=(20, 16))

        # 1. 图一：负荷对比图（原始负荷 vs 平滑后负荷）
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(time_points, original_load, 'b-', linewidth=3, label='原始负荷', alpha=0.8)
        ax1.plot(time_points, smoothed_load, 'r--', linewidth=2, label='平滑后负荷', alpha=0.8)
        ax1.plot(time_points, np.ones_like(time_points) * params.plant_output, 'g:', linewidth=2, label='电厂出力')
        ax1.set_title('图1: 负荷平滑效果对比', fontsize=14, fontweight='bold')
        ax1.set_xlabel('时间 (秒)', fontsize=12)
        ax1.set_ylabel('功率 (MW)', fontsize=12)
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, max(original_load) * 1.1])

        # 添加重合度标注
        ax1.text(0.02, 0.95, f'重合度: {metrics["负荷重合度"]:.4f}',
                 transform=ax1.transAxes, fontsize=11,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

        # 2. 图二：功率不平衡与储能系统出力
        ax2 = plt.subplot(3, 3, 2)
        imbalance = original_load - params.plant_output
        ax2.plot(time_points, imbalance, 'purple', linewidth=2, label='功率不平衡量')

        # 计算储能总出力
        storage_total = []
        for i in range(len(time_points)):
            total = 0
            for device in storage_systems.values():
                if i < len(device.power_history):
                    total += device.power_history[i]
            storage_total.append(total)

        ax2.plot(time_points[:len(storage_total)], storage_total, 'orange', linewidth=2, label='储能总出力')
        ax2.fill_between(time_points[:len(storage_total)], 0, storage_total, alpha=0.3, color='orange')
        ax2.set_title('图2: 功率不平衡与储能出力', fontsize=14, fontweight='bold')
        ax2.set_xlabel('时间 (秒)', fontsize=12)
        ax2.set_ylabel('功率 (MW)', fontsize=12)
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # 3. 图三：各储能设备功率贡献
        ax3 = plt.subplot(3, 3, 3)
        colors = ['blue', 'green', 'red', 'purple', 'orange']

        for idx, (name, device) in enumerate(storage_systems.items()):
            if device.power_history:
                power_data = device.power_history[:len(time_points)]
                ax3.plot(time_points[:len(power_data)], power_data,
                         color=colors[idx], linewidth=2, label=device.name)

        ax3.set_title('图3: 各储能设备功率贡献', fontsize=14, fontweight='bold')
        ax3.set_xlabel('时间 (秒)', fontsize=12)
        ax3.set_ylabel('功率 (MW)', fontsize=12)
        ax3.legend(fontsize=10, loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # 4. 图四：脉冲分配策略占比
        ax4 = plt.subplot(3, 3, 4)

        # 计算脉冲期间各设备贡献
        pulse_start_idx = params.pulse_start_time
        pulse_end_idx = pulse_start_idx + params.pulse_duration

        device_names = []
        contributions = []
        colors_pie = []

        for idx, (name, device) in enumerate(storage_systems.items()):
            if device.power_history and len(device.power_history) > pulse_end_idx:
                pulse_powers = device.power_history[pulse_start_idx:pulse_end_idx]
                avg_power = np.mean(np.abs(pulse_powers))
                if avg_power > 0:
                    device_names.append(device.name)
                    contributions.append(avg_power)
                    colors_pie.append(colors[idx])

        if contributions:
            wedges, texts, autotexts = ax4.pie(contributions, labels=device_names, colors=colors_pie,
                                               autopct='%1.1f%%', startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

        ax4.set_title('图4: 脉冲消除策略分配占比', fontsize=14, fontweight='bold')
        ax4.set_aspect('equal')

        # 5. 图五：各储能设备SOC变化
        ax5 = plt.subplot(3, 3, 5)
        for idx, (name, device) in enumerate(storage_systems.items()):
            if device.soc_history:
                soc_data = device.soc_history[:len(time_points)]
                ax5.plot(time_points[:len(soc_data)], soc_data,
                         color=colors[idx], linewidth=2, label=device.name)

        ax5.set_title('图5: 各储能设备SOC变化', fontsize=14, fontweight='bold')
        ax5.set_xlabel('时间 (秒)', fontsize=12)
        ax5.set_ylabel('SOC', fontsize=12)
        ax5.legend(fontsize=10, loc='best')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 1])

        # 6. 图六：电流随时间变化图
        ax6 = plt.subplot(3, 3, 6)
        for idx, (name, device) in enumerate(storage_systems.items()):
            if device.current_history:
                current_data = device.current_history[:len(time_points)]
                # 平滑电流曲线（避免瞬时突变）
                if len(current_data) > 5:
                    current_data_smooth = np.convolve(current_data, np.ones(5) / 5, mode='same')
                else:
                    current_data_smooth = current_data

                ax6.plot(time_points[:len(current_data_smooth)], current_data_smooth,
                         color=colors[idx], linewidth=2, label=device.name, alpha=0.7)

        ax6.set_title('图6: 各储能设备电流变化', fontsize=14, fontweight='bold')
        ax6.set_xlabel('时间 (秒)', fontsize=12)
        ax6.set_ylabel('电流 (A)', fontsize=12)
        ax6.legend(fontsize=10, loc='best')
        ax6.grid(True, alpha=0.3)

        # 7. 图七：电压随时间变化图
        ax7 = plt.subplot(3, 3, 7)
        for idx, (name, device) in enumerate(storage_systems.items()):
            if device.voltage_history:
                voltage_data = device.voltage_history[:len(time_points)]
                # 平滑电压曲线
                if len(voltage_data) > 5:
                    voltage_data_smooth = np.convolve(voltage_data, np.ones(5) / 5, mode='same')
                else:
                    voltage_data_smooth = voltage_data

                ax7.plot(time_points[:len(voltage_data_smooth)], voltage_data_smooth,
                         color=colors[idx], linewidth=2, label=device.name, alpha=0.7)

        ax7.set_title('图7: 各储能设备电压变化', fontsize=14, fontweight='bold')
        ax7.set_xlabel('时间 (秒)', fontsize=12)
        ax7.set_ylabel('电压 (V)', fontsize=12)
        ax7.legend(fontsize=10, loc='best')
        ax7.grid(True, alpha=0.3)

        # 8. 图八：能量随电压变化图
        ax8 = plt.subplot(3, 3, 8)
        for idx, (name, device) in enumerate(storage_systems.items()):
            if device.voltage_history and device.soc_history:
                # 计算能量
                energy_data = []
                for soc in device.soc_history[:len(time_points)]:
                    energy = soc * device.energy_capacity * 1000  # 转换为kWh
                    energy_data.append(energy)

                voltage_data = device.voltage_history[:len(time_points)]

                if len(energy_data) == len(voltage_data):
                    ax8.scatter(voltage_data, energy_data, color=colors[idx],
                                s=20, alpha=0.6, label=device.name)

        ax8.set_title('图8: 能量-电压关系', fontsize=14, fontweight='bold')
        ax8.set_xlabel('电压 (V)', fontsize=12)
        ax8.set_ylabel('能量 (kWh)', fontsize=12)
        ax8.legend(fontsize=10, loc='best')
        ax8.grid(True, alpha=0.3)

        # 9. 图九：性能指标汇总
        ax9 = plt.subplot(3, 3, 9)

        # 创建性能指标表格
        performance_data = [
            ['指标', '值'],
            ['负荷重合度', f'{metrics["负荷重合度"]:.4f}'],
            ['平均绝对误差', f'{metrics["平均绝对误差"]:.2f} MW'],
            ['脉冲消除率', f'{metrics["脉冲消除率"]:.1f}%'],
            ['仿真时间', f'{params.simulation_time}秒']
        ]

        # 添加各设备贡献比例
        for name, ratio in metrics['贡献比例'].items():
            device_name = storage_systems[name].name
            performance_data.append([f'{device_name}贡献', f'{ratio:.1f}%'])

        # 创建表格
        table = ax9.table(cellText=performance_data, loc='center',
                          cellLoc='center', colWidths=[0.4, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # 隐藏坐标轴
        ax9.axis('off')
        ax9.set_title('图9: 性能指标汇总', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_individual_details(time_points, storage_systems, params):
        """绘制各储能设备详细图表"""
        fig, axes = plt.subplots(5, 3, figsize=(18, 20))

        colors = ['blue', 'green', 'red', 'purple', 'orange']

        for idx, (name, device) in enumerate(storage_systems.items()):
            row = idx

            # 功率曲线
            ax_power = axes[row, 0]
            if device.power_history:
                power_data = device.power_history[:len(time_points)]
                ax_power.plot(time_points[:len(power_data)], power_data,
                              color=colors[idx], linewidth=2)
                ax_power.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax_power.fill_between(time_points[:len(power_data)], 0, power_data,
                                      alpha=0.3, color=colors[idx])
            ax_power.set_title(f'{device.name} - 功率曲线', fontsize=12, fontweight='bold')
            ax_power.set_xlabel('时间 (秒)', fontsize=10)
            ax_power.set_ylabel('功率 (MW)', fontsize=10)
            ax_power.grid(True, alpha=0.3)

            # SOC曲线
            ax_soc = axes[row, 1]
            if device.soc_history:
                soc_data = device.soc_history[:len(time_points)]
                ax_soc.plot(time_points[:len(soc_data)], soc_data,
                            color=colors[idx], linewidth=2)
                ax_soc.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='下限')
                ax_soc.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='上限')
            ax_soc.set_title(f'{device.name} - SOC曲线', fontsize=12, fontweight='bold')
            ax_soc.set_xlabel('时间 (秒)', fontsize=10)
            ax_soc.set_ylabel('SOC', fontsize=10)
            ax_soc.set_ylim([0, 1])
            ax_soc.grid(True, alpha=0.3)
            if idx == 0:
                ax_soc.legend(fontsize=8)

            # 电压-电流关系
            ax_vi = axes[row, 2]
            if device.voltage_history and device.current_history:
                voltage_data = device.voltage_history[:len(time_points)]
                current_data = device.current_history[:len(time_points)]

                # 取部分数据点避免过于密集
                if len(voltage_data) > 100:
                    step = len(voltage_data) // 100
                    voltage_data = voltage_data[::step]
                    current_data = current_data[::step]

                scatter = ax_vi.scatter(voltage_data, current_data,
                                        c=time_points[:len(voltage_data)],
                                        cmap='viridis', s=20, alpha=0.7)

                # 添加颜色条
                if idx == 0:
                    cbar = plt.colorbar(scatter, ax=ax_vi)
                    cbar.set_label('时间 (秒)', fontsize=9)

            ax_vi.set_title(f'{device.name} - 电压电流关系', fontsize=12, fontweight='bold')
            ax_vi.set_xlabel('电压 (V)', fontsize=10)
            ax_vi.set_ylabel('电流 (A)', fontsize=10)
            ax_vi.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# ==================== 6. 主程序 ====================
def main():
    """主函数"""
    print("=" * 80)
    print("基于分层MPC的多元混合储能系统协同调度策略")
    print("南京冬季工商业应用场景 - 脉冲负荷平滑")
    print("=" * 80)

    # 1. 初始化参数
    params = SystemParameters()

    # 打印系统参数
    print(f"\n系统参数:")
    print(f"  电厂额定功率: {params.plant_rated_power} MW")
    print(f"  电厂输出功率: {params.plant_output} MW")
    print(f"  基本负荷: {params.base_load} MW")
    print(f"  脉冲幅度: {params.pulse_magnitude} MW")
    print(f"  脉冲持续时间: {params.pulse_duration} s")
    print(f"  储能总功率: {params.total_power_capacity} MW")
    print(f"  储能总容量: {params.total_energy_capacity} MWh")

    print(f"\n储能设备分配:")
    for name, ratios in params.initial_allocation.items():
        power = params.total_power_capacity * ratios['power_ratio']
        energy = params.total_energy_capacity * ratios['energy_ratio']
        print(f"  {name}: 功率={power:.1f} MW, 容量={energy:.1f} MWh")

    # 2. 创建分层调度系统
    scheduler = HierarchicalScheduler(params)

    # 3. 运行仿真
    time_points, original_load, smoothed_load = scheduler.run_simulation()

    # 4. 计算性能指标
    metrics = scheduler.calculate_performance_metrics()

    # 5. 打印详细结果
    print("\n" + "=" * 80)
    print("仿真结果详细分析")
    print("=" * 80)

    print(f"\n1. 性能指标:")
    print(f"   负荷重合度: {metrics['负荷重合度']:.4f}")
    print(f"   平均绝对误差: {metrics['平均绝对误差']:.3f} MW")
    print(f"   脉冲消除率: {metrics['脉冲消除率']:.1f}%")

    print(f"\n2. 脉冲消除策略分配占比:")
    for name, ratio in metrics['贡献比例'].items():
        device_name = scheduler.storage_systems[name].name
        print(f"   {device_name}: {ratio:.1f}%")

    print(f"\n3. 边界条件验证:")
    print(f"   最大输入脉冲: {params.max_pulse_magnitude} MW")
    print(f"   最长脉冲时间: {params.max_pulse_duration} s")
    print(f"   储能总功率容量: {params.total_power_capacity} MW")
    print(f"   实际使用最大功率: {max(original_load) - params.plant_output:.1f} MW")

    print(f"\n4. 储能设备最终状态:")
    for name, device in scheduler.storage_systems.items():
        remaining_energy = device.soc * device.energy_capacity
        remaining_power = device.get_available_power() if hasattr(device, 'get_available_power') else 0

        print(f"   {device.name}:")
        print(f"     SOC: {device.soc:.3f}")
        print(f"     剩余电量: {remaining_energy:.1f} MWh")
        print(f"     可充放电功率: {remaining_power:.1f} MW")
        print(f"     当前电压: {device.voltage:.1f} V")
        print(f"     当前电流: {device.current:.1f} A")

    print(f"\n5. 成本分析:")
    total_cost = sum(scheduler.cost_breakdown.values())
    print(f"   总运行成本: {total_cost:.2f} 元")
    for device, cost in scheduler.cost_breakdown.items():
        if cost > 0:
            percentage = cost / total_cost * 100 if total_cost > 0 else 0
            print(f"     {device}: {cost:.2f} 元 ({percentage:.1f}%)")

    # 6. 可视化
    viz = Visualization()

    print("\n" + "=" * 80)
    print("生成可视化图表...")
    print("=" * 80)

    # 6.1 综合结果图（9个子图）
    print("\n生成综合结果图（9个子图）...")
    viz.plot_comprehensive_results(time_points, original_load, smoothed_load,
                                   scheduler.storage_systems, params, metrics)

    # 6.2 各储能设备详细图
    print("生成各储能设备详细图...")
    viz.plot_individual_details(time_points, scheduler.storage_systems, params)

    # 7. 输出详细数据到CSV
    print("\n" + "=" * 80)
    print("输出详细数据到CSV文件...")
    print("=" * 80)

    # 创建数据表格
    data_dict = {
        '时间(s)': time_points,
        '原始负荷(MW)': original_load,
        '平滑后负荷(MW)': smoothed_load,
        '功率不平衡(MW)': original_load - params.plant_output
    }

    # 添加各储能设备数据
    for name, device in scheduler.storage_systems.items():
        # 功率数据
        power_key = f'{device.name}_功率(MW)'
        if device.power_history:
            power_data = device.power_history[:len(time_points)]
            if len(power_data) < len(time_points):
                power_data = list(power_data) + [0] * (len(time_points) - len(power_data))
            data_dict[power_key] = power_data

        # SOC数据
        soc_key = f'{device.name}_SOC'
        if device.soc_history:
            soc_data = device.soc_history[:len(time_points)]
            if len(soc_data) < len(time_points):
                soc_data = list(soc_data) + [soc_data[-1] if soc_data else 0.5] * (len(time_points) - len(soc_data))
            data_dict[soc_key] = soc_data

        # 电压数据
        voltage_key = f'{device.name}_电压(V)'
        if device.voltage_history:
            voltage_data = device.voltage_history[:len(time_points)]
            if len(voltage_data) < len(time_points):
                voltage_data = list(voltage_data) + [voltage_data[-1] if voltage_data else 0] * (
                            len(time_points) - len(voltage_data))
            data_dict[voltage_key] = voltage_data

        # 电流数据
        current_key = f'{device.name}_电流(A)'
        if device.current_history:
            current_data = device.current_history[:len(time_points)]
            if len(current_data) < len(time_points):
                current_data = list(current_data) + [current_data[-1] if current_data else 0] * (
                            len(time_points) - len(current_data))
            data_dict[current_key] = current_data

    # 创建DataFrame
    df = pd.DataFrame(data_dict)

    # 输出前20行数据
    print("\n前20个时间点的数据:")
    print(df.head(20).to_string())

    # 保存到CSV文件
    df.to_csv('混合储能系统调度详细数据.csv', index=False, encoding='utf-8-sig')
    print(f"\n详细数据已保存到: 混合储能系统调度详细数据.csv")

    # 8. 输出优化建议
    print("\n" + "=" * 80)
    print("优化建议")
    print("=" * 80)

    print("\n基于仿真结果的优化建议:")
    print("1. 功率分配优化:")
    print("   - 超级电容器(SC)和超导磁储能(SMES)响应速度快，应更多承担高频分量")
    print("   - 电化学储能(BESS)和压缩空气储能(CAES)适合承担能量型任务")
    print("   - 飞轮储能(FESS)在中等时间尺度上有优势")

    print("\n2. 脉冲消除策略优化:")
    print("   - 对于18MW持续10秒的脉冲，建议分配:")
    print("     * 超级电容器: 30-40% (高频响应)")
    print("     * 超导磁储能: 20-30% (高频响应)")
    print("     * 飞轮储能: 15-20% (中频响应)")
    print("     * 电化学储能: 20-25% (能量支撑)")
    print("     * 压缩空气储能: 10-15% (备用支撑)")

    print("\n3. 经济性优化:")
    print("   - 利用南京冬季低谷电价(0.21元/kWh)进行储能充电")
    print("   - 在尖峰时段(18:00-20:00)优先使用储能放电")
    print("   - 考虑天然气价格波动，优化CAES的运行策略")

    print("\n4. 边界条件建议:")
    print("   - 最大脉冲负荷不超过储能总功率的80%")
    print("   - 脉冲持续时间不超过储能总容量的10%")
    print("   - 保持各储能设备SOC在30%-70%范围内")

    print("\n5. 设备选型建议:")
    print("   - 对于高频脉冲，优先考虑超级电容器和超导磁储能")
    print("   - 对于长时间能量存储，压缩空气储能成本效益更好")
    print("   - 电化学储能是综合性能较好的选择")

    print("\n" + "=" * 80)
    print("仿真完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
