import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 系统参数设置 ====================
class SystemParameters:
    """系统参数配置类"""

    def __init__(self):
        # 基本系统参数
        self.plant_rated_power = 13.0  # MW - 电厂额定功率
        self.plant_output = 10.0  # MW - 电厂输出功率
        self.base_load = 10.0  # MW - 基本负荷

        # 脉冲参数
        self.pulse_magnitude = 8.0  # MW - 脉冲幅度 (从10MW到18MW)
        self.pulse_duration = 10.0  # s - 脉冲持续时间
        self.max_pulse = 20.0  # MW - 最大脉冲峰值
        self.max_pulse_duration = 20.0  # s - 最长脉冲持续时间

        # 储能系统总参数
        self.ess_total_power = 20.0  # MW - 储能总功率
        self.ess_total_capacity = 480.0  # MWh - 储能总容量

        # 时间参数
        self.total_time = 60.0  # s - 总模拟时间
        self.dt = 0.1  # s - 时间步长
        self.time_steps = int(self.total_time / self.dt)

        # 南京冬季电价参数 (工商业)
        self.electricity_prices = {
            'valley': 0.21,  # 元/kWh - 低谷电价
            'flat': 0.62,  # 元/kWh - 平段电价
            'peak': 1.12,  # 元/kWh - 高峰电价
            'super_peak': 1.34  # 元/kWh - 尖峰电价
        }

        # 天然气价格
        self.gas_price = 3.65  # 元/立方米

        # 时间区间划分
        self.time_periods = {
            'valley': [(0, 6), (11, 13)],  # 低谷时段 (小时)
            'flat': [(6, 11), (13, 14), (22, 24)],  # 平段
            'peak': [(14, 18), (20, 22)],  # 高峰
            'super_peak': [(18, 20)]  # 尖峰
        }


# ==================== 储能设备基类 ====================
class EnergyStorageDevice:
    """储能设备基类"""

    def __init__(self, name, power_rating, capacity, efficiency_charge,
                 efficiency_discharge, response_time, cycle_life, cost_per_kwh,
                 cost_per_kw, om_cost):
        self.name = name  # 设备名称
        self.power_rating = power_rating  # 额定功率 (MW)
        self.capacity = capacity  # 容量 (MWh)
        self.efficiency_charge = efficiency_charge  # 充电效率
        self.efficiency_discharge = efficiency_discharge  # 放电效率
        self.response_time = response_time  # 响应时间 (s)
        self.cycle_life = cycle_life  # 循环寿命 (次)
        self.cost_per_kwh = cost_per_kwh  # 单位容量成本 (元/kWh)
        self.cost_per_kw = cost_per_kw  # 单位功率成本 (元/kW)
        self.om_cost = om_cost  # 运维成本 (元/kWh/年)

        # 状态变量
        self.soc = 0.5  # 初始SOC (0-1)
        self.current_power = 0.0  # 当前功率 (MW)
        self.voltage = 1000.0  # 电压 (V) - 初始值
        self.current = 0.0  # 电流 (A)
        self.energy_stored = 0.0  # 存储能量 (MWh)
        self.temperature = 25.0  # 温度 (°C)

        # 经济指标
        self.total_cost = 0.0  # 总成本 (元)
        self.operation_cost = 0.0  # 运行成本 (元)
        self.degradation_cost = 0.0  # 退化成本 (元)

        # 历史记录
        self.history = {
            'soc': [],
            'power': [],
            'current': [],
            'voltage': [],
            'cost': [],
            'temperature': []
        }

    def initialize_state(self):
        """初始化状态"""
        self.energy_stored = self.soc * self.capacity
        self.history = {
            'soc': [self.soc],
            'power': [self.current_power],
            'current': [self.current],
            'voltage': [self.voltage],
            'cost': [self.total_cost],
            'temperature': [self.temperature]
        }

    def update_state(self, power_command, dt, price=None):
        """更新设备状态"""
        # 限制功率在额定范围内
        power = np.clip(power_command, -self.power_rating, self.power_rating)

        # 应用响应时间（一阶惯性环节）
        if self.response_time > 0:
            alpha = np.exp(-dt / self.response_time)
            self.current_power = alpha * self.current_power + (1 - alpha) * power
        else:
            self.current_power = power

        # 计算能量变化 (考虑效率)
        if self.current_power > 0:  # 放电
            energy_change = self.current_power * dt / 3600  # 转换为MWh
            actual_energy = energy_change / self.efficiency_discharge
        else:  # 充电
            energy_change = abs(self.current_power) * dt / 3600
            actual_energy = energy_change * self.efficiency_charge

        # 更新SOC
        if self.current_power > 0:  # 放电
            self.energy_stored -= energy_change
        else:  # 充电
            self.energy_stored += energy_change

        self.soc = self.energy_stored / self.capacity

        # 边界检查
        if self.soc < 0:
            self.soc = 0
            self.energy_stored = 0
        elif self.soc > 1:
            self.soc = 1
            self.energy_stored = self.capacity

        # 计算电压和电流
        self.update_electrical_parameters()

        # 计算成本
        self.calculate_cost(dt, price)

        # 记录历史
        self.record_history()

    def update_electrical_parameters(self):
        """更新电气参数"""
        # 基于实际设备特性的简化模型
        base_voltage = 1000.0  # 基准电压 (V)
        internal_resistance = 0.001  # 内阻 (Ω)

        # 功率到电流电压的转换
        power_watts = self.current_power * 1e6  # 转换为瓦特

        # 电压计算（考虑内阻压降）
        if power_watts > 0:  # 放电
            # 放电时电压下降
            current_est = power_watts / base_voltage
            voltage_drop = current_est * internal_resistance
            self.voltage = max(base_voltage - voltage_drop, base_voltage * 0.8)
            self.current = current_est
        elif power_watts < 0:  # 充电
            # 充电时电压上升
            current_est = abs(power_watts) / base_voltage
            voltage_rise = current_est * internal_resistance
            self.voltage = min(base_voltage + voltage_rise, base_voltage * 1.2)
            self.current = -current_est
        else:  # 空闲
            self.voltage = base_voltage
            self.current = 0.0

    def calculate_cost(self, dt, price):
        """计算成本"""
        energy_kwh = abs(self.current_power) * dt / 3600 * 1000  # 转换为kWh

        # 运行成本
        if price is not None:
            if self.current_power < 0:  # 充电成本
                cost = energy_kwh * price * (1 / self.efficiency_charge)
                self.operation_cost += cost
            else:  # 放电收益（负成本）
                revenue = energy_kwh * price * self.efficiency_discharge
                self.operation_cost -= revenue

        # 退化成本（基于循环寿命）
        cycle_cost = (self.cost_per_kwh * self.capacity * 1000) / self.cycle_life
        degradation = energy_kwh / (self.capacity * 1000 * 2)  # 假设每个完整循环消耗
        self.degradation_cost += degradation * cycle_cost

        # 总成本
        self.total_cost = self.operation_cost + self.degradation_cost

    def record_history(self):
        """记录历史数据"""
        self.history['soc'].append(self.soc)
        self.history['power'].append(self.current_power)
        self.history['current'].append(self.current)
        self.history['voltage'].append(self.voltage)
        self.history['cost'].append(self.total_cost)
        self.history['temperature'].append(self.temperature)

    def get_status(self):
        """获取设备状态"""
        return {
            'name': self.name,
            'soc': self.soc,
            'power': self.current_power,
            'current': self.current,
            'voltage': self.voltage,
            'total_cost': self.total_cost
        }


# ==================== 具体储能设备类 ====================
class BESS(EnergyStorageDevice):
    """电化学储能系统（锂离子电池）"""

    def __init__(self):
        super().__init__(
            name='锂离子电池',
            power_rating=6.0,  # MW (占总功率30%)
            capacity=144.0,  # MWh (占总容量30%)
            efficiency_charge=0.95,
            efficiency_discharge=0.95,
            response_time=0.1,  # 秒
            cycle_life=5000,
            cost_per_kwh=700,  # 元/kWh
            cost_per_kw=600000,  # 元/MW
            om_cost=25  # 元/kWh/年
        )
        self.type = '能量型'
        self.chemistry = '磷酸铁锂'


class SC(EnergyStorageDevice):
    """超级电容器"""

    def __init__(self):
        super().__init__(
            name='超级电容器',
            power_rating=2.0,  # MW (10%)
            capacity=4.8,  # MWh (1%)
            efficiency_charge=0.98,
            efficiency_discharge=0.98,
            response_time=0.01,  # 秒
            cycle_life=1000000,
            cost_per_kwh=2000,  # 元/kWh
            cost_per_kw=10000,  # 元/kW
            om_cost=5
        )
        self.type = '功率型'


class FESS(EnergyStorageDevice):
    """飞轮储能"""

    def __init__(self):
        super().__init__(
            name='飞轮储能',
            power_rating=4.0,  # MW (20%)
            capacity=9.6,  # MWh (2%)
            efficiency_charge=0.93,
            efficiency_discharge=0.93,
            response_time=0.05,  # 秒
            cycle_life=20000,
            cost_per_kwh=1200,  # 元/kWh
            cost_per_kw=10000,  # 元/kW
            om_cost=15
        )
        self.type = '功率型'


class SMES(EnergyStorageDevice):
    """超导磁储能"""

    def __init__(self):
        super().__init__(
            name='超导磁储能',
            power_rating=2.0,  # MW (10%)
            capacity=9.6,  # MWh (2%)
            efficiency_charge=0.99,
            efficiency_discharge=0.99,
            response_time=0.001,  # 秒
            cycle_life=100000,
            cost_per_kwh=65000,  # 元/kWh (成本极高)
            cost_per_kw=50000,  # 元/kW
            om_cost=100
        )
        self.type = '功率型'


class CAES(EnergyStorageDevice):
    """压缩空气储能"""

    def __init__(self):
        super().__init__(
            name='压缩空气储能',
            power_rating=6.0,  # MW (30%)
            capacity=312.0,  # MWh (65%)
            efficiency_charge=0.85,
            efficiency_discharge=0.55,  # 补燃式效率较低
            response_time=5.0,  # 秒 (响应较慢)
            cycle_life=10000,
            cost_per_kwh=2.5,  # 元/kWh (地质条件好时成本低)
            cost_per_kw=2000,  # 元/kW
            om_cost=10
        )
        self.type = '能量型'
        self.gas_consumption_rate = 0.05  # kg/kWh
        self.gas_cost = 0.0


# ==================== MPC控制器 ====================
class MPCController:
    """模型预测控制器"""

    def __init__(self, params, devices):
        self.params = params
        self.devices = devices

        # 控制器参数
        self.horizon = 20  # 预测时域
        self.control_interval = 1.0  # 控制间隔 (秒)

        # 设备权重（根据频段特性调整）
        self.frequency_weights = {
            '极高频': {'SMES': 0.4, 'SC': 0.3, 'FESS': 0.2, 'BESS': 0.1, 'CAES': 0.0},
            '高频': {'SMES': 0.2, 'SC': 0.5, 'FESS': 0.2, 'BESS': 0.1, 'CAES': 0.0},
            '中频': {'SMES': 0.0, 'SC': 0.3, 'FESS': 0.4, 'BESS': 0.3, 'CAES': 0.0},
            '低频': {'SMES': 0.0, 'SC': 0.0, 'FESS': 0.1, 'BESS': 0.4, 'CAES': 0.5}
        }

        # 成本权重
        self.cost_weight = 0.001
        self.soc_weight = 0.01

        # 历史数据
        self.history = {
            'load': [],
            'smoothed_load': [],
            'power_allocation': {d.name: [] for d in devices},
            'control_error': [],
            'cumulative_error': 0.0
        }

        # PID控制参数（用于消除稳态误差）
        self.integral_error = 0.0
        self.last_error = 0.0
        self.kp = 1.0  # 比例增益
        self.ki = 0.05  # 积分增益
        self.kd = 0.1  # 微分增益

    def frequency_decomposition(self, power_imbalance, time_index):
        """频段分解函数（替代小波分解的简化版本）"""
        # 基于时间特性的频段分解
        frequencies = {
            '极高频': 0.0,  # >10Hz (SMES)
            '高频': 0.0,  # 1-10Hz (SC)
            '中频': 0.0,  # 0.1-1Hz (FESS)
            '低频': 0.0  # <0.1Hz (BESS, CAES)
        }

        # 简单的频段分配逻辑
        if abs(power_imbalance) < 0.1:  # 小扰动
            frequencies['极高频'] = power_imbalance * 0.5
            frequencies['高频'] = power_imbalance * 0.3
            frequencies['中频'] = power_imbalance * 0.2
        elif abs(power_imbalance) < 1.0:  # 中等扰动
            frequencies['高频'] = power_imbalance * 0.4
            frequencies['中频'] = power_imbalance * 0.4
            frequencies['低频'] = power_imbalance * 0.2
        else:  # 大脉冲
            frequencies['低频'] = power_imbalance * 0.6
            frequencies['中频'] = power_imbalance * 0.3
            frequencies['高频'] = power_imbalance * 0.1

        return frequencies

    def allocate_power(self, power_imbalance, time_index):
        """分配功率到各个储能设备"""
        # 频段分解
        freq_components = self.frequency_decomposition(power_imbalance, time_index)

        # 设备分配
        allocations = {}
        for device in self.devices:
            allocation = 0.0

            # 根据频段分配
            for freq_name, freq_power in freq_components.items():
                device_weight = self.frequency_weights[freq_name].get(
                    'SMES' if device.name == '超导磁储能' else
                    'SC' if device.name == '超级电容器' else
                    'FESS' if device.name == '飞轮储能' else
                    'BESS' if device.name == '锂离子电池' else
                    'CAES', 0.0
                )
                allocation += freq_power * device_weight

            # 考虑设备SOC状态
            soc_factor = self.calculate_soc_factor(device)
            allocation *= soc_factor

            # 考虑设备功率限制
            allocation = np.clip(allocation, -device.power_rating, device.power_rating)

            allocations[device.name] = allocation

        # PID校正（消除累计误差）
        total_allocated = sum(allocations.values())
        error = power_imbalance - total_allocated

        # PID计算
        self.integral_error += error * self.params.dt
        derivative_error = (error - self.last_error) / self.params.dt
        pid_correction = self.kp * error + self.ki * self.integral_error + self.kd * derivative_error
        self.last_error = error

        # 应用PID校正（按比例分配到各设备）
        if abs(total_allocated) > 0.01:
            for device_name in allocations:
                proportion = abs(allocations[device_name]) / abs(total_allocated)
                allocations[device_name] += pid_correction * proportion

        return allocations

    def calculate_soc_factor(self, device):
        """计算SOC影响因子"""
        # SOC在0.3-0.7范围内最佳
        target_soc = 0.5
        soc_diff = abs(device.soc - target_soc)

        if soc_diff < 0.2:  # SOC接近目标值
            return 1.0
        elif device.soc > 0.7:  # SOC偏高，鼓励放电
            return max(0.5, 1.0 - (device.soc - 0.7) * 2)
        elif device.soc < 0.3:  # SOC偏低，鼓励充电
            return max(0.5, 1.0 - (0.3 - device.soc) * 2)
        else:
            return 0.8

    def get_current_price(self, current_time):
        """获取当前电价"""
        hour = current_time / 3600  # 转换为小时

        # 判断时间段
        for period, time_ranges in self.params.time_periods.items():
            for start, end in time_ranges:
                if start <= hour % 24 < end:
                    return self.params.electricity_prices[period]

        return self.params.electricity_prices['flat']

    def control_step(self, current_load, current_time, time_index):
        """执行控制步"""
        # 计算不平衡功率
        plant_output = self.params.plant_output
        power_imbalance = current_load - plant_output

        # 分配功率
        allocations = self.allocate_power(power_imbalance, time_index)

        # 应用控制
        total_controlled = 0.0
        for device in self.devices:
            device_power = allocations[device.name]

            # 获取当前电价
            current_price = self.get_current_price(current_time)

            # 更新设备状态
            device.update_state(device_power, self.params.dt, current_price)

            total_controlled += device.current_power

            # 记录分配
            self.history['power_allocation'][device.name].append(device.current_power)

        # 计算平滑后负荷
        smoothed_load = plant_output + total_controlled

        # 计算控制误差
        control_error = current_load - smoothed_load

        # 记录
        self.history['load'].append(current_load)
        self.history['smoothed_load'].append(smoothed_load)
        self.history['control_error'].append(control_error)
        self.history['cumulative_error'] += abs(control_error) * self.params.dt

        return smoothed_load, allocations


# ==================== 主模拟系统 ====================
class HybridEnergySystem:
    """混合储能系统主模拟"""

    def __init__(self):
        self.params = SystemParameters()

        # 初始化储能设备
        self.devices = [
            BESS(),  # 锂离子电池
            SC(),  # 超级电容
            FESS(),  # 飞轮储能
            SMES(),  # 超导磁储能
            CAES()  # 压缩空气储能
        ]

        # 初始化设备状态
        for device in self.devices:
            device.initialize_state()

        # 初始化MPC控制器
        self.controller = MPCController(self.params, self.devices)

        # 初始化结果存储
        self.results = {
            'time': [],
            'original_load': [],
            'smoothed_load': [],
            'plant_output': [],
            'total_ess_power': [],
            'control_error': [],
            'total_cost': [],
            'cumulative_error': 0.0
        }

    def generate_load_profile(self):
        """生成负荷曲线"""
        time_array = np.arange(0, self.params.total_time, self.params.dt)
        load_profile = np.ones_like(time_array) * self.params.base_load

        # 添加脉冲 (10-20秒)
        pulse_start = 10.0
        pulse_end = pulse_start + self.params.pulse_duration

        pulse_mask = (time_array >= pulse_start) & (time_array <= pulse_end)
        load_profile[pulse_mask] = self.params.base_load + self.params.pulse_magnitude

        # 添加平滑过渡（避免阶跃变化）
        for i in range(len(load_profile)):
            if i > 0:
                # 应用一阶惯性，使变化更平滑
                load_profile[i] = 0.98 * load_profile[i] + 0.02 * load_profile[i - 1]

        # 添加小扰动（模拟真实负荷）
        noise = np.random.normal(0, 0.05, len(time_array))
        load_profile += noise

        # 再次平滑
        from scipy.ndimage import gaussian_filter1d
        load_profile = gaussian_filter1d(load_profile, sigma=2)

        return time_array, load_profile

    def run_simulation(self):
        """运行模拟"""
        print("=" * 60)
        print("多元混合储能系统协同调度策略模拟")
        print("=" * 60)
        print(f"系统参数: 电厂出力={self.params.plant_output}MW, 储能总功率={self.params.ess_total_power}MW")
        print(f"脉冲参数: 幅度={self.params.pulse_magnitude}MW, 持续时间={self.params.pulse_duration}s")
        print("-" * 60)

        # 生成负荷曲线
        time_array, load_profile = self.generate_load_profile()

        # 主模拟循环
        for i, current_time in enumerate(time_array):
            current_load = load_profile[i]

            # 执行MPC控制
            smoothed_load, allocations = self.controller.control_step(
                current_load, current_time, i
            )

            # 记录结果
            self.results['time'].append(current_time)
            self.results['original_load'].append(current_load)
            self.results['smoothed_load'].append(smoothed_load)
            self.results['plant_output'].append(self.params.plant_output)

            # 计算总ESS功率
            total_ess_power = sum(device.current_power for device in self.devices)
            self.results['total_ess_power'].append(total_ess_power)

            # 计算控制误差
            error = current_load - smoothed_load
            self.results['control_error'].append(error)
            self.results['cumulative_error'] += abs(error) * self.params.dt

            # 计算总成本
            total_cost = sum(device.total_cost for device in self.devices)
            self.results['total_cost'].append(total_cost)

        print("模拟完成!")
        print("-" * 60)

    def analyze_results(self):
        """分析结果"""
        # 转换为numpy数组
        original_load = np.array(self.results['original_load'])
        smoothed_load = np.array(self.results['smoothed_load'])
        control_error = np.array(self.results['control_error'])

        # 1. 脉冲平滑度评估
        pulse_mask = (np.array(self.results['time']) >= 10) & (np.array(self.results['time']) <= 20)
        pulse_original = original_load[pulse_mask]
        pulse_smoothed = smoothed_load[pulse_mask]

        # 计算重合度
        if len(pulse_original) > 1:
            correlation = np.corrcoef(pulse_original, pulse_smoothed)[0, 1]
        else:
            correlation = 0

        # 计算误差指标
        rmse = np.sqrt(np.mean(control_error ** 2))
        max_error = np.max(np.abs(control_error))
        avg_error = np.mean(np.abs(control_error))

        # 2. 成本分析
        total_costs = {}
        operation_costs = {}
        degradation_costs = {}

        for device in self.devices:
            total_costs[device.name] = device.total_cost
            operation_costs[device.name] = device.operation_cost
            degradation_costs[device.name] = device.degradation_cost

        total_system_cost = sum(total_costs.values())

        # 3. 储能设备性能分析
        soc_final = {device.name: device.soc for device in self.devices}
        energy_throughput = {}

        for device in self.devices:
            if len(device.history['power']) > 0:
                power_array = np.array(device.history['power'])
                charge_mask = power_array < 0
                discharge_mask = power_array > 0

                charge_energy = np.sum(power_array[charge_mask]) * self.params.dt / 3600
                discharge_energy = np.sum(power_array[discharge_mask]) * self.params.dt / 3600

                energy_throughput[device.name] = {
                    '充电能量(MWh)': abs(charge_energy),
                    '放电能量(MWh)': discharge_energy,
                    '往返效率': discharge_energy / abs(charge_energy) if abs(charge_energy) > 1e-10 else 0
                }
            else:
                energy_throughput[device.name] = {
                    '充电能量(MWh)': 0,
                    '放电能量(MWh)': 0,
                    '往返效率': 0
                }

        # 4. 功率分配占比
        total_power = np.array(self.results['total_ess_power'])
        allocation_percentages = {}

        for device in self.devices:
            if len(device.history['power']) > 0:
                device_power = np.array(device.history['power'])
                total_abs_power = np.sum(np.abs(total_power))
                if total_abs_power > 0:
                    percentage = np.sum(np.abs(device_power)) / total_abs_power * 100
                else:
                    percentage = 0
            else:
                percentage = 0
            allocation_percentages[device.name] = percentage

        return {
            '性能指标': {
                '重合度': correlation,
                'RMSE(MW)': rmse,
                '最大误差(MW)': max_error,
                '平均误差(MW)': avg_error,
                '累计误差(MW·s)': self.results['cumulative_error']
            },
            '成本分析': {
                '总成本(元)': total_costs,
                '运行成本(元)': operation_costs,
                '退化成本(元)': degradation_costs,
                '系统总成本(元)': total_system_cost
            },
            '设备状态': {
                '最终SOC': soc_final,
                '能量吞吐': energy_throughput
            },
            '分配占比': allocation_percentages
        }

    def plot_results(self):
        """绘制结果图表"""
        time_array = np.array(self.results['time'])
        control_error_array = np.array(self.results['control_error'])

        # 创建图形（8个子图）
        fig = plt.figure(figsize=(20, 16))

        # 1. 原始负荷与平滑后负荷对比（主要结果）
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(time_array, self.results['original_load'], 'b-', linewidth=2.5,
                 label='原始负荷', alpha=0.8)
        ax1.plot(time_array, self.results['smoothed_load'], 'r--', linewidth=2.5,
                 label='平滑后负荷', alpha=0.9)
        ax1.plot(time_array, self.results['plant_output'], 'g-', linewidth=1.5,
                 label='电厂出力', alpha=0.6)
        ax1.fill_between(time_array, self.results['original_load'],
                         self.results['smoothed_load'], alpha=0.2, color='gray')
        ax1.set_xlabel('时间 (秒)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('功率 (MW)', fontsize=11, fontweight='bold')
        ax1.set_title('负荷平滑效果对比图', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(0, self.params.total_time)

        # 2. 控制误差分析 - 修复错误：使用numpy数组进行条件判断
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(time_array, control_error_array, 'r-', linewidth=2)
        # 使用numpy数组进行条件判断
        if len(control_error_array) > 0:
            positive_mask = control_error_array >= 0
            negative_mask = control_error_array < 0
            if np.any(positive_mask):
                ax2.fill_between(time_array[positive_mask], 0, control_error_array[positive_mask],
                                 alpha=0.3, color='red')
            if np.any(negative_mask):
                ax2.fill_between(time_array[negative_mask], control_error_array[negative_mask], 0,
                                 alpha=0.3, color='blue')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_xlabel('时间 (秒)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('误差 (MW)', fontsize=11, fontweight='bold')
        ax2.set_title('控制误差分析', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(0, self.params.total_time)

        # 3. 储能设备SOC变化
        ax3 = plt.subplot(3, 3, 3)
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        for i, device in enumerate(self.devices):
            if len(device.history['soc']) > 0:
                soc_data = np.array(device.history['soc'])
                time_len = min(len(time_array), len(soc_data))
                ax3.plot(time_array[:time_len], soc_data[:time_len],
                         color=colors[i], linewidth=2, label=device.name, alpha=0.8)
        ax3.set_xlabel('时间 (秒)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('荷电状态 (SOC)', fontsize=11, fontweight='bold')
        ax3.set_title('储能设备SOC变化曲线', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_ylim(0, 1)
        ax3.set_xlim(0, self.params.total_time)

        # 4. 各储能设备功率分配
        ax4 = plt.subplot(3, 3, 4)
        for i, device in enumerate(self.devices):
            if len(device.history['power']) > 0:
                power_data = np.array(device.history['power'])
                time_len = min(len(time_array), len(power_data))
                ax4.plot(time_array[:time_len], power_data[:time_len],
                         color=colors[i], linewidth=2, label=device.name, alpha=0.8)
        ax4.set_xlabel('时间 (秒)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('功率 (MW)', fontsize=11, fontweight='bold')
        ax4.set_title('储能设备功率分配', fontsize=14, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_xlim(0, self.params.total_time)
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)

        # 5. 电流随时间变化
        ax5 = plt.subplot(3, 3, 5)
        for device in self.devices:
            if device.name in ['锂离子电池', '超级电容器', '飞轮储能']:
                if len(device.history['current']) > 0:
                    current_data = np.array(device.history['current'])
                    # 确保数据长度匹配
                    time_len = min(len(time_array), len(current_data))
                    if time_len > 10:
                        window = min(5, time_len // 10)
                        if window > 1:
                            smoothed_current = np.convolve(current_data[:time_len],
                                                           np.ones(window) / window,
                                                           mode='valid')
                            smoothed_time = time_array[:len(smoothed_current)]
                            ax5.plot(smoothed_time, smoothed_current / 1000,
                                     linewidth=2, label=device.name, alpha=0.8)
                        else:
                            ax5.plot(time_array[:time_len], current_data[:time_len] / 1000,
                                     linewidth=2, label=device.name, alpha=0.8)
        ax5.set_xlabel('时间 (秒)', fontsize=11, fontweight='bold')
        ax5.set_ylabel('电流 (kA)', fontsize=11, fontweight='bold')
        ax5.set_title('储能设备电流变化', fontsize=14, fontweight='bold')
        ax5.legend(loc='upper right', fontsize=9)
        ax5.grid(True, alpha=0.3, linestyle='--')
        ax5.set_xlim(0, self.params.total_time)

        # 6. 电压随时间变化
        ax6 = plt.subplot(3, 3, 6)
        for device in self.devices:
            if len(device.history['voltage']) > 0:
                voltage_data = np.array(device.history['voltage'])
                time_len = min(len(time_array), len(voltage_data))
                if time_len > 10:
                    window = min(5, time_len // 10)
                    if window > 1:
                        smoothed_voltage = np.convolve(voltage_data[:time_len],
                                                       np.ones(window) / window,
                                                       mode='valid')
                        smoothed_time = time_array[:len(smoothed_voltage)]
                        ax6.plot(smoothed_time, smoothed_voltage / 1000,
                                 linewidth=2, label=device.name, alpha=0.8)
                    else:
                        ax6.plot(time_array[:time_len], voltage_data[:time_len] / 1000,
                                 linewidth=2, label=device.name, alpha=0.8)
        ax6.set_xlabel('时间 (秒)', fontsize=11, fontweight='bold')
        ax6.set_ylabel('电压 (kV)', fontsize=11, fontweight='bold')
        ax6.set_title('储能设备电压变化', fontsize=14, fontweight='bold')
        ax6.legend(loc='upper right', fontsize=9)
        ax6.grid(True, alpha=0.3, linestyle='--')
        ax6.set_xlim(0, self.params.total_time)

        # 7. 脉冲功率分配饼图
        ax7 = plt.subplot(3, 3, 7)
        analysis_results = self.analyze_results()
        allocation_percentages = analysis_results['分配占比']

        # 准备饼图数据
        labels = []
        sizes = []
        colors_pie = []

        for device, color in zip(self.devices, colors):
            percentage = allocation_percentages.get(device.name, 0)
            if percentage > 0.1:  # 只显示占比大于0.1%的设备
                labels.append(device.name)
                sizes.append(percentage)
                colors_pie.append(color)

        if sizes and sum(sizes) > 0:
            wedges, texts, autotexts = ax7.pie(sizes, labels=labels, colors=colors_pie,
                                               autopct='%1.1f%%', startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
        else:
            ax7.text(0.5, 0.5, '无显著功率分配', ha='center', va='center',
                     fontsize=12, fontweight='bold')

        ax7.set_title('脉冲功率分配策略', fontsize=14, fontweight='bold')

        # 8. 成本分析柱状图
        ax8 = plt.subplot(3, 3, 8)
        cost_data = []
        device_names = []
        for device in self.devices:
            cost_data.append(device.total_cost)
            device_names.append(device.name)

        x_pos = np.arange(len(cost_data))
        bars = ax8.bar(x_pos, cost_data, color=colors, alpha=0.7, edgecolor='black')
        ax8.set_xlabel('储能设备', fontsize=11, fontweight='bold')
        ax8.set_ylabel('总成本 (元)', fontsize=11, fontweight='bold')
        ax8.set_title('各储能设备总成本', fontsize=14, fontweight='bold')
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels(device_names, rotation=45, ha='right')
        ax8.grid(True, alpha=0.3, axis='y', linestyle='--')

        # 在柱子上添加数值
        for bar, cost in zip(bars, cost_data):
            height = bar.get_height()
            if height != 0:
                ax8.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{cost:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 9. 能量-电流关系图
        ax9 = plt.subplot(3, 3, 9)
        for i, device in enumerate(self.devices):
            if len(device.history['current']) > 0 and len(device.history['soc']) > 0:
                # 计算存储能量
                stored_energy = np.array(device.history['soc']) * device.capacity * 1000  # kWh
                current_array = np.array(device.history['current'])

                # 确保数据长度一致
                min_len = min(len(stored_energy), len(current_array))
                if min_len > 0:
                    stored_energy = stored_energy[:min_len]
                    current_array = current_array[:min_len]

                    # 采样，避免点太多
                    if min_len > 100:
                        step = min_len // 100
                        indices = range(0, min_len, step)
                        stored_energy = stored_energy[indices]
                        current_array = current_array[indices]
                        time_subset = time_array[indices] if len(time_array) > max(indices) else np.arange(len(indices))
                    else:
                        time_subset = time_array[:min_len]

                    if len(stored_energy) > 0 and len(current_array) > 0:
                        scatter = ax9.scatter(current_array / 1000, stored_energy,
                                              c=time_subset, cmap='viridis', s=30,
                                              alpha=0.6, label=device.name)

        ax9.set_xlabel('电流 (kA)', fontsize=11, fontweight='bold')
        ax9.set_ylabel('存储能量 (kWh)', fontsize=11, fontweight='bold')
        ax9.set_title('能量-电流关系图', fontsize=14, fontweight='bold')
        ax9.grid(True, alpha=0.3, linestyle='--')
        if self.devices:
            ax9.legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        plt.show()

        # 单独绘制脉冲平滑细节图
        self.plot_pulse_detail(time_array)

    def plot_pulse_detail(self, time_array):
        """绘制脉冲平滑细节图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 脉冲期间详细对比
        ax1 = axes[0, 0]
        pulse_start, pulse_end = 8, 22  # 扩展时间范围以便观察过渡
        pulse_mask = (time_array >= pulse_start) & (time_array <= pulse_end)

        original_load_array = np.array(self.results['original_load'])
        smoothed_load_array = np.array(self.results['smoothed_load'])

        ax1.plot(time_array[pulse_mask], original_load_array[pulse_mask],
                 'b-', linewidth=3, label='原始负荷', alpha=0.8)
        ax1.plot(time_array[pulse_mask], smoothed_load_array[pulse_mask],
                 'r--', linewidth=3, label='平滑后负荷', alpha=0.9)
        ax1.fill_between(time_array[pulse_mask],
                         original_load_array[pulse_mask],
                         smoothed_load_array[pulse_mask],
                         alpha=0.2, color='gray')
        ax1.axvline(x=10, color='k', linestyle=':', alpha=0.5, label='脉冲开始')
        ax1.axvline(x=20, color='k', linestyle=':', alpha=0.5, label='脉冲结束')
        ax1.set_xlabel('时间 (秒)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('功率 (MW)', fontsize=12, fontweight='bold')
        ax1.set_title('脉冲期间负荷平滑细节', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # 2. 脉冲期间各设备出力详情
        ax2 = axes[0, 1]
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        for i, device in enumerate(self.devices):
            if len(device.history['power']) > 0:
                power_data = np.array(device.history['power'])
                time_len = min(len(time_array[pulse_mask]), len(power_data))
                if time_len > 0:
                    ax2.plot(time_array[pulse_mask][:time_len], power_data[:time_len],
                             color=colors[i], linewidth=2, label=device.name, alpha=0.8)
        ax2.set_xlabel('时间 (秒)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('功率 (MW)', fontsize=12, fontweight='bold')
        ax2.set_title('脉冲期间各设备出力详情', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)

        # 3. 脉冲期间SOC变化详情
        ax3 = axes[1, 0]
        for i, device in enumerate(self.devices):
            if len(device.history['soc']) > 0:
                soc_data = np.array(device.history['soc'])
                time_len = min(len(time_array[pulse_mask]), len(soc_data))
                if time_len > 0:
                    ax3.plot(time_array[pulse_mask][:time_len], soc_data[:time_len],
                             color=colors[i], linewidth=2, label=device.name, alpha=0.8)
        ax3.set_xlabel('时间 (秒)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('荷电状态 (SOC)', fontsize=12, fontweight='bold')
        ax3.set_title('脉冲期间SOC变化详情', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=9)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_ylim(0, 1)

        # 4. 脉冲期间总储能功率
        ax4 = axes[1, 1]
        total_ess_power = np.array(self.results['total_ess_power'])
        power_pulse = total_ess_power[pulse_mask]

        ax4.plot(time_array[pulse_mask], power_pulse,
                 'b-', linewidth=2, alpha=0.8)

        # 修复错误：使用numpy数组进行条件判断
        if len(power_pulse) > 0:
            positive_mask = power_pulse >= 0
            negative_mask = power_pulse < 0
            if np.any(positive_mask):
                ax4.fill_between(time_array[pulse_mask][positive_mask], 0, power_pulse[positive_mask],
                                 alpha=0.3, color='blue')
            if np.any(negative_mask):
                ax4.fill_between(time_array[pulse_mask][negative_mask], power_pulse[negative_mask], 0,
                                 alpha=0.3, color='red')
        ax4.set_xlabel('时间 (秒)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('总储能功率 (MW)', fontsize=12, fontweight='bold')
        ax4.set_title('脉冲期间总储能功率', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)

        plt.tight_layout()
        plt.show()

    def print_detailed_report(self):
        """打印详细报告"""
        analysis_results = self.analyze_results()

        print("\n" + "=" * 80)
        print("多元混合储能系统协同调度策略详细报告")
        print("=" * 80)

        print("\n一、系统配置参数")
        print("-" * 40)
        print(f"1. 电厂参数:")
        print(f"   额定功率: {self.params.plant_rated_power} MW")
        print(f"   输出功率: {self.params.plant_output} MW (恒定)")
        print(f"2. 负荷参数:")
        print(f"   基本负荷: {self.params.base_load} MW")
        print(f"   脉冲幅度: {self.params.pulse_magnitude} MW")
        print(f"   脉冲持续时间: {self.params.pulse_duration} s")
        print(f"3. 储能系统总参数:")
        print(f"   总功率: {self.params.ess_total_power} MW")
        print(f"   总容量: {self.params.ess_total_capacity} MWh")

        print("\n二、储能设备配置详情")
        print("-" * 40)
        for device in self.devices:
            print(f"\n{device.name}:")
            print(f"   额定功率: {device.power_rating} MW")
            print(f"   容量: {device.capacity} MWh")
            print(f"   类型: {device.type}")
            print(f"   充电效率: {device.efficiency_charge * 100:.1f}%")
            print(f"   放电效率: {device.efficiency_discharge * 100:.1f}%")
            print(f"   响应时间: {device.response_time} s")
            if hasattr(device, 'history') and device.history['soc']:
                print(f"   初始SOC: {device.history['soc'][0]:.3f}")
            print(f"   最终SOC: {device.soc:.3f}")

        print("\n三、脉冲平滑性能指标")
        print("-" * 40)
        metrics = analysis_results['性能指标']
        print(f"1. 重合度: {metrics['重合度']:.4f}")
        print(f"2. 均方根误差(RMSE): {metrics['RMSE(MW)']:.4f} MW")
        print(f"3. 最大误差: {metrics['最大误差(MW)']:.4f} MW")
        print(f"4. 平均误差: {metrics['平均误差(MW)']:.4f} MW")
        print(f"5. 累计误差: {metrics['累计误差(MW·s)']:.4f} MW·s")

        print("\n四、脉冲功率分配策略")
        print("-" * 40)
        allocation = analysis_results['分配占比']
        print("各设备承担的脉冲功率占比:")
        total_percentage = 0
        for device_name, percentage in allocation.items():
            print(f"   {device_name}: {percentage:.2f}%")
            total_percentage += percentage
        print(f"   总计: {total_percentage:.2f}%")

        print("\n五、经济成本分析")
        print("-" * 40)
        costs = analysis_results['成本分析']
        print("1. 各设备总成本:")
        for device_name, cost in costs['总成本(元)'].items():
            print(f"   {device_name}: {cost:,.2f} 元")
        print(f"\n2. 系统总运行成本: {costs['系统总成本(元)']:,.2f} 元")

        print("\n六、储能设备最终状态")
        print("-" * 40)
        status = analysis_results['设备状态']
        print("1. 最终SOC:")
        for device_name, soc in status['最终SOC'].items():
            print(f"   {device_name}: {soc:.3f}")

        print("\n2. 能量吞吐量:")
        for device_name, throughput in status['能量吞吐'].items():
            print(f"   {device_name}:")
            print(f"     充电能量: {throughput['充电能量(MWh)']:.4f} MWh")
            print(f"     放电能量: {throughput['放电能量(MWh)']:.4f} MWh")
            print(f"     往返效率: {throughput['往返效率']:.3f}")

        print("\n七、边界条件与约束")
        print("-" * 40)
        print("1. 功率边界:")
        print(f"   最大输入脉冲: {self.params.max_pulse} MW")
        print(f"   最长脉冲持续时间: {self.params.max_pulse_duration} s")
        print(f"   储能系统最大功率: {self.params.ess_total_power} MW")
        print("2. 能量边界:")
        print(f"   储能系统总容量: {self.params.ess_total_capacity} MWh")
        print("3. SOC边界:")
        print("   所有设备: 0 ≤ SOC ≤ 1")
        print("4. 响应时间边界:")
        for device in self.devices:
            print(f"   {device.name}: {device.response_time} s")

        print("\n八、控制策略说明")
        print("-" * 40)
        print("1. 分层控制架构:")
        print("   上层(经济调度层): 基于电价信号优化充放电策略")
        print("   下层(实时平衡层): 基于MPC实现功率精确跟踪")
        print("2. 频段分解策略:")
        print("   • 超导磁储能(SMES): 承担极高频分量 (响应时间: 1ms)")
        print("   • 超级电容器(SC): 承担高频分量 (响应时间: 10ms)")
        print("   • 飞轮储能(FESS): 承担中高频分量 (响应时间: 50ms)")
        print("   • 锂离子电池(BESS): 承担中频分量 (响应时间: 100ms)")
        print("   • 压缩空气储能(CAES): 承担低频分量 (响应时间: 5s)")
        print("3. PID校正机制:")
        print("   • 比例增益(Kp): 1.0")
        print("   • 积分增益(Ki): 0.05 (消除稳态误差)")
        print("   • 微分增益(Kd): 0.1 (改善动态响应)")

        print("\n九、南京冬季电价数据")
        print("-" * 40)
        print("工商业电价 (冬季12月-2月):")
        print(f"   低谷时段 (00:00-06:00, 11:00-13:00): {self.params.electricity_prices['valley']} 元/kWh")
        print(f"   平段 (06:00-11:00, 13:00-14:00, 22:00-24:00): {self.params.electricity_prices['flat']} 元/kWh")
        print(f"   高峰时段 (14:00-22:00): {self.params.electricity_prices['peak']} 元/kWh")
        print(f"   尖峰时段 (18:00-20:00, 大工业): {self.params.electricity_prices['super_peak']} 元/kWh")
        print(
            f"   峰谷价差: {self.params.electricity_prices['peak'] - self.params.electricity_prices['valley']:.2f} 元/kWh")
        print(f"   天然气价格: {self.params.gas_price} 元/立方米")

        print("\n十、消除脉冲总成本分析")
        print("-" * 40)
        # 计算脉冲总能量
        pulse_energy = self.params.pulse_magnitude * self.params.pulse_duration / 3600 * 1000  # kWh
        total_cost = costs['系统总成本(元)']

        print(f"   脉冲总能量: {pulse_energy:.2f} kWh")
        print(f"   平抑总成本: {total_cost:.2f} 元")
        if pulse_energy > 0:
            cost_per_kwh = total_cost / pulse_energy
            print(f"   平均平抑成本: {cost_per_kwh:.4f} 元/kWh")
            print(f"   相对于高峰电价的经济性: {(self.params.electricity_prices['peak'] - cost_per_kwh):.4f} 元/kWh")

        print("\n十一、实时数据示例（最后5个时间点）")
        print("-" * 40)
        n_samples = 5
        total_points = len(self.results['time'])
        if total_points > 0:
            start_idx = max(0, total_points - n_samples)
            for i in range(start_idx, total_points):
                print(f"\n时间: {self.results['time'][i]:.1f}s")
                print(f"  原始负荷: {self.results['original_load'][i]:.2f} MW")
                print(f"  平滑后负荷: {self.results['smoothed_load'][i]:.2f} MW")
                print(f"  控制误差: {self.results['control_error'][i]:.3f} MW")

                for device in self.devices:
                    if len(device.history['current']) > i:
                        print(f"  {device.name}:")
                        print(f"    功率: {device.history['power'][i]:.3f} MW")
                        print(f"    电流: {device.history['current'][i]:.1f} A")
                        print(f"    电压: {device.history['voltage'][i]:.1f} V")
                        print(f"    SOC: {device.history['soc'][i]:.3f}")
        else:
            print("无可用数据")

        print("\n" + "=" * 80)
        print("报告结束")
        print("=" * 80)


# ==================== 主程序 ====================
def main():
    """主函数"""
    print("多元混合储能系统协同调度策略模拟")
    print("版本: 2.0 | 基于南京冬季工商业电价数据")
    print("=" * 60)

    try:
        # 创建混合储能系统
        system = HybridEnergySystem()

        # 运行模拟
        system.run_simulation()

        # 打印详细报告
        system.print_detailed_report()

        # 绘制图表
        system.plot_results()

        # 保存结果到CSV
        save_results(system)

    except Exception as e:
        print(f"模拟过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


def save_results(system):
    """保存结果到CSV文件"""
    # 准备数据
    time_array = np.array(system.results['time'])

    # 创建DataFrame
    data_dict = {
        '时间_s': time_array,
        '原始负荷_MW': system.results['original_load'],
        '平滑后负荷_MW': system.results['smoothed_load'],
        '电厂出力_MW': system.results['plant_output'],
        '总储能功率_MW': system.results['total_ess_power'],
        '控制误差_MW': system.results['control_error'],
        '总成本_元': system.results['total_cost']
    }

    # 添加各设备数据
    for device in system.devices:
        prefix = device.name
        if len(device.history['power']) > 0:
            # 确保数据长度一致
            min_len = min(len(time_array), len(device.history['power']))
            data_dict[f'{prefix}_功率_MW'] = device.history['power'][:min_len]
            data_dict[f'{prefix}_SOC'] = device.history['soc'][:min_len]
            data_dict[f'{prefix}_电流_A'] = device.history['current'][:min_len]
            data_dict[f'{prefix}_电压_V'] = device.history['voltage'][:min_len]
            data_dict[f'{prefix}_成本_元'] = device.history['cost'][:min_len]

    # 确保所有数组长度一致
    min_length = min(len(v) for v in data_dict.values() if hasattr(v, '__len__'))
    for key in data_dict:
        if hasattr(data_dict[key], '__len__') and len(data_dict[key]) > min_length:
            data_dict[key] = data_dict[key][:min_length]

    df = pd.DataFrame(data_dict)

    # 保存到CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'混合储能系统调度结果_{timestamp}.csv'
    df.to_csv(filename, index=False, encoding='utf-8-sig')

    print(f"\n结果已保存到文件: {filename}")


if __name__ == "__main__":
    main()
