import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, filedialog
import warnings
from datetime import datetime
import csv

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 成本参数（基于文件数据） ====================
class CostParameters:
    """成本参数配置类"""

    def __init__(self):
        # 南京冬季工商业电价（2026年1月）
        self.electricity_prices = {
            'valley': 0.21,  # 低谷电价 0.21元/kWh (00:00-06:00, 11:00-13:00)
            'flat': 0.62,  # 平段电价 0.62元/kWh
            'peak': 1.12,  # 高峰电价 1.12元/kWh (14:00-22:00)
            'sharp': 1.34  # 尖峰电价 1.34元/kWh (18:00-20:00)
        }

        # 天然气价格（南京冬季）
        self.gas_price = 3.6  # 元/立方米

        # 设备投资成本（元/kWh，基于文件数据）
        self.device_capex = {
            '锂电池储能': 700,  # 元/kWh
            '超级电容器': 2000,  # 元/kWh
            '飞轮储能': 10000,  # 元/kWh
            '超导磁储能': 60000,  # 元/kWh
            '压缩空气储能': 3000  # 元/kWh
        }

        # 运维成本（元/MWh/年）
        self.device_opex = {
            '锂电池储能': 25,
            '超级电容器': 10,
            '飞轮储能': 15,
            '超导磁储能': 50,
            '压缩空气储能': 5
        }

        # 寿命年限（年）
        self.device_lifespan = {
            '锂电池储能': 10,
            '超级电容器': 20,
            '飞轮储能': 20,
            '超导磁储能': 25,
            '压缩空气储能': 30
        }

    def get_electricity_price(self, hour):
        """根据小时获取电价"""
        if 0 <= hour < 6 or 11 <= hour < 13:
            return self.electricity_prices['valley']
        elif 14 <= hour < 18:
            return self.electricity_prices['peak']
        elif 18 <= hour < 20:
            return self.electricity_prices['sharp']
        elif 14 <= hour < 22:
            return self.electricity_prices['peak']
        else:
            return self.electricity_prices['flat']

    def calculate_lcoe(self, device_name, power, capacity):
        """计算度电成本 LCOE"""
        capex = self.device_capex[device_name] * capacity * 1000  # 转换为元
        opex = self.device_opex[device_name] * capacity * 1000  # 年运维成本
        lifespan = self.device_lifespan[device_name]

        # 简化LCOE计算：年化成本 / 年发电量
        annual_cost = capex / lifespan + opex
        annual_energy = capacity * 365 * 0.8  # 假设年利用率80%

        return annual_cost / annual_energy  # 元/kWh


# ==================== 储能设备基类（增强版） ====================
class EnergyStorageDevice:
    """储能设备基类（增强响应特性）"""

    def __init__(self, name, power_rating, capacity, efficiency,
                 cost_params, response_time, ramp_rate):
        self.name = name
        self.power_rating = power_rating  # MW
        self.capacity = capacity  # MWh
        self.efficiency = efficiency
        self.response_time = response_time  # 响应时间 (s)
        self.ramp_rate = ramp_rate  # 爬坡率 (MW/s)
        self.cost_params = cost_params

        # 状态变量
        self.soc = 0.5  # 初始SOC
        self.current_power = 0  # 当前功率 (MW)
        self.target_power = 0  # 目标功率 (MW)
        self.voltage = 0  # 电压 (V)
        self.current = 0  # 电流 (A)
        self.energy_stored = capacity * 0.5  # 当前存储能量 (MWh)

        # 历史记录
        self.history_soc = []
        self.history_power = []
        self.history_target_power = []
        self.history_energy = []
        self.history_cost = []
        self.history_voltage = []
        self.history_current = []

        # 成本统计
        self.total_op_cost = 0  # 总运维成本
        self.total_energy_cost = 0  # 总能源成本
        self.total_gas_cost = 0  # 总燃气成本（仅CAES）

    def update_state(self, target_power, dt, hour=None):
        """更新设备状态（带爬坡限制）"""
        self.target_power = target_power

        # 计算最大功率变化（考虑爬坡率和响应时间）
        max_power_change = min(
            self.ramp_rate * dt,
            self.power_rating * dt / self.response_time
        )

        # 缓慢调整到目标功率
        power_diff = target_power - self.current_power
        if abs(power_diff) > max_power_change:
            power_change = np.sign(power_diff) * max_power_change
            actual_power = self.current_power + power_change
        else:
            actual_power = target_power

        # 功率限制
        actual_power = np.clip(actual_power, -self.power_rating, self.power_rating)
        self.current_power = actual_power

        # 能量更新（考虑效率）
        if actual_power >= 0:  # 放电
            energy_change = actual_power * dt / 3600 * self.efficiency
        else:  # 充电
            energy_change = actual_power * dt / 3600 / self.efficiency

        self.energy_stored -= energy_change

        # SOC边界保护
        soc_min = 0.1
        soc_max = 0.9
        if self.energy_stored < soc_min * self.capacity:
            self.energy_stored = soc_min * self.capacity
            self.current_power = 0
        elif self.energy_stored > soc_max * self.capacity:
            self.energy_stored = soc_max * self.capacity
            self.current_power = 0

        self.soc = self.energy_stored / self.capacity

        # 计算电气参数
        self.calculate_electrical()

        # 计算成本
        cost = self.calculate_cost(actual_power, dt, hour)

        # 记录历史数据
        self.history_soc.append(self.soc)
        self.history_power.append(self.current_power)
        self.history_target_power.append(self.target_power)
        self.history_energy.append(self.energy_stored)
        self.history_cost.append(cost)
        self.history_voltage.append(self.voltage)
        self.history_current.append(self.current)

        return self.soc, cost, self.energy_stored

    def calculate_electrical(self):
        """计算电气参数 - 由子类实现"""
        pass

    def calculate_cost(self, power, dt, hour):
        """计算成本（增强版）"""
        # 运维成本（与功率成正比）
        op_cost = abs(power) * dt / 3600 * self.cost_params.device_opex[self.name]
        self.total_op_cost += op_cost

        # 电价成本/收益
        if hour is not None:
            electricity_price = self.cost_params.get_electricity_price(hour)
        else:
            electricity_price = 0.62  # 默认平段电价

        if power < 0:  # 充电（成本）
            energy_cost = abs(power) * dt / 3600 * electricity_price * 1000
            self.total_energy_cost += energy_cost
            total_cost = op_cost + energy_cost
        elif power > 0:  # 放电（收益）
            energy_revenue = power * dt / 3600 * electricity_price * 1000
            self.total_energy_cost -= energy_revenue  # 负成本表示收益
            total_cost = op_cost - energy_revenue
        else:
            total_cost = op_cost

        return total_cost

    def get_available_charge(self):
        """获取可充电量 (MWh)"""
        return (0.9 - self.soc) * self.capacity

    def get_available_discharge(self):
        """获取可放电量 (MWh)"""
        return (self.soc - 0.1) * self.capacity

    def get_power_ramp_limit(self, dt):
        """获取功率爬坡限制"""
        return self.ramp_rate * dt

    def get_status_report(self):
        """获取设备状态报告"""
        return {
            '设备名称': self.name,
            '额定功率': f"{self.power_rating:.2f} MW",
            '额定容量': f"{self.capacity:.2f} MWh",
            '当前SOC': f"{self.soc * 100:.1f}%",
            '当前功率': f"{self.current_power:.2f} MW",
            '当前电压': f"{self.voltage:.1f} V",
            '当前电流': f"{self.current:.1f} A",
            '可充电量': f"{self.get_available_charge():.2f} MWh",
            '可放电量': f"{self.get_available_discharge():.2f} MWh",
            '运维成本': f"{self.total_op_cost:.2f} 元",
            '能源成本': f"{self.total_energy_cost:.2f} 元"
        }


# ==================== 具体的储能模型（增强版） ====================
class BESS(EnergyStorageDevice):
    """锂电池储能（增强版）"""

    def __init__(self, power_rating, capacity, cost_params):
        super().__init__(
            name="锂电池储能",
            power_rating=power_rating,
            capacity=capacity,
            efficiency=0.92,
            cost_params=cost_params,
            response_time=0.1,  # 0.1秒响应
            ramp_rate=power_rating * 2  # 2倍额定功率/秒爬坡率
        )
        self.voltage_nominal = 400  # V
        self.internal_resistance = 0.001  # Ω
        self.voltage_soc_slope = 0.4  # 电压随SOC变化的斜率

    def calculate_electrical(self):
        """计算电气参数（带平滑变化）"""
        # 电压随SOC变化（线性关系）
        base_voltage = self.voltage_nominal * 0.8
        soc_voltage = self.voltage_nominal * self.voltage_soc_slope * self.soc
        target_voltage = base_voltage + soc_voltage

        # 缓慢调整电压（避免突变）
        if hasattr(self, 'last_voltage'):
            voltage_diff = target_voltage - self.last_voltage
            max_voltage_change = self.voltage_nominal * 0.01  # 每秒1%变化
            if abs(voltage_diff) > max_voltage_change:
                target_voltage = self.last_voltage + np.sign(voltage_diff) * max_voltage_change

        # 电流计算
        if abs(self.current_power) > 0.001:
            target_current = self.current_power * 1e6 / target_voltage  # A
        else:
            target_current = 0

        # 缓慢调整电流（避免突变）
        if hasattr(self, 'last_current'):
            current_diff = target_current - self.last_current
            max_current_change = self.power_rating * 1e6 / target_voltage * 0.02  # 每秒2%变化
            if abs(current_diff) > max_current_change:
                target_current = self.last_current + np.sign(current_diff) * max_current_change

        # 考虑内阻压降
        voltage_drop = target_current * self.internal_resistance
        if self.current_power > 0:  # 放电
            target_voltage -= voltage_drop
        elif self.current_power < 0:  # 充电
            target_voltage += voltage_drop

        self.voltage = target_voltage
        self.current = target_current

        # 保存当前值供下次使用
        self.last_voltage = self.voltage
        self.last_current = self.current


class SC(EnergyStorageDevice):
    """超级电容器（增强版）"""

    def __init__(self, power_rating, capacity, cost_params):
        super().__init__(
            name="超级电容器",
            power_rating=power_rating,
            capacity=capacity,
            efficiency=0.95,
            cost_params=cost_params,
            response_time=0.01,  # 0.01秒响应
            ramp_rate=power_rating * 5  # 5倍额定功率/秒爬坡率
        )
        self.voltage_nominal = 300  # V
        self.max_voltage = 330  # V
        self.min_voltage = 150  # V
        self.last_voltage = self.voltage_nominal
        self.last_current = 0

    def calculate_electrical(self):
        """计算电气参数（带平滑变化）"""
        # 超级电容器电压与SOC关系明显
        target_voltage = self.min_voltage + (self.max_voltage - self.min_voltage) * self.soc

        # 缓慢调整电压
        voltage_diff = target_voltage - self.last_voltage
        max_voltage_change = self.voltage_nominal * 0.05  # 每秒5%变化
        if abs(voltage_diff) > max_voltage_change:
            target_voltage = self.last_voltage + np.sign(voltage_diff) * max_voltage_change

        # 电流计算
        if abs(self.current_power) > 0.001:
            target_current = self.current_power * 1e6 / target_voltage
        else:
            target_current = 0

        # 缓慢调整电流
        current_diff = target_current - self.last_current
        max_current_change = self.power_rating * 1e6 / target_voltage * 0.1  # 每秒10%变化
        if abs(current_diff) > max_current_change:
            target_current = self.last_current + np.sign(current_diff) * max_current_change

        self.voltage = target_voltage
        self.current = target_current

        # 保存当前值
        self.last_voltage = self.voltage
        self.last_current = self.current


class FESS(EnergyStorageDevice):
    """飞轮储能（增强版）"""

    def __init__(self, power_rating, capacity, cost_params):
        super().__init__(
            name="飞轮储能",
            power_rating=power_rating,
            capacity=capacity,
            efficiency=0.90,
            cost_params=cost_params,
            response_time=0.05,  # 0.05秒响应
            ramp_rate=power_rating * 3  # 3倍额定功率/秒爬坡率
        )
        self.voltage_nominal = 480  # V
        self.last_voltage = self.voltage_nominal
        self.last_current = 0

    def calculate_electrical(self):
        """计算电气参数（带平滑变化）"""
        # 飞轮电压相对稳定，随SOC轻微变化
        target_voltage = self.voltage_nominal * (0.95 + 0.1 * self.soc)

        # 缓慢调整电压
        voltage_diff = target_voltage - self.last_voltage
        max_voltage_change = self.voltage_nominal * 0.03  # 每秒3%变化
        if abs(voltage_diff) > max_voltage_change:
            target_voltage = self.last_voltage + np.sign(voltage_diff) * max_voltage_change

        # 电流计算
        if abs(self.current_power) > 0.001:
            target_current = self.current_power * 1e6 / target_voltage
        else:
            target_current = 0

        # 缓慢调整电流
        current_diff = target_current - self.last_current
        max_current_change = self.power_rating * 1e6 / target_voltage * 0.05  # 每秒5%变化
        if abs(current_diff) > max_current_change:
            target_current = self.last_current + np.sign(current_diff) * max_current_change

        self.voltage = target_voltage
        self.current = target_current

        # 保存当前值
        self.last_voltage = self.voltage
        self.last_current = self.current


class SMES(EnergyStorageDevice):
    """超导磁储能（增强版）"""

    def __init__(self, power_rating, capacity, cost_params):
        super().__init__(
            name="超导磁储能",
            power_rating=power_rating,
            capacity=capacity,
            efficiency=0.97,
            cost_params=cost_params,
            response_time=0.001,  # 0.001秒响应
            ramp_rate=power_rating * 10  # 10倍额定功率/秒爬坡率
        )
        self.voltage_nominal = 600  # V
        self.critical_current = 10000  # A
        self.last_voltage = self.voltage_nominal
        self.last_current = 0

    def calculate_electrical(self):
        """计算电气参数（带平滑变化）"""
        # 超导电压非常稳定
        target_voltage = self.voltage_nominal

        # 缓慢调整电压
        voltage_diff = target_voltage - self.last_voltage
        max_voltage_change = self.voltage_nominal * 0.02  # 每秒2%变化
        if abs(voltage_diff) > max_voltage_change:
            target_voltage = self.last_voltage + np.sign(voltage_diff) * max_voltage_change

        # 电流计算
        if abs(self.current_power) > 0.001:
            target_current = self.current_power * 1e6 / target_voltage
            # 限制电流不超过临界电流
            if abs(target_current) > self.critical_current:
                target_current = np.sign(target_current) * self.critical_current
                self.current_power = target_current * target_voltage / 1e6
        else:
            target_current = 0

        # 缓慢调整电流
        current_diff = target_current - self.last_current
        max_current_change = self.power_rating * 1e6 / target_voltage * 0.08  # 每秒8%变化
        if abs(current_diff) > max_current_change:
            target_current = self.last_current + np.sign(current_diff) * max_current_change

        self.voltage = target_voltage
        self.current = target_current

        # 保存当前值
        self.last_voltage = self.voltage
        self.last_current = self.current


class CAES(EnergyStorageDevice):
    """压缩空气储能（增强版）"""

    def __init__(self, power_rating, capacity, cost_params):
        super().__init__(
            name="压缩空气储能",
            power_rating=power_rating,
            capacity=capacity,
            efficiency=0.65,
            cost_params=cost_params,
            response_time=1.0,  # 1秒响应
            ramp_rate=power_rating * 0.5  # 0.5倍额定功率/秒爬坡率
        )
        self.voltage_nominal = 10000  # V（高压）
        self.gas_consumption_rate = 0.3  # 立方米/MWh
        self.last_voltage = self.voltage_nominal
        self.last_current = 0

    def calculate_electrical(self):
        """计算电气参数（带平滑变化）"""
        # CAES电压稳定
        target_voltage = self.voltage_nominal

        # 缓慢调整电压
        voltage_diff = target_voltage - self.last_voltage
        max_voltage_change = self.voltage_nominal * 0.01  # 每秒1%变化
        if abs(voltage_diff) > max_voltage_change:
            target_voltage = self.last_voltage + np.sign(voltage_diff) * max_voltage_change

        # 电流计算
        if abs(self.current_power) > 0.001:
            target_current = self.current_power * 1e6 / target_voltage
        else:
            target_current = 0

        # 缓慢调整电流
        current_diff = target_current - self.last_current
        max_current_change = self.power_rating * 1e6 / target_voltage * 0.02  # 每秒2%变化
        if abs(current_diff) > max_current_change:
            target_current = self.last_current + np.sign(current_diff) * max_current_change

        self.voltage = target_voltage
        self.current = target_current

        # 保存当前值
        self.last_voltage = self.voltage
        self.last_current = self.current

    def calculate_cost(self, power, dt, hour):
        """重写成本计算，包括天然气成本"""
        cost = super().calculate_cost(power, dt, hour)

        # 添加天然气成本（仅放电时）
        if power > 0:
            gas_used = power * dt / 3600 * self.gas_consumption_rate  # 立方米
            gas_cost = gas_used * self.cost_params.gas_price * 1000  # 转换为元
            cost += gas_cost
            self.total_gas_cost += gas_cost

        return cost

    def get_status_report(self):
        """获取设备状态报告（包含燃气成本）"""
        report = super().get_status_report()
        report['燃气成本'] = f"{self.total_gas_cost:.2f} 元"
        return report


# ==================== 分层MPC控制器（增强版） ====================
class EnhancedHierarchicalMPC:
    """增强版分层模型预测控制器"""

    def __init__(self, prediction_horizon=10, control_horizon=5):
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.frequency_bands = {
            'ultra_high': 0.1,  # 超高频 (>10Hz): SMES
            'high': 0.3,  # 高频 (1-10Hz): SC
            'medium': 0.5,  # 中频 (0.1-1Hz): FESS
            'low': 0.8,  # 低频 (0.01-0.1Hz): BESS
            'ultra_low': 1.0  # 超低频 (<0.01Hz): CAES
        }

    def wavelet_packet_decomposition(self, signal, dt):
        """小波包分解（增强版）"""
        n = len(signal)

        # 使用多级滤波器组进行频率分解
        t = np.arange(n) * dt

        # 超高频分量（极快变化）
        f_uh = 10  # Hz
        ultra_high = signal * 0.1 * np.sin(2 * np.pi * f_uh * t)
        ultra_high = gaussian_filter1d(ultra_high, sigma=0.1 / dt)

        # 高频分量（快速变化）
        f_h = 2  # Hz
        high = signal * 0.3 * np.sin(2 * np.pi * f_h * t)
        high = gaussian_filter1d(high, sigma=0.5 / dt)

        # 中频分量（中等速度变化）
        medium = gaussian_filter1d(signal, sigma=1 / dt) - gaussian_filter1d(signal, sigma=2 / dt)

        # 低频分量（缓慢变化）
        low = gaussian_filter1d(signal, sigma=2 / dt) - gaussian_filter1d(signal, sigma=5 / dt)

        # 超低频分量（非常缓慢变化）
        ultra_low = gaussian_filter1d(signal, sigma=5 / dt)

        # 确保总和等于原始信号
        total = ultra_high + high + medium + low + ultra_low
        if np.abs(np.sum(total)) > 0.001:
            scale_factor = np.sum(signal) / np.sum(total)
            ultra_high *= scale_factor
            high *= scale_factor
            medium *= scale_factor
            low *= scale_factor
            ultra_low *= scale_factor

        return {
            'ultra_high': ultra_high,
            'high': high,
            'medium': medium,
            'low': low,
            'ultra_low': ultra_low
        }

    def economic_dispatch(self, devices, power_demand, time_index, dt, hour):
        """上层经济调度（增强版）"""
        n_devices = len(devices)

        def objective(x):
            """目标函数：总成本最小 + 平滑惩罚"""
            total_cost = 0
            smooth_penalty = 0

            for i, device in enumerate(devices):
                power = x[i]

                # 设备运维成本
                op_cost = device.cost_params.device_opex[device.name] * abs(power) * dt / 3600

                # 电价成本/收益
                electricity_price = device.cost_params.get_electricity_price(hour)
                if power < 0:  # 充电
                    energy_cost = abs(power) * dt / 3600 * electricity_price * 1000
                elif power > 0:  # 放电
                    energy_cost = -power * dt / 3600 * electricity_price * 1000  # 负成本表示收益
                else:
                    energy_cost = 0

                # 设备寿命损耗（与功率平方成正比）
                life_cost = (power / device.power_rating) ** 2 * 5

                # 平滑惩罚（减少功率突变）
                if hasattr(device, 'last_optimized_power'):
                    power_change = abs(power - device.last_optimized_power)
                    smooth_penalty += power_change * 0.1

                total_cost += op_cost + energy_cost + life_cost

            # 添加SOC平衡惩罚
            soc_penalty = 0
            for device in devices:
                soc_penalty += (device.soc - 0.5) ** 2 * 10

            return total_cost + smooth_penalty + soc_penalty

        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - power_demand}
        ]

        # 边界条件（考虑SOC和爬坡率限制）
        bounds = []
        initial_guess = []
        for device in devices:
            # 考虑SOC限制的功率边界
            max_charge = device.get_available_charge() * 3600 / dt
            max_discharge = device.get_available_discharge() * 3600 / dt

            # 考虑爬坡率限制
            ramp_limit = device.get_power_ramp_limit(dt)
            if hasattr(device, 'last_optimized_power'):
                current_power = device.last_optimized_power
                min_power = max(-device.power_rating, current_power - ramp_limit, -max_charge)
                max_power = min(device.power_rating, current_power + ramp_limit, max_discharge)
            else:
                min_power = max(-device.power_rating, -max_charge)
                max_power = min(device.power_rating, max_discharge)

            bounds.append((min_power, max_power))
            initial_guess.append(0)

        # 优化求解
        result = minimize(objective, initial_guess, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-8, 'disp': False})

        # 保存优化结果供下次使用
        if result.success:
            for i, device in enumerate(devices):
                device.last_optimized_power = result.x[i]
            return result.x
        else:
            # 如果优化失败，使用按比例分配的策略
            return self.proportional_allocation(devices, power_demand)

    def proportional_allocation(self, devices, power_demand):
        """按比例分配功率（备用策略）"""
        n_devices = len(devices)
        if n_devices == 0:
            return np.array([])

        # 按设备额定功率比例分配
        ratings = np.array([d.power_rating for d in devices])
        weights = ratings / np.sum(ratings)

        # 考虑SOC限制调整权重
        soc_adjustments = np.array([1.0 - abs(d.soc - 0.5) for d in devices])
        adjusted_weights = weights * soc_adjustments
        adjusted_weights = adjusted_weights / np.sum(adjusted_weights)

        return power_demand * adjusted_weights

    def real_time_balance(self, devices, power_demand, dt, hour):
        """下层实时平衡（增强版）"""
        # 频率分解
        power_array = np.full(self.prediction_horizon, power_demand)
        freq_components = self.wavelet_packet_decomposition(power_array, dt)

        # 按设备类型分组
        device_groups = {}
        for device in devices:
            if device.name not in device_groups:
                device_groups[device.name] = []
            device_groups[device.name].append(device)

        # 按频率分配功率
        power_allocation = {}

        # 超高频分量 -> SMES
        if '超导磁储能' in device_groups:
            ultra_high_power = freq_components['ultra_high'][0]
            for smes in device_groups['超导磁储能']:
                power_allocation[smes] = ultra_high_power * 0.5 / len(device_groups['超导磁储能'])

        # 高频分量 -> SC
        if '超级电容器' in device_groups:
            high_power = freq_components['high'][0]
            for sc in device_groups['超级电容器']:
                power_allocation[sc] = high_power * 0.3 / len(device_groups['超级电容器'])

        # 中频分量 -> FESS
        if '飞轮储能' in device_groups:
            medium_power = freq_components['medium'][0]
            for fess in device_groups['飞轮储能']:
                power_allocation[fess] = medium_power * 0.4 / len(device_groups['飞轮储能'])

        # 低频分量 -> BESS
        if '锂电池储能' in device_groups:
            low_power = freq_components['low'][0]
            for bess in device_groups['锂电池储能']:
                power_allocation[bess] = low_power * 0.6 / len(device_groups['锂电池储能'])

        # 超低频分量 -> CAES
        if '压缩空气储能' in device_groups:
            ultra_low_power = freq_components['ultra_low'][0]
            for caes in device_groups['压缩空气储能']:
                power_allocation[caes] = ultra_low_power * 0.8 / len(device_groups['压缩空气储能'])

        # 剩余功率按额定功率比例分配
        allocated_power = sum(power_allocation.values())
        remaining_power = power_demand - allocated_power

        if abs(remaining_power) > 0.001:
            # 计算各设备剩余容量权重
            weights = {}
            total_weight = 0
            for device in devices:
                if device not in power_allocation:
                    power_allocation[device] = 0

                # 权重 = 额定功率 * (1 - |SOC-0.5|)
                weight = device.power_rating * (1.0 - abs(device.soc - 0.5))
                weights[device] = weight
                total_weight += weight

            if total_weight > 0:
                for device in devices:
                    additional_power = remaining_power * weights[device] / total_weight
                    power_allocation[device] += additional_power

        return power_allocation


# ==================== 混合储能系统（增强版） ====================
class EnhancedHybridEnergyStorageSystem:
    """增强版混合储能系统"""

    def __init__(self, plant_power=10, plant_rating=13,
                 total_power=20, total_capacity=480):
        self.plant_power = plant_power  # 电厂输出功率 (MW)
        self.plant_rating = plant_rating  # 电厂额定功率 (MW)
        self.total_power = total_power  # 储能系统总功率 (MW)
        self.total_capacity = total_capacity  # 储能系统总容量 (MWh)

        self.cost_params = CostParameters()
        self.mpc = EnhancedHierarchicalMPC()
        self.devices = []
        self.simulation_results = None

        # 默认分配比例（基于设备特性优化）
        self.default_allocation_ratios = {
            '锂电池储能': 0.35,  # 35% - 主要能量型储能
            '超级电容器': 0.15,  # 15% - 高频功率型
            '飞轮储能': 0.15,  # 15% - 中频功率型
            '超导磁储能': 0.10,  # 10% - 超高频功率型
            '压缩空气储能': 0.25  # 25% - 低频能量型
        }

    def create_devices_from_selection(self, selected_devices, custom_ratios=None):
        """根据选择的设备和分配比例创建储能设备"""
        self.devices = []

        # 使用自定义比例或默认比例
        if custom_ratios:
            allocation_ratios = custom_ratios
        else:
            allocation_ratios = self.default_allocation_ratios

        # 归一化比例（确保总和为1）
        total_ratio = sum(allocation_ratios.get(name, 0) for name in selected_devices)
        if total_ratio > 0:
            for device_name in selected_devices:
                ratio = allocation_ratios.get(device_name, 0) / total_ratio
                power = self.total_power * ratio
                capacity = self.total_capacity * ratio

                if power > 0 and capacity > 0:
                    if device_name == "锂电池储能":
                        device = BESS(power, capacity, self.cost_params)
                    elif device_name == "超级电容器":
                        device = SC(power, capacity, self.cost_params)
                    elif device_name == "飞轮储能":
                        device = FESS(power, capacity, self.cost_params)
                    elif device_name == "超导磁储能":
                        device = SMES(power, capacity, self.cost_params)
                    elif device_name == "压缩空气储能":
                        device = CAES(power, capacity, self.cost_params)
                    else:
                        continue

                    self.devices.append(device)

    def simulate_pulse_smoothing(self, pulse_amplitude, pulse_duration,
                                 simulation_time=60, dt=0.1):
        """模拟脉冲平滑（增强版）"""
        # 创建时间序列
        time = np.arange(0, simulation_time, dt)
        n_steps = len(time)

        # 创建负荷曲线（更真实的脉冲形状）
        base_load = self.plant_power  # MW 基本负荷
        pulse_start = 20  # 第20秒开始脉冲
        pulse_end = pulse_start + pulse_duration

        # 带上升沿和下降沿的脉冲
        load_profile = base_load * np.ones_like(time)
        for i, t in enumerate(time):
            if pulse_start <= t < pulse_end:
                # 脉冲上升沿（2秒）
                if t < pulse_start + 2:
                    load_profile[i] = base_load + pulse_amplitude * (t - pulse_start) / 2
                # 脉冲下降沿（2秒）
                elif t > pulse_end - 2:
                    load_profile[i] = base_load + pulse_amplitude * (pulse_end - t) / 2
                # 脉冲平台
                else:
                    load_profile[i] = base_load + pulse_amplitude

        # 电厂出力（恒定）
        plant_power = self.plant_power * np.ones_like(time)

        # 需要储能系统平滑的功率
        power_to_smooth = load_profile - plant_power

        # 初始化结果存储
        results = {
            'time': time,
            'original_load': load_profile.copy(),
            'plant_power': plant_power.copy(),
            'power_to_smooth': power_to_smooth.copy(),
            'smoothed_load': np.zeros_like(time),
            'total_ess_power': np.zeros_like(time),
            'device_powers': {},
            'device_target_powers': {},
            'device_socs': {},
            'device_energies': {},
            'device_costs': {},
            'device_currents': {},
            'device_voltages': {},
            'power_allocation': {},
            'hour_of_day': np.zeros_like(time)  # 模拟小时
        }

        # 为每个设备初始化存储数组
        for device in self.devices:
            results['device_powers'][device] = np.zeros_like(time)
            results['device_target_powers'][device] = np.zeros_like(time)
            results['device_socs'][device] = np.zeros_like(time)
            results['device_energies'][device] = np.zeros_like(time)
            results['device_costs'][device] = np.zeros_like(time)
            results['device_currents'][device] = np.zeros_like(time)
            results['device_voltages'][device] = np.zeros_like(time)
            results['power_allocation'][device] = np.zeros_like(time)

        total_cost = 0

        # 滚动优化控制
        for t in range(n_steps):
            current_power_demand = power_to_smooth[t]
            hour = (t * dt) / 3600  # 转换为小时（用于电价计算）
            results['hour_of_day'][t] = hour

            # 分层MPC控制
            # 1. 上层经济调度
            economic_powers = self.mpc.economic_dispatch(
                self.devices, current_power_demand, t, dt, hour
            )

            # 2. 下层实时平衡
            real_time_allocation = self.mpc.real_time_balance(
                self.devices, current_power_demand, dt, hour
            )

            # 应用控制信号
            total_ess_power = 0
            for i, device in enumerate(self.devices):
                # 基础经济调度功率
                base_power = economic_powers[i] if i < len(economic_powers) else 0

                # 实时调整
                adjustment = real_time_allocation.get(device, 0)

                # 最终目标功率
                target_power = base_power + adjustment

                # 更新设备状态（带缓慢变化）
                soc, cost, energy = device.update_state(target_power, dt, hour)

                # 记录结果
                results['device_powers'][device][t] = device.current_power
                results['device_target_powers'][device][t] = target_power
                results['device_socs'][device][t] = soc
                results['device_energies'][device][t] = energy
                results['device_costs'][device][t] = cost
                results['device_currents'][device][t] = device.current
                results['device_voltages'][device][t] = device.voltage
                results['power_allocation'][device][t] = target_power

                total_ess_power += device.current_power
                total_cost += cost

            # 记录总功率和平滑后的负荷
            results['total_ess_power'][t] = total_ess_power
            results['smoothed_load'][t] = plant_power[t] + total_ess_power

        self.simulation_results = results
        self.total_cost = total_cost

        # 计算性能指标
        self.calculate_performance_metrics(results)

        return results

    def calculate_performance_metrics(self, results):
        """计算性能指标"""
        original = results['original_load']
        smoothed = results['smoothed_load']

        # RMSE（均方根误差）
        self.rmse = np.sqrt(np.mean((original - smoothed) ** 2))

        # 重合度（1 - 平均绝对误差百分比）
        abs_error = np.abs(original - smoothed)
        self.overlap_ratio = 1 - np.mean(abs_error) / np.mean(np.abs(original))

        # 脉冲消除效率
        pulse_mask = results['time'] >= 20  # 脉冲开始后
        pulse_original = original[pulse_mask]
        pulse_smoothed = smoothed[pulse_mask]
        pulse_reduction = np.abs(pulse_original - pulse_smoothed)
        self.pulse_elimination_efficiency = np.sum(pulse_reduction) / np.sum(np.abs(pulse_original))

        # 计算各设备贡献
        self.device_contributions = {}
        for device in self.devices:
            device_power = np.abs(results['device_powers'][device])
            total_ess_power = np.abs(results['total_ess_power'])
            if np.sum(total_ess_power) > 0:
                contribution = np.sum(device_power) / np.sum(total_ess_power)
                self.device_contributions[device.name] = contribution * 100

        # 计算成本明细
        self.cost_breakdown = {
            '总成本': self.total_cost,
            '运维成本': sum(device.total_op_cost for device in self.devices),
            '能源成本': sum(device.total_energy_cost for device in self.devices),
            '燃气成本': sum(getattr(device, 'total_gas_cost', 0) for device in self.devices)
        }

    def generate_report(self):
        """生成详细报告"""
        if not self.simulation_results:
            return "没有仿真数据"

        report = "=" * 80 + "\n"
        report += "混合储能系统(HESS)脉冲平滑仿真报告\n"
        report += "=" * 80 + "\n\n"

        # 系统配置
        report += "【系统配置】\n"
        report += f"电厂额定功率: {self.plant_rating} MW\n"
        report += f"电厂输出功率: {self.plant_power} MW\n"
        report += f"储能系统总功率: {self.total_power} MW\n"
        report += f"储能系统总容量: {self.total_capacity} MWh\n"
        report += f"储能设备数量: {len(self.devices)}\n\n"

        # 储能设备详情
        report += "【储能设备配置】\n"
        for device in self.devices:
            report += f"- {device.name}: {device.power_rating:.2f} MW, {device.capacity:.2f} MWh\n"
        report += "\n"

        # 性能指标
        report += "【性能指标】\n"
        report += f"平滑度(RMSE): {self.rmse:.4f} MW\n"
        report += f"负荷重合度: {self.overlap_ratio * 100:.2f}%\n"
        report += f"脉冲消除效率: {self.pulse_elimination_efficiency * 100:.2f}%\n\n"

        # 成本分析
        report += "【成本分析】\n"
        for cost_type, cost_value in self.cost_breakdown.items():
            report += f"{cost_type}: {cost_value:.2f} 元\n"
        report += "\n"

        # 设备贡献度
        report += "【设备贡献度】\n"
        for device_name, contribution in self.device_contributions.items():
            report += f"{device_name}: {contribution:.1f}%\n"
        report += "\n"

        # 设备状态汇总
        report += "【设备状态汇总】\n"
        for device in self.devices:
            status = device.get_status_report()
            for key, value in status.items():
                report += f"{key}: {value}, "
            report += "\n"

        report += "\n" + "=" * 80 + "\n"
        report += "报告生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
        report += "=" * 80

        return report

    def export_results(self, filename):
        """导出结果到CSV文件"""
        if not self.simulation_results:
            return False

        try:
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)

                # 写入标题
                headers = ['时间(s)', '原始负荷(MW)', '平滑后负荷(MW)', '电厂功率(MW)',
                           '储能总功率(MW)', '需平滑功率(MW)']

                # 添加各设备数据标题
                for device in self.devices:
                    headers.extend([
                        f'{device.name}功率(MW)',
                        f'{device.name}目标功率(MW)',
                        f'{device.name}SOC',
                        f'{device.name}电压(V)',
                        f'{device.name}电流(A)',
                        f'{device.name}储能(MWh)',
                        f'{device.name}成本(元)'
                    ])

                writer.writerow(headers)

                # 写入数据
                for i in range(len(self.simulation_results['time'])):
                    row = [
                        self.simulation_results['time'][i],
                        self.simulation_results['original_load'][i],
                        self.simulation_results['smoothed_load'][i],
                        self.simulation_results['plant_power'][i],
                        self.simulation_results['total_ess_power'][i],
                        self.simulation_results['power_to_smooth'][i]
                    ]

                    for device in self.devices:
                        row.extend([
                            self.simulation_results['device_powers'][device][i],
                            self.simulation_results['device_target_powers'][device][i],
                            self.simulation_results['device_socs'][device][i],
                            self.simulation_results['device_voltages'][device][i],
                            self.simulation_results['device_currents'][device][i],
                            self.simulation_results['device_energies'][device][i],
                            self.simulation_results['device_costs'][device][i]
                        ])

                    writer.writerow(row)

            return True
        except Exception as e:
            print(f"导出失败: {e}")
            return False


# ==================== 增强版GUI交互界面 ====================
class EnhancedHESSGUI:
    """增强版混合储能系统GUI界面"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("混合储能系统脉冲平滑控制（增强版）")
        self.root.geometry("750x650")

        # 变量初始化
        self.pulse_amplitude = tk.DoubleVar(value=8.0)
        self.pulse_duration = tk.DoubleVar(value=10.0)
        self.simulation_time = tk.DoubleVar(value=60.0)
        self.dt = tk.DoubleVar(value=0.1)

        self.selected_devices = {
            "锂电池储能": tk.BooleanVar(value=True),
            "超级电容器": tk.BooleanVar(value=True),
            "飞轮储能": tk.BooleanVar(value=True),
            "超导磁储能": tk.BooleanVar(value=False),
            "压缩空气储能": tk.BooleanVar(value=True)
        }

        self.custom_ratios = {
            "锂电池储能": tk.DoubleVar(value=35.0),
            "超级电容器": tk.DoubleVar(value=15.0),
            "飞轮储能": tk.DoubleVar(value=15.0),
            "超导磁储能": tk.DoubleVar(value=10.0),
            "压缩空气储能": tk.DoubleVar(value=25.0)
        }

        self.hess = None
        self.results = None

        self.setup_ui()

    def setup_ui(self):
        """设置UI界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 标题
        title_label = ttk.Label(main_frame, text="混合储能系统(HESS)脉冲平滑控制仿真",
                                font=("宋体", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # 左列：参数设置
        left_frame = ttk.LabelFrame(main_frame, text="仿真参数设置", padding="10")
        left_frame.grid(row=1, column=0, padx=(0, 10), sticky=(tk.W, tk.E, tk.N, tk.S))

        # 脉冲参数
        ttk.Label(left_frame, text="脉冲幅值 (MW):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Scale(left_frame, from_=1, to=20, orient="horizontal",
                  variable=self.pulse_amplitude, length=200).grid(row=0, column=1, padx=10)
        ttk.Label(left_frame, textvariable=self.pulse_amplitude).grid(row=0, column=2)

        ttk.Label(left_frame, text="脉冲持续时间 (s):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Scale(left_frame, from_=1, to=20, orient="horizontal",
                  variable=self.pulse_duration, length=200).grid(row=1, column=1, padx=10)
        ttk.Label(left_frame, textvariable=self.pulse_duration).grid(row=1, column=2)

        ttk.Label(left_frame, text="仿真时间 (s):").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Scale(left_frame, from_=30, to=120, orient="horizontal",
                  variable=self.simulation_time, length=200).grid(row=2, column=1, padx=10)
        ttk.Label(left_frame, textvariable=self.simulation_time).grid(row=2, column=2)

        ttk.Label(left_frame, text="时间步长 (s):").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Scale(left_frame, from_=0.01, to=0.5, orient="horizontal",
                  variable=self.dt, length=200).grid(row=3, column=1, padx=10)
        ttk.Label(left_frame, textvariable=self.dt).grid(row=3, column=2)

        # 中列：设备选择
        middle_frame = ttk.LabelFrame(main_frame, text="储能设备选择", padding="10")
        middle_frame.grid(row=1, column=1, padx=10, sticky=(tk.W, tk.E, tk.N, tk.S))

        row = 0
        for name, var in self.selected_devices.items():
            cb = ttk.Checkbutton(middle_frame, text=name, variable=var,
                                 command=self.update_ratio_entries)
            cb.grid(row=row, column=0, sticky=tk.W, pady=2)
            row += 1

        # 右列：分配比例
        right_frame = ttk.LabelFrame(main_frame, text="设备分配比例 (%)", padding="10")
        right_frame.grid(row=1, column=2, padx=(10, 0), sticky=(tk.W, tk.E, tk.N, tk.S))

        self.ratio_entries = {}
        row = 0
        for name, var in self.custom_ratios.items():
            ttk.Label(right_frame, text=name).grid(row=row, column=0, sticky=tk.W, pady=2)
            entry = ttk.Entry(right_frame, textvariable=var, width=8)
            entry.grid(row=row, column=1, padx=5)
            self.ratio_entries[name] = entry
            row += 1

        # 底部按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=20)

        ttk.Button(button_frame, text="开始仿真", command=self.run_simulation,
                   width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="显示图表", command=self.plot_results,
                   width=15, state='disabled').pack(side=tk.LEFT, padx=5)
        self.plot_button = button_frame.winfo_children()[1]

        ttk.Button(button_frame, text="生成报告", command=self.generate_report,
                   width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="导出数据", command=self.export_data,
                   width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="退出", command=self.root.quit,
                   width=15).pack(side=tk.LEFT, padx=5)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        # 配置权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)

        # 初始更新
        self.update_ratio_entries()

    def update_ratio_entries(self):
        """更新比例输入框状态"""
        for name, entry in self.ratio_entries.items():
            if self.selected_devices[name].get():
                entry.config(state='normal')
            else:
                entry.config(state='disabled')
                self.custom_ratios[name].set(0.0)

    def run_simulation(self):
        """运行仿真"""
        # 检查设备选择
        selected_count = sum(var.get() for var in self.selected_devices.values())
        if selected_count == 0:
            messagebox.showerror("错误", "请至少选择一个储能设备！")
            return

        # 检查比例总和
        if selected_count > 1:
            total_ratio = sum(self.custom_ratios[name].get()
                              for name, var in self.selected_devices.items()
                              if var.get())
            if abs(total_ratio - 100.0) > 0.1:
                messagebox.showwarning("警告", f"分配比例总和为{total_ratio:.1f}%，已自动调整为100%")
                # 自动调整比例
                for name, var in self.selected_devices.items():
                    if var.get():
                        self.custom_ratios[name].set(
                            self.custom_ratios[name].get() * 100 / total_ratio
                        )

        # 更新状态
        self.status_var.set("正在运行仿真...")
        self.root.update()

        try:
            # 获取参数
            pulse_amp = self.pulse_amplitude.get()
            pulse_dur = self.pulse_duration.get()
            sim_time = self.simulation_time.get()
            dt = self.dt.get()

            # 获取选择的设备
            selected_devices = [name for name, var in self.selected_devices.items()
                                if var.get()]

            # 获取分配比例
            allocation_ratios = {}
            for name in selected_devices:
                ratio = self.custom_ratios[name].get() / 100.0  # 转换为小数
                allocation_ratios[name] = ratio

            # 创建混合储能系统
            self.hess = EnhancedHybridEnergyStorageSystem(
                plant_power=10,
                plant_rating=13,
                total_power=20,
                total_capacity=480
            )

            # 创建设备
            self.hess.create_devices_from_selection(selected_devices, allocation_ratios)

            # 运行仿真
            self.results = self.hess.simulate_pulse_smoothing(
                pulse_amp, pulse_dur, sim_time, dt
            )

            # 启用绘图按钮
            self.plot_button.config(state='normal')

            self.status_var.set(f"仿真完成！重合度: {self.hess.overlap_ratio * 100:.2f}%")

            # 显示简要结果
            messagebox.showinfo("仿真完成",
                                f"仿真完成！\n"
                                f"重合度: {self.hess.overlap_ratio * 100:.2f}%\n"
                                f"脉冲消除效率: {self.hess.pulse_elimination_efficiency * 100:.2f}%\n"
                                f"总成本: {self.hess.total_cost:.2f} 元")

        except Exception as e:
            messagebox.showerror("错误", f"仿真过程中出现错误：\n{str(e)}")
            self.status_var.set("仿真失败！")

    def plot_results(self):
        """绘制结果图表"""
        if not self.hess or not self.results:
            messagebox.showerror("错误", "请先运行仿真！")
            return

        try:
            # 创建图表窗口
            fig = plt.figure(figsize=(20, 16))
            fig.suptitle('混合储能系统脉冲平滑仿真结果', fontsize=16, fontweight='bold')

            # 1. 脉冲平滑效果图
            ax1 = plt.subplot(4, 4, 1)
            ax1.plot(self.results['time'], self.results['original_load'],
                     'r-', linewidth=2.5, label='原始负荷')
            ax1.plot(self.results['time'], self.results['smoothed_load'],
                     'b--', linewidth=2.5, label='平滑后负荷', alpha=0.8)
            ax1.plot(self.results['time'], self.results['plant_power'],
                     'g-', linewidth=1.5, label='电厂出力', alpha=0.6)
            ax1.fill_between(self.results['time'],
                             self.results['original_load'],
                             self.results['smoothed_load'],
                             alpha=0.2, color='gray', label='平滑区域')
            ax1.set_xlabel('时间 (秒)')
            ax1.set_ylabel('功率 (MW)')
            ax1.set_title('脉冲平滑效果对比')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, max(self.results['original_load']) * 1.2])

            # 2. 各储能设备功率分配（实际功率）
            ax2 = plt.subplot(4, 4, 2)
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.hess.devices)))
            for idx, device in enumerate(self.hess.devices):
                powers = self.results['device_powers'][device]
                # 平滑显示
                powers_smooth = gaussian_filter1d(powers, sigma=2)
                ax2.plot(self.results['time'], powers_smooth,
                         color=colors[idx], linewidth=2, label=device.name)
            ax2.set_xlabel('时间 (秒)')
            ax2.set_ylabel('功率 (MW)')
            ax2.set_title('各储能设备实际功率输出')
            ax2.legend(loc='upper right', fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)

            # 3. 各储能设备目标功率
            ax3 = plt.subplot(4, 4, 3)
            for idx, device in enumerate(self.hess.devices):
                target_powers = self.results['device_target_powers'][device]
                target_smooth = gaussian_filter1d(target_powers, sigma=2)
                ax3.plot(self.results['time'], target_smooth,
                         color=colors[idx], linewidth=2, linestyle=':',
                         label=f'{device.name}(目标)', alpha=0.7)
            ax3.set_xlabel('时间 (秒)')
            ax3.set_ylabel('功率 (MW)')
            ax3.set_title('各储能设备目标功率')
            ax3.legend(loc='upper right', fontsize=9)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='k', linestyle='-', alpha=0.2)

            # 4. 脉冲功率分配策略饼图
            ax4 = plt.subplot(4, 4, 4)
            pulse_start = 20
            pulse_end = pulse_start + self.pulse_duration.get()
            pulse_mask = (self.results['time'] >= pulse_start) & (self.results['time'] < pulse_end)

            pulse_power_avg = {}
            for device in self.hess.devices:
                avg_power = np.mean(np.abs(self.results['device_powers'][device][pulse_mask]))
                if avg_power > 0.001:
                    pulse_power_avg[device.name] = avg_power

            if pulse_power_avg:
                labels = list(pulse_power_avg.keys())
                sizes = list(pulse_power_avg.values())
                colors_pie = plt.cm.tab20c(np.linspace(0, 1, len(labels)))
                wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors_pie,
                                                   autopct='%1.1f%%', startangle=90)
                ax4.set_title('脉冲期间功率分配比例')
                # 美化文本
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            else:
                ax4.text(0.5, 0.5, '无脉冲数据', ha='center', va='center')
                ax4.set_title('脉冲期间功率分配比例')

            # 5. 储能系统总功率
            ax5 = plt.subplot(4, 4, 5)
            ess_power = self.results['total_ess_power']
            ess_power_smooth = gaussian_filter1d(ess_power, sigma=2)
            ax5.plot(self.results['time'], ess_power_smooth, 'purple', linewidth=2.5)
            ax5.fill_between(self.results['time'], 0, ess_power_smooth,
                             where=ess_power_smooth > 0,
                             alpha=0.4, color='red', label='放电')
            ax5.fill_between(self.results['time'], 0, ess_power_smooth,
                             where=ess_power_smooth < 0,
                             alpha=0.4, color='blue', label='充电')
            ax5.set_xlabel('时间 (秒)')
            ax5.set_ylabel('功率 (MW)')
            ax5.set_title('储能系统总功率')
            ax5.legend(loc='upper right')
            ax5.grid(True, alpha=0.3)
            ax5.axhline(y=0, color='k', linestyle='-', alpha=0.2)

            # 6. 各储能设备SOC变化
            ax6 = plt.subplot(4, 4, 6)
            for idx, device in enumerate(self.hess.devices):
                socs = self.results['device_socs'][device]
                socs_smooth = gaussian_filter1d(socs, sigma=2)
                ax6.plot(self.results['time'], socs_smooth,
                         color=colors[idx], linewidth=2, label=device.name)
            ax6.set_xlabel('时间 (秒)')
            ax6.set_ylabel('SOC (0-1)')
            ax6.set_title('各储能设备SOC变化')
            ax6.legend(loc='upper right', fontsize=9)
            ax6.grid(True, alpha=0.3)
            ax6.set_ylim([0, 1])

            # 7. 电流随时间变化图（所有设备）
            ax7 = plt.subplot(4, 4, 7)
            for idx, device in enumerate(self.hess.devices):
                currents = self.results['device_currents'][device]
                currents_smooth = gaussian_filter1d(currents, sigma=3)
                # 转换为kA显示
                ax7.plot(self.results['time'], currents_smooth / 1000,
                         color=colors[idx], linewidth=2, label=device.name)
            ax7.set_xlabel('时间 (秒)')
            ax7.set_ylabel('电流 (kA)')
            ax7.set_title('各储能设备电流变化')
            ax7.legend(loc='upper right', fontsize=9)
            ax7.grid(True, alpha=0.3)

            # 8. 电压随时间变化图（所有设备）
            ax8 = plt.subplot(4, 4, 8)
            for idx, device in enumerate(self.hess.devices):
                voltages = self.results['device_voltages'][device]
                voltages_smooth = gaussian_filter1d(voltages, sigma=3)
                ax8.plot(self.results['time'], voltages_smooth,
                         color=colors[idx], linewidth=2, label=device.name)
            ax8.set_xlabel('时间 (秒)')
            ax8.set_ylabel('电压 (V)')
            ax8.set_title('各储能设备电压变化')
            ax8.legend(loc='upper right', fontsize=9)
            ax8.grid(True, alpha=0.3)

            # 9. 能量-电压关系图（锂电池）
            ax9 = plt.subplot(4, 4, 9)
            bess_devices = [d for d in self.hess.devices if d.name == "锂电池储能"]
            if bess_devices:
                bess = bess_devices[0]
                energies = self.results['device_energies'][bess]
                voltages = self.results['device_voltages'][bess]
                sc = ax9.scatter(energies, voltages, c=self.results['time'],
                                 cmap='viridis', alpha=0.7, s=30)
                ax9.set_xlabel('储存能量 (MWh)')
                ax9.set_ylabel('电压 (V)')
                ax9.set_title('锂电池储能能量-电压关系')
                plt.colorbar(sc, ax=ax9, label='时间 (秒)')
                ax9.grid(True, alpha=0.3)

            # 10. 能量-电流关系图（超级电容）
            ax10 = plt.subplot(4, 4, 10)
            sc_devices = [d for d in self.hess.devices if d.name == "超级电容器"]
            if sc_devices:
                sc_device = sc_devices[0]
                energies = self.results['device_energies'][sc_device]
                currents = self.results['device_currents'][sc_device]
                sc = ax10.scatter(energies, currents / 1000, c=self.results['time'],
                                  cmap='plasma', alpha=0.7, s=30)
                ax10.set_xlabel('储存能量 (MWh)')
                ax10.set_ylabel('电流 (kA)')
                ax10.set_title('超级电容器能量-电流关系')
                plt.colorbar(sc, ax=ax10, label='时间 (秒)')
                ax10.grid(True, alpha=0.3)

            # 11. 实时可充放电能量
            ax11 = plt.subplot(4, 4, 11)
            for idx, device in enumerate(self.hess.devices):
                # 可充电量
                chargeable = [(0.9 - soc) * device.capacity
                              for soc in self.results['device_socs'][device]]
                # 可放电量
                dischargeable = [(soc - 0.1) * device.capacity
                                 for soc in self.results['device_socs'][device]]

                ax11.plot(self.results['time'], chargeable, '--',
                          color=colors[idx], linewidth=1.5,
                          label=f'{device.name}可充电量')
                ax11.plot(self.results['time'], dischargeable, '-',
                          color=colors[idx], linewidth=1.5,
                          label=f'{device.name}可放电量', alpha=0.7)

            ax11.set_xlabel('时间 (秒)')
            ax11.set_ylabel('能量 (MWh)')
            ax11.set_title('实时可充放电能量')
            ax11.legend(loc='upper right', fontsize=7)
            ax11.grid(True, alpha=0.3)

            # 12. 成本累计图
            ax12 = plt.subplot(4, 4, 12)
            cumulative_costs = {}
            for device in self.hess.devices:
                costs = np.cumsum(self.results['device_costs'][device])
                cumulative_costs[device.name] = costs
                ax12.plot(self.results['time'], costs,
                          color=colors[self.hess.devices.index(device)],
                          linewidth=2, label=device.name)

            ax12.set_xlabel('时间 (秒)')
            ax12.set_ylabel('累计成本 (元)')
            ax12.set_title('各设备累计成本')
            ax12.legend(loc='upper left', fontsize=8)
            ax12.grid(True, alpha=0.3)

            # 13. 功率误差分析
            ax13 = plt.subplot(4, 4, 13)
            power_error = self.results['original_load'] - self.results['smoothed_load']
            ax13.plot(self.results['time'], power_error, 'r-', linewidth=1.5)
            ax13.fill_between(self.results['time'], 0, power_error,
                              where=power_error > 0, alpha=0.3, color='red', label='正误差')
            ax13.fill_between(self.results['time'], 0, power_error,
                              where=power_error < 0, alpha=0.3, color='blue', label='负误差')
            ax13.axhline(y=0, color='k', linestyle='-', alpha=0.5)
            ax13.set_xlabel('时间 (秒)')
            ax13.set_ylabel('功率误差 (MW)')
            ax13.set_title('平滑功率误差分析')
            ax13.legend(loc='upper right')
            ax13.grid(True, alpha=0.3)

            # 14. 设备贡献度条形图
            ax14 = plt.subplot(4, 4, 14)
            if hasattr(self.hess, 'device_contributions'):
                device_names = list(self.hess.device_contributions.keys())
                contributions = list(self.hess.device_contributions.values())
                y_pos = np.arange(len(device_names))

                bars = ax14.barh(y_pos, contributions, color=plt.cm.Set3(np.linspace(0, 1, len(device_names))))
                ax14.set_yticks(y_pos)
                ax14.set_yticklabels(device_names)
                ax14.set_xlabel('贡献度 (%)')
                ax14.set_title('各设备功率贡献度')

                # 在条形图上显示数值
                for bar, value in zip(bars, contributions):
                    width = bar.get_width()
                    ax14.text(width + 1, bar.get_y() + bar.get_height() / 2,
                              f'{value:.1f}%', ha='left', va='center', fontsize=9)

                ax14.grid(True, alpha=0.3, axis='x')

            # 15. 成本构成饼图
            ax15 = plt.subplot(4, 4, 15)
            if hasattr(self.hess, 'cost_breakdown'):
                cost_labels = list(self.hess.cost_breakdown.keys())
                cost_values = list(self.hess.cost_breakdown.values())
                # 只显示非零成本
                nonzero_indices = [i for i, v in enumerate(cost_values) if abs(v) > 0.01]
                if nonzero_indices:
                    cost_labels = [cost_labels[i] for i in nonzero_indices]
                    cost_values = [cost_values[i] for i in nonzero_indices]

                    colors_cost = plt.cm.Pastel1(np.linspace(0, 1, len(cost_labels)))
                    wedges, texts, autotexts = ax15.pie(cost_values, labels=cost_labels,
                                                        colors=colors_cost, autopct='%1.1f%%',
                                                        startangle=90)
                    ax15.set_title('成本构成分析')
                    # 美化文本
                    for autotext in autotexts:
                        autotext.set_color('black')
                        autotext.set_fontweight('bold')

            # 16. 性能指标汇总
            ax16 = plt.subplot(4, 4, 16)
            ax16.axis('off')

            performance_text = f"性能指标汇总:\n\n"
            performance_text += f"重合度: {self.hess.overlap_ratio * 100:.2f}%\n"
            performance_text += f"RMSE: {self.hess.rmse:.4f} MW\n"
            performance_text += f"脉冲消除效率: {self.hess.pulse_elimination_efficiency * 100:.2f}%\n"
            performance_text += f"总成本: {self.hess.total_cost:.2f} 元\n\n"
            performance_text += f"设备数量: {len(self.hess.devices)}\n"
            performance_text += f"仿真时长: {self.results['time'][-1]:.1f} 秒\n"
            performance_text += f"时间步长: {self.dt.get():.3f} 秒"

            ax16.text(0.1, 0.5, performance_text, fontsize=11,
                      verticalalignment='center', transform=ax16.transAxes,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror("绘图错误", f"绘制图表时出现错误：\n{str(e)}")

    def generate_report(self):
        """生成报告"""
        if not self.hess:
            messagebox.showerror("错误", "请先运行仿真！")
            return

        try:
            report = self.hess.generate_report()

            # 创建报告窗口
            report_window = tk.Toplevel(self.root)
            report_window.title("仿真报告")
            report_window.geometry("800x600")

            # 文本框架
            text_frame = ttk.Frame(report_window)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # 文本区域
            text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("宋体", 10))
            scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
            text_widget.config(yscrollcommand=scrollbar.set)

            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            # 插入报告内容
            text_widget.insert(tk.END, report)
            text_widget.config(state=tk.DISABLED)  # 设置为只读

            # 保存按钮
            save_button = ttk.Button(report_window, text="保存报告",
                                     command=lambda: self.save_text_to_file(text_widget.get("1.0", tk.END)))
            save_button.pack(pady=10)

        except Exception as e:
            messagebox.showerror("错误", f"生成报告时出现错误：\n{str(e)}")

    def save_text_to_file(self, text):
        """保存文本到文件"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text)
                messagebox.showinfo("成功", "报告保存成功！")
            except Exception as e:
                messagebox.showerror("错误", f"保存文件时出现错误：\n{str(e)}")

    def export_data(self):
        """导出数据"""
        if not self.hess:
            messagebox.showerror("错误", "请先运行仿真！")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")],
            initialfile=f"hess_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        if filename:
            success = self.hess.export_results(filename)
            if success:
                messagebox.showinfo("成功", f"数据已导出到：\n{filename}")
            else:
                messagebox.showerror("错误", "数据导出失败！")


# ==================== 主程序 ====================
def main():
    """主函数"""
    print("=" * 80)
    print("混合储能系统(HESS)脉冲平滑控制仿真（增强版）")
    print("=" * 80)
    print("\n系统配置：")
    print("- 电厂额定功率: 13 MW")
    print("- 电厂输出功率: 10 MW")
    print("- 储能系统总功率: 20 MW")
    print("- 储能系统总容量: 480 MWh")
    print("- 脉冲幅值范围: 1-20 MW")
    print("- 脉冲持续时间: 1-20 s")
    print("=" * 80)

    # 创建并运行GUI
    app = EnhancedHESSGUI()
    app.root.mainloop()


if __name__ == "__main__":
    main()
