import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import sys
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 储能设备基类 ====================
class EnergyStorageDevice:
    """储能设备基类"""

    def __init__(self, name, power_rating, capacity, efficiency, cost_per_kwh,
                 cost_per_kw, op_cost, response_time):
        self.name = name
        self.power_rating = power_rating  # MW
        self.capacity = capacity  # MWh
        self.efficiency = efficiency
        self.cost_per_kwh = cost_per_kwh  # 元/kWh
        self.cost_per_kw = cost_per_kw  # 元/kW
        self.op_cost = op_cost  # 运维成本 元/MWh
        self.response_time = response_time  # 响应时间 (s)
        self.soc = 0.5  # 初始SOC
        self.current_power = 0  # 当前功率 (MW)
        self.voltage = 0  # 电压 (V)
        self.current = 0  # 电流 (A)
        self.energy_stored = capacity * 0.5  # 当前存储能量 (MWh)
        self.history_soc = []
        self.history_power = []
        self.history_energy = []
        self.history_cost = []
        self.history_voltage = []
        self.history_current = []

    def update_state(self, power, dt):
        """更新设备状态"""
        # 考虑响应时间的功率斜坡限制
        max_power_change = self.power_rating * dt / self.response_time
        power_change = power - self.current_power
        if abs(power_change) > max_power_change:
            power_change = np.sign(power_change) * max_power_change
            power = self.current_power + power_change

        # 功率限制
        power = np.clip(power, -self.power_rating, self.power_rating)
        self.current_power = power

        # 能量更新 (考虑效率)
        if power >= 0:  # 放电
            energy_change = power * dt / 3600 * self.efficiency  # 转换为MWh
        else:  # 充电
            energy_change = power * dt / 3600 / self.efficiency

        self.energy_stored -= energy_change

        # SOC边界保护
        soc_min = 0.1
        soc_max = 0.9
        if self.energy_stored < soc_min * self.capacity:
            self.energy_stored = soc_min * self.capacity
            self.current_power = 0  # 无法继续放电
        elif self.energy_stored > soc_max * self.capacity:
            self.energy_stored = soc_max * self.capacity
            self.current_power = 0  # 无法继续充电

        self.soc = self.energy_stored / self.capacity

        # 计算电气参数
        self.calculate_electrical()

        # 计算成本
        cost = self.calculate_cost(power, dt)

        # 记录历史数据
        self.history_soc.append(self.soc)
        self.history_power.append(self.current_power)
        self.history_energy.append(self.energy_stored)
        self.history_cost.append(cost)
        self.history_voltage.append(self.voltage)
        self.history_current.append(self.current)

        return self.soc, cost, self.energy_stored

    def calculate_electrical(self):
        """计算电气参数 - 由子类实现"""
        pass

    def calculate_cost(self, power, dt):
        """计算成本"""
        # 基本运维成本
        cost = abs(power) * dt / 3600 * self.op_cost  # 转换为小时

        # 电价成本/收益（简化模型）
        hour = (len(self.history_power) * dt) / 3600
        if 0 <= hour < 6 or 11 <= hour < 13:  # 低谷时段
            electricity_price = 0.21  # 元/kWh
        elif 14 <= hour < 22:  # 高峰时段
            electricity_price = 1.12  # 元/kWh
        else:  # 平段
            electricity_price = 0.62  # 元/kWh

        if power < 0:  # 充电
            cost += abs(power) * dt / 3600 * electricity_price * 1000  # 转换为元
        elif power > 0:  # 放电
            cost -= power * dt / 3600 * electricity_price * 1000  # 收益为负成本

        return cost

    def get_available_charge(self):
        """获取可充电量 (MWh)"""
        return (0.9 - self.soc) * self.capacity

    def get_available_discharge(self):
        """获取可放电量 (MWh)"""
        return (self.soc - 0.1) * self.capacity


# ==================== 具体的储能模型 ====================
class BESS(EnergyStorageDevice):
    """锂电池储能"""

    def __init__(self, power_rating, capacity):
        super().__init__(
            name="锂电池储能",
            power_rating=power_rating,
            capacity=capacity,
            efficiency=0.92,
            cost_per_kwh=700,  # 元/kWh
            cost_per_kw=0,
            op_cost=25,  # 元/MWh
            response_time=0.1  # 0.1秒响应
        )
        self.voltage_nominal = 400  # V
        self.internal_resistance = 0.001  # Ω

    def calculate_electrical(self):
        """计算电气参数"""
        # 电流计算
        if abs(self.current_power) > 0.001:
            self.current = self.current_power * 1e6 / self.voltage_nominal  # A
        else:
            self.current = 0

        # 电压随SOC变化
        self.voltage = self.voltage_nominal * (0.8 + 0.4 * self.soc)

        # 考虑内阻压降
        voltage_drop = self.current * self.internal_resistance
        if self.current_power > 0:  # 放电
            self.voltage -= voltage_drop
        elif self.current_power < 0:  # 充电
            self.voltage += voltage_drop


class SC(EnergyStorageDevice):
    """超级电容器"""

    def __init__(self, power_rating, capacity):
        super().__init__(
            name="超级电容器",
            power_rating=power_rating,
            capacity=capacity,
            efficiency=0.95,
            cost_per_kwh=2000,  # 元/kWh
            cost_per_kw=0,
            op_cost=10,  # 元/MWh
            response_time=0.01  # 0.01秒响应
        )
        self.voltage_nominal = 300  # V
        self.max_voltage = 330  # V
        self.min_voltage = 150  # V

    def calculate_electrical(self):
        """计算电气参数"""
        # 电流计算
        if abs(self.current_power) > 0.001:
            self.current = self.current_power * 1e6 / self.voltage_nominal
        else:
            self.current = 0

        # 超级电容器电压与SOC关系更明显
        self.voltage = self.min_voltage + (self.max_voltage - self.min_voltage) * self.soc


class FESS(EnergyStorageDevice):
    """飞轮储能"""

    def __init__(self, power_rating, capacity):
        super().__init__(
            name="飞轮储能",
            power_rating=power_rating,
            capacity=capacity,
            efficiency=0.90,
            cost_per_kwh=10000,  # 元/kWh
            cost_per_kw=10000,  # 元/kW
            op_cost=15,  # 元/MWh
            response_time=0.05  # 0.05秒响应
        )
        self.voltage_nominal = 480  # V

    def calculate_electrical(self):
        """计算电气参数"""
        if abs(self.current_power) > 0.001:
            self.current = self.current_power * 1e6 / self.voltage_nominal
        else:
            self.current = 0

        # 飞轮电压相对稳定
        self.voltage = self.voltage_nominal * (0.95 + 0.1 * self.soc)


class SMES(EnergyStorageDevice):
    """超导磁储能"""

    def __init__(self, power_rating, capacity):
        super().__init__(
            name="超导磁储能",
            power_rating=power_rating,
            capacity=capacity,
            efficiency=0.97,
            cost_per_kwh=60000,  # 元/kWh
            cost_per_kw=0,
            op_cost=50,  # 元/MWh (维护成本高)
            response_time=0.001  # 0.001秒响应
        )
        self.voltage_nominal = 600  # V
        self.critical_current = 10000  # A

    def calculate_electrical(self):
        """计算电气参数"""
        if abs(self.current_power) > 0.001:
            self.current = self.current_power * 1e6 / self.voltage_nominal
            # 限制电流不超过临界电流
            if abs(self.current) > self.critical_current:
                self.current = np.sign(self.current) * self.critical_current
                self.current_power = self.current * self.voltage_nominal / 1e6
        else:
            self.current = 0

        # 超导电压非常稳定
        self.voltage = self.voltage_nominal


class CAES(EnergyStorageDevice):
    """压缩空气储能"""

    def __init__(self, power_rating, capacity):
        super().__init__(
            name="压缩空气储能",
            power_rating=power_rating,
            capacity=capacity,
            efficiency=0.65,  # 效率较低
            cost_per_kwh=3000,  # 元/kWh
            cost_per_kw=0,
            op_cost=5,  # 元/MWh
            response_time=1.0  # 1秒响应
        )
        self.gas_price = 3.6  # 元/立方米
        self.voltage_nominal = 10000  # V (高压)
        self.gas_consumption = []

    def calculate_electrical(self):
        """计算电气参数"""
        if abs(self.current_power) > 0.001:
            self.current = self.current_power * 1e6 / self.voltage_nominal
        else:
            self.current = 0

        self.voltage = self.voltage_nominal

        # 天然气消耗 (放电时需要)
        if self.current_power > 0:
            gas_used = self.current_power * 0.3  # 0.3立方米/MWh
            self.gas_consumption.append(gas_used)

    def calculate_cost(self, power, dt):
        """重写成本计算，包括天然气成本"""
        cost = super().calculate_cost(power, dt)

        # 添加天然气成本
        if power > 0:  # 放电
            gas_cost = power * dt / 3600 * self.gas_price * 1000  # 转换为元
            cost += gas_cost

        return cost


# ==================== 分层MPC控制器 ====================
class HierarchicalMPC:
    """分层模型预测控制器"""

    def __init__(self, prediction_horizon=10, control_horizon=5):
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon

    def wavelet_decomposition(self, signal, levels=3):
        """小波包分解 - 简化版本"""
        n = len(signal)

        # 低通滤波器
        low_pass = np.array([0.482962, 0.836516, 0.224144, -0.129409])
        # 高通滤波器
        high_pass = np.array([-0.129409, -0.224144, 0.836516, -0.482962])

        # 简化分解：使用移动平均作为低频，残差作为高频
        low_freq = np.convolve(signal, np.ones(10) / 10, mode='same')
        residual = signal - low_freq
        mid_freq = np.convolve(residual, np.ones(5) / 5, mode='same')
        high_freq = residual - mid_freq

        return {
            'low': low_freq,
            'mid': mid_freq,
            'high': high_freq
        }

    def economic_dispatch(self, devices, power_demand, time_index, dt):
        """上层经济调度"""
        # 构建优化问题
        n_devices = len(devices)

        def objective(x):
            """目标函数：总成本最小"""
            total_cost = 0
            for i, device in enumerate(devices):
                power = x[i]
                # 设备成本
                device_cost = device.op_cost * abs(power) * dt / 3600
                # 电价成本/收益
                hour = (time_index * dt) / 3600
                if 0 <= hour < 6 or 11 <= hour < 13:  # 低谷
                    electricity_price = 0.21
                elif 14 <= hour < 22:  # 高峰
                    electricity_price = 1.12
                else:  # 平段
                    electricity_price = 0.62

                if power < 0:  # 充电
                    device_cost += abs(power) * dt / 3600 * electricity_price * 1000
                elif power > 0:  # 放电
                    device_cost -= power * dt / 3600 * electricity_price * 1000

                # 寿命损耗（与功率平方成正比）
                device_cost += (power / device.power_rating) ** 2 * 10

                total_cost += device_cost
            return total_cost

        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - power_demand}
        ]

        # 边界条件
        bounds = []
        initial_guess = []
        for device in devices:
            # 考虑SOC限制的功率边界
            max_charge = device.get_available_charge() * 3600 / dt  # 转换为MW
            max_discharge = device.get_available_discharge() * 3600 / dt

            # 实际可用的充放电功率
            actual_max_charge = min(device.power_rating, max_charge)
            actual_max_discharge = min(device.power_rating, max_discharge)

            bounds.append((-actual_max_charge, actual_max_discharge))
            initial_guess.append(0)  # 初始猜测为0

        # 优化求解
        result = minimize(objective, initial_guess, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 100, 'ftol': 1e-6})

        return result.x if result.success else np.zeros(n_devices)

    def real_time_balance(self, devices, power_demand, dt):
        """下层实时平衡"""
        # 小波分解功率需求
        freq_components = self.wavelet_decomposition(np.array([power_demand]))

        # 按频段分配功率
        power_allocation = {}

        # 查找适合处理各频段的设备
        sc_devices = [d for d in devices if d.name == "超级电容器"]
        smes_devices = [d for d in devices if d.name == "超导磁储能"]
        fess_devices = [d for d in devices if d.name == "飞轮储能"]
        bess_devices = [d for d in devices if d.name == "锂电池储能"]
        caes_devices = [d for d in devices if d.name == "压缩空气储能"]

        # 高频分量 -> SMES和SC
        high_power = freq_components['high'][0] if len(freq_components['high']) > 0 else 0
        if smes_devices and sc_devices:
            smes_power = high_power * 0.6
            sc_power = high_power * 0.4
            power_allocation['超导磁储能'] = smes_power
            power_allocation['超级电容器'] = sc_power
        elif smes_devices:
            power_allocation['超导磁储能'] = high_power
        elif sc_devices:
            power_allocation['超级电容器'] = high_power

        # 中频分量 -> FESS和BESS
        mid_power = freq_components['mid'][0] if len(freq_components['mid']) > 0 else 0
        if fess_devices and bess_devices:
            fess_power = mid_power * 0.5
            bess_power = mid_power * 0.5
            power_allocation['飞轮储能'] = fess_power
            power_allocation['锂电池储能'] = power_allocation.get('锂电池储能', 0) + bess_power
        elif fess_devices:
            power_allocation['飞轮储能'] = mid_power
        elif bess_devices:
            power_allocation['锂电池储能'] = power_allocation.get('锂电池储能', 0) + mid_power

        # 低频分量 -> CAES和BESS
        low_power = freq_components['low'][0] if len(freq_components['low']) > 0 else 0
        if caes_devices:
            caes_power = low_power * 0.7
            power_allocation['压缩空气储能'] = caes_power
            remaining_power = low_power * 0.3
        else:
            remaining_power = low_power

        # 剩余功率分配给BESS
        if bess_devices:
            power_allocation['锂电池储能'] = power_allocation.get('锂电池储能', 0) + remaining_power

        return power_allocation


# ==================== 混合储能系统 ====================
class HybridEnergyStorageSystem:
    """混合储能系统"""

    def __init__(self, total_power=20, total_capacity=480):
        self.total_power = total_power  # MW
        self.total_capacity = total_capacity  # MWh
        self.devices = []
        self.mpc = HierarchicalMPC()
        self.simulation_results = None

    def create_devices_from_selection(self, selected_devices, allocation_ratios):
        """根据选择的设备和分配比例创建储能设备"""
        self.devices = []

        for device_name, ratio in allocation_ratios.items():
            if device_name not in selected_devices:
                continue

            power = self.total_power * ratio
            capacity = self.total_capacity * ratio

            if device_name == "锂电池储能":
                device = BESS(power, capacity)
            elif device_name == "超级电容器":
                device = SC(power, capacity)
            elif device_name == "飞轮储能":
                device = FESS(power, capacity)
            elif device_name == "超导磁储能":
                device = SMES(power, capacity)
            elif device_name == "压缩空气储能":
                device = CAES(power, capacity)
            else:
                continue

            self.devices.append(device)

    def simulate_pulse_smoothing(self, pulse_amplitude, pulse_duration,
                                 simulation_time=30, dt=0.1):
        """模拟脉冲平滑"""
        # 创建时间序列
        time = np.arange(0, simulation_time, dt)
        n_steps = len(time)

        # 创建负荷曲线
        base_load = 10  # MW 基本负荷
        pulse_start = 10  # 第10秒开始脉冲
        pulse_end = pulse_start + pulse_duration

        load_profile = base_load * np.ones_like(time)
        pulse_indices = (time >= pulse_start) & (time < pulse_end)
        load_profile[pulse_indices] = base_load + pulse_amplitude

        # 电厂出力设定为10MW（恒定）
        plant_power = 10 * np.ones_like(time)

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
            'device_socs': {},
            'device_energies': {},
            'device_costs': {},
            'device_currents': {},
            'device_voltages': {},
            'power_allocation': {}
        }

        # 为每个设备初始化存储数组
        for device in self.devices:
            results['device_powers'][device.name] = np.zeros_like(time)
            results['device_socs'][device.name] = np.zeros_like(time)
            results['device_energies'][device.name] = np.zeros_like(time)
            results['device_costs'][device.name] = np.zeros_like(time)
            results['device_currents'][device.name] = np.zeros_like(time)
            results['device_voltages'][device.name] = np.zeros_like(time)
            results['power_allocation'][device.name] = np.zeros_like(time)

        total_cost = 0

        # 滚动优化控制
        for t in range(n_steps):
            current_power_demand = power_to_smooth[t]

            # 分层MPC控制
            # 上层经济调度
            economic_powers = self.mpc.economic_dispatch(
                self.devices, current_power_demand, t, dt
            )

            # 下层实时平衡（调整分配）
            real_time_allocation = self.mpc.real_time_balance(
                self.devices, current_power_demand, dt
            )

            # 应用控制信号
            total_ess_power = 0
            for i, device in enumerate(self.devices):
                # 基础经济调度功率
                base_power = economic_powers[i]

                # 实时调整
                adjustment = real_time_allocation.get(device.name, 0)

                # 最终功率（考虑设备限制）
                final_power = base_power + adjustment

                # 更新设备状态
                soc, cost, energy = device.update_state(final_power, dt)

                # 记录结果
                results['device_powers'][device.name][t] = device.current_power
                results['device_socs'][device.name][t] = soc
                results['device_energies'][device.name][t] = energy
                results['device_costs'][device.name][t] = cost
                results['device_currents'][device.name][t] = device.current
                results['device_voltages'][device.name][t] = device.voltage
                results['power_allocation'][device.name][t] = final_power

                total_ess_power += device.current_power
                total_cost += cost

            # 记录总功率和平滑后的负荷
            results['total_ess_power'][t] = total_ess_power
            results['smoothed_load'][t] = plant_power[t] + total_ess_power

        self.simulation_results = results
        self.total_cost = total_cost

        return results


# ==================== GUI交互界面 ====================
class HESSGUI:
    """混合储能系统GUI界面"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("混合储能系统脉冲平滑控制")
        self.root.geometry("600x500")

        self.pulse_amplitude = tk.DoubleVar(value=8.0)
        self.pulse_duration = tk.DoubleVar(value=10.0)
        self.selected_devices = {
            "锂电池储能": tk.BooleanVar(value=True),
            "超级电容器": tk.BooleanVar(value=True),
            "飞轮储能": tk.BooleanVar(value=True),
            "超导磁储能": tk.BooleanVar(value=False),
            "压缩空气储能": tk.BooleanVar(value=True)
        }

        self.setup_ui()

    def setup_ui(self):
        """设置UI界面"""
        # 标题
        title_label = tk.Label(self.root, text="混合储能系统(HESS)脉冲平滑控制",
                               font=("宋体", 16, "bold"))
        title_label.pack(pady=10)

        # 脉冲参数设置
        param_frame = tk.LabelFrame(self.root, text="脉冲参数设置", padx=10, pady=10)
        param_frame.pack(padx=20, pady=10, fill="x")

        # 脉冲幅值
        amp_frame = tk.Frame(param_frame)
        amp_frame.pack(fill="x", pady=5)
        tk.Label(amp_frame, text="脉冲幅值 (MW):", width=15, anchor="w").pack(side="left")
        tk.Scale(amp_frame, from_=1, to=20, orient="horizontal",
                 variable=self.pulse_amplitude, length=300).pack(side="left", padx=10)
        tk.Label(amp_frame, textvariable=self.pulse_amplitude, width=5).pack(side="left")

        # 脉冲持续时间
        dur_frame = tk.Frame(param_frame)
        dur_frame.pack(fill="x", pady=5)
        tk.Label(dur_frame, text="脉冲持续时间 (s):", width=15, anchor="w").pack(side="left")
        tk.Scale(dur_frame, from_=1, to=20, orient="horizontal",
                 variable=self.pulse_duration, length=300).pack(side="left", padx=10)
        tk.Label(dur_frame, textvariable=self.pulse_duration, width=5).pack(side="left")

        # 储能设备选择
        device_frame = tk.LabelFrame(self.root, text="选择储能设备", padx=10, pady=10)
        device_frame.pack(padx=20, pady=10, fill="x")

        for name, var in self.selected_devices.items():
            cb = tk.Checkbutton(device_frame, text=name, variable=var,
                                font=("宋体", 11))
            cb.pack(anchor="w", pady=2)

        # 运行按钮
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)

        tk.Button(button_frame, text="开始仿真", command=self.run_simulation,
                  bg="green", fg="white", font=("宋体", 12), width=15).pack(side="left", padx=10)
        tk.Button(button_frame, text="退出", command=self.root.quit,
                  bg="red", fg="white", font=("宋体", 12), width=15).pack(side="left", padx=10)

        # 状态信息
        self.status_label = tk.Label(self.root, text="等待开始仿真...",
                                     font=("宋体", 10), fg="blue")
        self.status_label.pack(pady=10)

    def run_simulation(self):
        """运行仿真"""
        # 检查是否选择了设备
        selected_count = sum(var.get() for var in self.selected_devices.values())
        if selected_count == 0:
            messagebox.showerror("错误", "请至少选择一个储能设备！")
            return

        # 更新状态
        self.status_label.config(text="正在运行仿真...")
        self.root.update()

        try:
            # 获取参数
            pulse_amp = self.pulse_amplitude.get()
            pulse_dur = self.pulse_duration.get()

            # 获取选择的设备
            selected_devices = [name for name, var in self.selected_devices.items()
                                if var.get()]

            # 分配比例（根据设备类型自动分配）
            allocation_ratios = self.calculate_allocation_ratios(selected_devices)

            # 创建混合储能系统
            hess = HybridEnergyStorageSystem(total_power=20, total_capacity=480)
            hess.create_devices_from_selection(selected_devices, allocation_ratios)

            # 运行仿真
            results = hess.simulate_pulse_smoothing(pulse_amp, pulse_dur)

            # 显示结果
            self.show_results(hess, results, pulse_amp, pulse_dur, allocation_ratios)

            self.status_label.config(text="仿真完成！")

        except Exception as e:
            messagebox.showerror("错误", f"仿真过程中出现错误：\n{str(e)}")
            self.status_label.config(text="仿真失败！")

    def calculate_allocation_ratios(self, selected_devices):
        """计算设备分配比例"""
        n_devices = len(selected_devices)

        # 根据设备类型分配不同的权重
        weights = {
            "锂电池储能": 1.0,
            "超级电容器": 0.8,
            "飞轮储能": 0.8,
            "超导磁储能": 0.6,
            "压缩空气储能": 1.2
        }

        total_weight = sum(weights[device] for device in selected_devices)
        allocation_ratios = {device: weights[device] / total_weight
                             for device in selected_devices}

        return allocation_ratios

    def show_results(self, hess, results, pulse_amp, pulse_dur, allocation_ratios):
        """显示仿真结果"""
        # 计算性能指标
        original = results['original_load']
        smoothed = results['smoothed_load']
        rmse = np.sqrt(np.mean((original - smoothed) ** 2))
        overlap_ratio = 1 - np.sum(np.abs(original - smoothed)) / np.sum(np.abs(original))

        # 计算各设备成本
        device_costs = {}
        total_cost = 0
        for device in hess.devices:
            cost = np.sum(results['device_costs'][device.name])
            device_costs[device.name] = cost
            total_cost += cost

        # 创建结果显示窗口
        result_window = tk.Toplevel(self.root)
        result_window.title("仿真结果")
        result_window.geometry("800x600")

        # 创建文本输出区域
        text_frame = tk.Frame(result_window)
        text_frame.pack(padx=10, pady=10, fill="both", expand=True)

        text_widget = tk.Text(text_frame, wrap="word", font=("宋体", 10))
        scrollbar = tk.Scrollbar(text_frame, command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)

        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 写入结果
        text_widget.insert("end", "=" * 60 + "\n")
        text_widget.insert("end", "混合储能系统脉冲平滑仿真结果\n")
        text_widget.insert("end", "=" * 60 + "\n\n")

        text_widget.insert("end", f"脉冲幅值: {pulse_amp} MW\n")
        text_widget.insert("end", f"脉冲持续时间: {pulse_dur} 秒\n\n")

        text_widget.insert("end", "储能设备分配比例:\n")
        for device, ratio in allocation_ratios.items():
            power = 20 * ratio
            capacity = 480 * ratio
            text_widget.insert("end", f"  {device}: {ratio * 100:.1f}% "
                                      f"(功率: {power:.1f} MW, 容量: {capacity:.1f} MWh)\n")

        text_widget.insert("end", f"\n平滑度(RMSE): {rmse:.3f} MW\n")
        text_widget.insert("end", f"负荷重合度: {overlap_ratio * 100:.2f}%\n")
        text_widget.insert("end", f"总运行成本: {total_cost:.2f} 元\n\n")

        text_widget.insert("end", "各设备成本明细:\n")
        for device, cost in device_costs.items():
            text_widget.insert("end", f"  {device}: {cost:.2f} 元\n")

        text_widget.insert("end", f"\n脉冲消除策略:\n")

        # 计算脉冲期间的平均功率分配
        pulse_start = 10
        pulse_end = pulse_start + pulse_dur
        pulse_mask = (results['time'] >= pulse_start) & (results['time'] < pulse_end)

        if np.any(pulse_mask):
            text_widget.insert("end", "脉冲期间功率分配:\n")
            for device in hess.devices:
                avg_power = np.mean(np.abs(results['device_powers'][device.name][pulse_mask]))
                if avg_power > 0.01:  # 只显示有贡献的设备
                    contribution = avg_power / np.mean(np.abs(results['total_ess_power'][pulse_mask]))
                    text_widget.insert("end", f"  {device.name}: {contribution * 100:.1f}%\n")

        text_widget.config(state="disabled")

        # 绘图按钮
        plot_button = tk.Button(result_window, text="显示图表",
                                command=lambda: self.plot_results(hess, results),
                                bg="blue", fg="white", font=("宋体", 12))
        plot_button.pack(pady=10)

    def plot_results(self, hess, results):
        """绘制结果图表"""
        # 创建图表窗口
        plt.figure(figsize=(16, 12))

        # 1. 脉冲平滑效果图
        plt.subplot(3, 3, 1)
        plt.plot(results['time'], results['original_load'], 'r-', linewidth=2, label='原始负荷')
        plt.plot(results['time'], results['smoothed_load'], 'b--', linewidth=2, label='平滑后负荷')
        plt.plot(results['time'], results['plant_power'], 'g-', linewidth=1.5, label='电厂出力')
        plt.fill_between(results['time'], results['original_load'], results['smoothed_load'],
                         alpha=0.3, color='gray', label='平滑区域')
        plt.xlabel('时间 (秒)')
        plt.ylabel('功率 (MW)')
        plt.title('脉冲平滑效果对比')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. 各储能设备功率分配
        plt.subplot(3, 3, 2)
        for device in hess.devices:
            powers = results['device_powers'][device.name]
            # 平滑显示
            powers_smooth = gaussian_filter1d(powers, sigma=2)
            plt.plot(results['time'], powers_smooth, label=device.name)
        plt.xlabel('时间 (秒)')
        plt.ylabel('功率 (MW)')
        plt.title('各储能设备功率分配')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. 脉冲功率分配策略饼图
        plt.subplot(3, 3, 3)
        pulse_start = 10
        pulse_end = pulse_start + results['pulse_duration'] if 'pulse_duration' in results else 20
        pulse_mask = (results['time'] >= pulse_start) & (results['time'] < pulse_end)

        pulse_power_avg = {}
        for device in hess.devices:
            avg_power = np.mean(np.abs(results['device_powers'][device.name][pulse_mask]))
            if avg_power > 0.01:
                pulse_power_avg[device.name] = avg_power

        if pulse_power_avg:
            labels = list(pulse_power_avg.keys())
            sizes = list(pulse_power_avg.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('脉冲期间功率分配比例')

        # 4. 储能系统总功率
        plt.subplot(3, 3, 4)
        ess_power = results['total_ess_power']
        ess_power_smooth = gaussian_filter1d(ess_power, sigma=2)
        plt.plot(results['time'], ess_power_smooth, 'purple', linewidth=2)
        plt.fill_between(results['time'], 0, ess_power_smooth, where=ess_power_smooth > 0,
                         alpha=0.3, color='red', label='放电')
        plt.fill_between(results['time'], 0, ess_power_smooth, where=ess_power_smooth < 0,
                         alpha=0.3, color='blue', label='充电')
        plt.xlabel('时间 (秒)')
        plt.ylabel('功率 (MW)')
        plt.title('储能系统总功率')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. 各储能设备SOC变化
        plt.subplot(3, 3, 5)
        for device in hess.devices:
            socs = results['device_socs'][device.name]
            plt.plot(results['time'], socs, label=device.name)
        plt.xlabel('时间 (秒)')
        plt.ylabel('SOC')
        plt.title('各储能设备SOC变化')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 6. 电流随时间变化图（以锂电池为例）
        plt.subplot(3, 3, 6)
        bess_devices = [d for d in hess.devices if d.name == "锂电池储能"]
        if bess_devices:
            bess = bess_devices[0]
            currents = results['device_currents'][bess.name]
            currents_smooth = gaussian_filter1d(currents, sigma=3)
            plt.plot(results['time'], currents_smooth / 1000, 'b-', linewidth=2)  # 转换为kA
            plt.xlabel('时间 (秒)')
            plt.ylabel('电流 (kA)')
            plt.title('锂电池储能电流变化')
            plt.grid(True, alpha=0.3)

        # 7. 电压随时间变化图（以锂电池为例）
        plt.subplot(3, 3, 7)
        if bess_devices:
            voltages = results['device_voltages'][bess.name]
            voltages_smooth = gaussian_filter1d(voltages, sigma=3)
            plt.plot(results['time'], voltages_smooth, 'r-', linewidth=2)
            plt.xlabel('时间 (秒)')
            plt.ylabel('电压 (V)')
            plt.title('锂电池储能电压变化')
            plt.grid(True, alpha=0.3)

        # 8. 能量-电压关系图（以超级电容为例）
        plt.subplot(3, 3, 8)
        sc_devices = [d for d in hess.devices if d.name == "超级电容器"]
        if sc_devices:
            sc = sc_devices[0]
            # 使用仿真数据绘制
            energies = results['device_energies'][sc.name]
            voltages = results['device_voltages'][sc.name]
            plt.scatter(energies, voltages, c=results['time'], cmap='viridis', alpha=0.6)
            plt.xlabel('储存能量 (MWh)')
            plt.ylabel('电压 (V)')
            plt.title('超级电容器能量-电压关系')
            plt.colorbar(label='时间 (秒)')
            plt.grid(True, alpha=0.3)

        # 9. 实时可充放电能量
        plt.subplot(3, 3, 9)
        for device in hess.devices:
            # 可充电量
            chargeable = [(0.9 - soc) * device.capacity for soc in results['device_socs'][device.name]]
            # 可放电量
            dischargeable = [(soc - 0.1) * device.capacity for soc in results['device_socs'][device.name]]

            plt.plot(results['time'], chargeable, '--', linewidth=1, label=f'{device.name}可充电量')
            plt.plot(results['time'], dischargeable, '-', linewidth=1, label=f'{device.name}可放电量')

        plt.xlabel('时间 (秒)')
        plt.ylabel('能量 (MWh)')
        plt.title('实时可充放电能量')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# ==================== 主程序 ====================
def main():
    """主函数"""
    print("=" * 60)
    print("混合储能系统(HESS)脉冲平滑控制仿真")
    print("=" * 60)

    # 创建并运行GUI
    app = HESSGUI()
    app.root.mainloop()


if __name__ == "__main__":
    main()
