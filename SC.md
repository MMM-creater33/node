
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题

class Supercapacitor:
    def __init__(self, capacitance, esr, max_voltage, min_voltage):
        # 初始化超级电容器参数
        self.capacitance = capacitance  # 电容量 (F)
        self.esr = esr  # 等效串联电阻 (Ω)
        self.max_voltage = max_voltage  # 最大工作电压 (V)
        self.min_voltage = min_voltage  # 最小工作电压 (V)
        self.current_voltage = 0  # 当前电压 (V)

    def calculate_energy(self, voltage=None):
        # 计算储能公式: E = 0.5 * C * V^2
        if voltage is None:
            voltage = self.current_voltage
        return 0.5 * self.capacitance * voltage ** 2

    def calculate_usable_energy(self):
        # 计算可用能量公式: E_usable = 0.5 * C * (V_max^2 - V_min^2)
        return 0.5 * self.capacitance * (self.max_voltage ** 2 - self.min_voltage ** 2)

    def calculate_peak_power(self):
        # 计算峰值功率公式: P_max = V^2 / (4 * R_ESR)
        return self.max_voltage ** 2 / (4 * self.esr)

    def charge(self, voltage):
        # 充电方法
        if voltage <= self.max_voltage:
            self.current_voltage = voltage
        else:
            self.current_voltage = self.max_voltage

    def discharge(self, current, time):
        # 放电方法
        voltage_drop = current * self.esr  # 电压降
        self.current_voltage -= voltage_drop
        if self.current_voltage < self.min_voltage:
            self.current_voltage = self.min_voltage


# 创建超级电容器实例
capacitor = Supercapacitor(capacitance=3000, esr=0.01, max_voltage=2.7, min_voltage=1.0)

# 模拟充电过程
capacitor.charge(2.7)
print(f"充电后能量: {capacitor.calculate_energy():.2f} J")
print(f"可用能量: {capacitor.calculate_usable_energy():.2f} J")
print(f"峰值功率: {capacitor.calculate_peak_power():.2f} W")

# 模拟放电过程
capacitor.discharge(current=100, time=1)
print(f"放电后电压: {capacitor.current_voltage:.2f} V")
print(f"放电后能量: {capacitor.calculate_energy():.2f} J")

# 绘制电压变化曲线
voltages = np.linspace(capacitor.min_voltage, capacitor.max_voltage, 100)
energies = [capacitor.calculate_energy(v) for v in voltages]

plt.figure(figsize=(10, 6))
plt.plot(voltages, energies)
plt.title('超级电容器能量-电压关系')
plt.xlabel('电压 (V)')
plt.ylabel('能量 (J)')
plt.grid(True)
plt.show()
