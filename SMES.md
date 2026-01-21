
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题

class SuperconductingMagneticEnergyStorage:
    def __init__(self, inductance, critical_current, critical_field, critical_temperature):
        # 初始化超导磁储能参数
        self.inductance = inductance  # 电感 (H)
        self.critical_current = critical_current  # 临界电流 (A)
        self.critical_field = critical_field  # 临界磁场 (T)
        self.critical_temperature = critical_temperature  # 临界温度 (K)
        self.current = 0  # 当前电流 (A)
        self.temperature = 0  # 当前温度 (K)
        self.magnetic_field = 0  # 当前磁场 (T)

    def calculate_energy(self):
        # 计算储能量公式: E = 0.5 * L * I^2
        return 0.5 * self.inductance * self.current ** 2

    def calculate_inductance(self, n, a, l):
        # 计算电感公式: L = (μ₀ * N² * A) / l (螺线管)
        mu_0 = 4 * np.pi * 1e-7  # 真空磁导率 (H/m)
        return (mu_0 * n ** 2 * a) / l

    def check_operation_limits(self):
        # 检查运行限制条件: I < I_c, B < B_c, T < T_c
        current_limit = self.current < self.critical_current
        field_limit = self.magnetic_field < self.critical_field
        temp_limit = self.temperature < self.critical_temperature
        return current_limit and field_limit and temp_limit

    def calculate_power_exchange(self, di_dt):
        # 计算功率交换公式: P = L * I * (dI/dt)
        return self.inductance * self.current * di_dt

    def charge(self, current):
        # 充电方法
        if self.check_operation_limits():
            self.current = current
            if self.current > self.critical_current:
                self.current = self.critical_current

    def discharge(self, current):
        # 放电方法
        if self.check_operation_limits():
            self.current = current
            if self.current < 0:
                self.current = 0


# 创建超导磁储能实例
smes = SuperconductingMagneticEnergyStorage(
    inductance=100,  # 电感 (H)
    critical_current=1000,  # 临界电流 (A)
    critical_field=8,  # 临界磁场 (T)
    critical_temperature=9.2  # 临界温度 (K)
)

# 模拟充电过程
smes.charge(800)
print(f"充电后能量: {smes.calculate_energy():.2f} J")
print(f"运行限制检查: {smes.check_operation_limits()}")

# 计算电感（螺线管参数）
n = 1000  # 匝数
a = 0.1  # 面积 (m²)
l = 1.0  # 长度 (m)
inductance = smes.calculate_inductance(n, a, l)
print(f"计算电感: {inductance:.6f} H")

# 计算功率交换
di_dt = 100  # 电流变化率 (A/s)
power = smes.calculate_power_exchange(di_dt)
print(f"功率交换: {power:.2f} W")

# 绘制电流与能量关系曲线
currents = np.linspace(0, smes.critical_current, 100)
energies = [0.5 * smes.inductance * i ** 2 for i in currents]

plt.figure(figsize=(10, 6))
plt.plot(currents, energies)
plt.title('超导磁储能能量-电流关系')
plt.xlabel('电流 (A)')
plt.ylabel('能量 (J)')
plt.grid(True)
plt.show()
