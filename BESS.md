
import matplotlib.pyplot as plt
import numpy as np       #输入       #
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题
#对绘图库的配置（曲线图）

class LithiumIonBattery:       #定义一个名为L的类
    def __init__(self, nominal_voltage, capacity, internal_resistance):
        #"-int-":类的构造函数,此代码首先运行，来设置初始参数，定义基本参数
        # 初始化锂离子电池参数
        self.nominal_voltage = nominal_voltage  # 标称电压/额定电压 (V)
        self.capacity = capacity  # 容量\存多少电 (Ah)
        self.internal_resistance = internal_resistance  # 内阻 (Ω)
        self.initial_soc = 1.0  # 初始荷电状态 (0-1)，1.0表示一开始是满电状态
        self.current_soc = self.initial_soc  # 当前荷电状态 (0-1)
        self.current = 0  # 当前电流 (A)
        self.time = 0  # 当前时间 (s)

    def calculate_energy(self):
        #计算电池总能量
        # 计算能量公式: E = V_nom * Q
        return self.nominal_voltage * self.capacity       #返回理论最大能量（VAh)


    def calculate_capacity(self, current, time):
        # 计算容量公式: Q = I * t
        return current * time

#核心算法-计算荷电状态（SOC）
    def calculate_soc(self, current, time):
        #self(代表对象自己）,current（电流）,time（时间）
        #模拟电池核心动态的关键部分，用于更新电池的剩余电量百分比
        # 计算荷电状态公式: SOC(t) = SOC₀ - (1/C_N) ∫I(t)dt
        charge_change = (current * time) / 3600  # 转换为 Ah
        # #charge-change计算这段时间内电量变化量（用了多少电），charge充电，电量
        self.current_soc = self.current_soc - charge_change / self.capacity
        #self.current-soc:更新当前的SOC
        if self.current_soc < 0:     #边界检查，防止电量算出是负值
            self.current_soc = 0     #强制电量锁定为0，不再减少
        return self.current_soc

#辅助功能-C倍率、电压计算
    def calculate_c_rate(self, current):   #定义计算C-rate（充放电倍率）的函数
        # 计算 C-Rate 公式: I = C-Rate * C_N
        return current / self.capacity

    def calculate_terminal_voltage(self, ocv): #定义计算端电压的函数
        # 计算端电压公式: V_t = OCV - I * R_int
        return ocv - self.current * self.internal_resistance

#充放电操作封装
    def charge(self, current, time):  #定义充电方法
        # 充电方法
        self.current = current
        self.time = time
        self.calculate_soc(-current, time)  # 负电流表示充电     #负电流，正充电，SOC增加

    def discharge(self, current, time):
        # 放电方法
        self.current = current
        self.time = time
        self.calculate_soc(current, time)  # 正电流表示放电

#创建电池对象
# 创建锂离子电池实例
battery = LithiumIonBattery(      #battery赋的名字
    nominal_voltage=3.7,  # 标称电压 (V)
    capacity=2.0,  # 容量 (Ah)
    internal_resistance=0.01  # 内阻 (Ω)
)


# 模拟充电过程
battery.charge(current=1.0, time=3600)  # 1A 充电 1 小时
print(f"充电后 SOC: {battery.current_soc:.2f}")
print(f"充电后容量: {battery.calculate_capacity(battery.current, battery.time / 3600):.2f} Ah")

# 模拟放电过程
battery.discharge(current=1.0, time=3600)  # 1A 放电 1 小时
print(f"放电后 SOC: {battery.current_soc:.2f}")
print(f"放电后容量: {battery.calculate_capacity(battery.current, battery.time / 3600):.2f} Ah")

#计算性能参数

# 计算 C-Rate
c_rate = battery.calculate_c_rate(battery.current)     #C-rate表示充放电的快慢
print(f"C-Rate: {c_rate:.2f} C")
# 计算端电压
ocv = 4.2  # 开路电压 (V)
terminal_voltage = battery.calculate_terminal_voltage(ocv)      #计算实际输出电压
print(f"端电压: {terminal_voltage:.2f} V")

# 绘制 SOC 变化曲线     #电池电量随时间下降的过程
times = np.linspace(0, 3600, 100)  # 0 到 3600 秒
soc_values = []          #定义空列表

for t in times:        #开启循环
    battery.calculate_soc(battery.current, t)        #计算特定时刻的电量
    soc_values.append(battery.current_soc)       #记录数据

plt.figure(figsize=(10, 6))          #创建画布
plt.plot(times / 3600, soc_values)       #绘制线条
plt.title('锂离子电池 SOC 变化')      #加标题
plt.xlabel('时间 (小时)')      #加x轴
plt.ylabel('SOC')        #加y轴
plt.grid(True)           #打开背景网格线
plt.show()       #显示图像
