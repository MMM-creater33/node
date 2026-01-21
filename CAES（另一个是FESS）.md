import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random  # 必须导入 random 模块，解决报错

class CAESystem:
    """压缩空气储能 (CAES) 系统模型"""

    def __init__(self, M_air_max=1000, M_air_min=100, P_comp_max=100, P_gen_max=80):
        """
        初始化参数 (对应论文物理意义)
        """
        # 物理限制
        self.M_air_max = M_air_max      # 储气室最大空气质量 (kg)
        self.M_air_min = M_air_min      # 储气室最小空气质量 (kg)
        self.P_comp_max = P_comp_max    # 压缩机最大输入功率 (kW)
        self.P_gen_max = P_gen_max      # 燃气轮机最大输出功率 (kW)

        # 效率参数 (对应论文公式系数)
        self.eta_comp = 0.85            # 压缩机效率
        self.eta_gen = 0.35             # 发电系统效率 (包含膨胀机和发电机)
        self.H_f = 45000                # 天然气低热值 (kJ/kg)

        # 状态变量初始化
        self.M_air_current = (M_air_max + M_air_min) / 2  # 初始空气质量
        self.history = []  # 用于记录数据

    def step(self, price):
        """
        单步仿真逻辑 (模拟论文中的运行策略)
        """
        P_demand = 0
        m_in = 0
        m_out = 0
        m_fuel = 0

        # 简单策略：低价充电，高价放电
        if price < 0.4: # 低价时段 -> 充电 (压缩空气)
            # 充电功率满负荷
            P_demand = -self.P_comp_max
            # 根据论文公式 P = f(m), 反推质量流速 m = P / (eff * Hf) 简化处理
            # 实际上论文中压缩功率与空气质量流速成正比
            m_in = self.P_comp_max * 0.01  # 简化比例系数

        elif price > 0.6: # 高价时段 -> 放电 (发电)
            # 放电功率满负荷
            P_demand = self.P_gen_max
            # 发电时消耗空气质量
            m_out = self.P_gen_max * 0.015 # 简化比例系数
            # 天然气消耗 (对应论文公式 28)
            m_fuel = P_demand * 0.05 # 简化燃料消耗系数

        # --- 物理约束检查 (防止溢出/抽空) ---

        # 1. 计算理论上的质量变化
        M_air_next = self.M_air_current + m_in - m_out

        # 2. 处理越界情况 (Clipping)
        if M_air_next > self.M_air_max:
            # 超过上限，只能充到满
            m_in = self.M_air_max - self.M_air_current
            if m_in < 0: m_in = 0
            M_air_next = self.M_air_max
            P_demand = 0 # 满了就不能充了(或者限制充入量)

        elif M_air_next < self.M_air_min:
            # 低于下限，只能放到空
            m_out = self.M_air_current - self.M_air_min
            if m_out < 0: m_out = 0
            M_air_next = self.M_air_min
            P_demand = 0 # 空了就不能放了(或者限制放出量)

        # --- 更新状态 ---
        self.M_air_current = M_air_next

        # --- 计算衍生指标 ---
        soc = (self.M_air_current - self.M_air_min) / (self.M_air_max - self.M_air_min)

        # 记录数据
        self.history.append({
            "Time": len(self.history),
            "Price": price,
            "Power_kW": P_demand,
            "SOC": soc,
            "Mass_air_kg": self.M_air_current,
            "Fuel_kg": m_fuel,
            "m_in": m_in,
            "m_out": m_out
        })

    def run_simulation(self, steps=24, price_volatility=0.2):
        """运行多步仿真"""
        for i in range(steps):
            # 模拟随机电价 (均值 0.5, 模拟峰谷)
            price = 0.5 + random.uniform(-price_volatility, price_volatility)
            self.step(price)

    def get_report_df(self):
        """生成 Pandas 报告表格"""
        return pd.DataFrame(self.history)

    def plot_results(self):
        """绘制结果图表"""
        df = self.get_report_df()

        plt.figure(figsize=(12, 8))

        # 1. 功率图 (Power)
        plt.subplot(3, 1, 1)
        plt.plot(df['Time'], df['Power_kW'], marker='o', color='b', label='Power (kW)')
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.title('CAES System Power Operation')
        plt.ylabel('Power (kW)')
        plt.legend()

        # 2. SOC 图 (State of Charge)
        plt.subplot(3, 1, 2)
        plt.plot(df['Time'], df['SOC'], marker='x', color='r', label='SOC')
        plt.fill_between(df['Time'], 0, 1, where=(df['SOC'] <= 0.2) | (df['SOC'] >= 0.8),
                         facecolor='yellow', alpha=0.3, label='Warning Zone')
        plt.title('CAES System SOC Variation')
        plt.ylabel('SOC (0-1)')
        plt.ylim(-0.1, 1.1)
        plt.legend()

        # 3. 能量/空气质量变化图
        plt.subplot(3, 1, 3)
        plt.plot(df['Time'], df['Mass_air_kg'], marker='^', color='g', label='Air Mass (kg)')
        plt.title('CAES Air Mass (Energy) Variation')
        plt.ylabel('Mass (kg)')
        plt.xlabel('Time (Hour)')
        plt.legend()

        plt.tight_layout()
        plt.show()

# --- 主程序执行 ---
if __name__ == "__main__":
    # 1. 初始化模型
    caes_model = CAESystem(
        M_air_max=1000,
        M_air_min=100,
        P_comp_max=100,
        P_gen_max=80
    )

    # 2. 运行仿真 (24小时)
    caes_model.run_simulation(steps=24)

    # 3. 输出直接数据 (表格)
    print("=== 📋 压缩空气储能 (CAES) 仿真数据报告 ===")
    print(caes_model.get_report_df())

    # 4. 输出图表
    caes_model.plot_results()
