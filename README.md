# Forex Hurst Cycles Analysis System
## 外汇蔡森周期分析系统

基于J.M. Hurst市场周期理论的多货币对、多时间框架外汇分析系统。

## 📊 项目简介

本项目实现了完整的外汇市场周期分析功能，包括：

- **蔡森周期分析** (Hurst Cycles Analysis)
- **多时间框架分析** (1/4/15/30/60/240分钟)
- **多货币对支持** (EUR/GBP/JPY/AUD/CAD/CNH)
- **自动化交易计划生成**
- **可视化图表生成**

## 🎯 核心功能

### 1. 蔡森周期分析 (`hurst_cycles_analysis.py`)

**核心类：**
- `HurstCyclesAnalyzer` - 单个货币对的周期分析
- `MultiTimeFrameAnalyzer` - 多时间框架分析
- `ForexPairsHurstAnalyzer` - 多货币对批量分析

**主要功能：**
- 频谱分析识别主导周期
- 去趋势化处理
- 周期成分提取
- 周期转折点识别
- 未来价格投影

### 2. 自动化分析 (`run_hurst_analysis.py`)

一键运行完整分析流程：
```bash
python3 run_hurst_analysis.py
```

**输出内容：**
- 终端详细报告
- 交易计划文件
- 周期分析图表

### 3. 可视化图表 (`generate_hurst_charts.py`)

生成专业的分析图表：
- 综合分析仪表板（8个面板）
- 多货币对对比图
- 周期相位分析
- 未来价格投影

## 📈 分析结果示例

### 最新分析 (2026-02-15)

| 货币对 | 当前价格 | 周期趋势 | 信心度 | 建议 |
|--------|----------|----------|--------|------|
| EUR/USD | 1.0913 | 📉 看跌 | 83.3% | 卖空 |
| GBP/USD | 1.2637 | 📈 看涨 | 83.3% | 买入 |
| USD/JPY | 149.57 | 📉 看跌 | 100% | ⭐ 卖空 |
| AUD/USD | 0.6529 | 📈 看涨 | 100% | ⭐ 买入 |
| USD/CAD | 1.3396 | 📉 看跌 | 100% | ⭐ 卖空 |
| USD/CNH | 7.2049 | 📈 看涨 | 83.3% | 买入 |

详细交易计划请查看：`next_week_trading_plan_summary.md`

## 🚀 快速开始

### 环境要求

```bash
pip install pandas numpy matplotlib scipy yfinance
```

### 运行分析

```bash
# 完整分析（包含图表生成）
python3 run_hurst_analysis.py

# 仅生成图表
python3 generate_hurst_charts.py
```

## 📁 项目结构

```
forex/
├── hurst_cycles_analysis.py          # 蔡森周期分析核心模块
├── run_hurst_analysis.py            # 主分析程序
├── generate_hurst_charts.py         # 图表生成程序
├── generate_forex_data.py           # 数据生成工具
├── forex_technical_analysis.py      # 技术指标库
├── next_week_trading_plan_summary.md # 交易计划摘要
└── README.md                         # 本文件
```

## 🎓 理论基础

### J.M. Hurst 周期理论核心原理

1. **周期叠加原理**: 价格由多个不同周期的波浪叠加而成
2. **主导周期**: 识别影响价格最大的2-3个周期
3. **周期相位**: 判断当前处于周期的哪个阶段
4. **多时间框架共振**: 多个时间框架同向信号最可靠

### 周期分析方法

- **频谱分析**: 使用FFT和Welch方法识别周期
- **带通滤波**: 提取特定频率的周期成分
- **去趋势化**: 移除趋势，提取周期性成分
- **相位分析**: 判断当前在周期中的位置

## 📊 生成的文件

运行分析后会生成以下文件：

- `hurst_trading_plan_YYYYMMDD_HHMMSS.txt` - 详细交易计划
- `hurst_cycles_dashboard_YYYYMMDD_HHMMSS.png` - 综合仪表板
- `hurst_cycles_comparison_YYYYMMDD_HHMMSS.png` - 货币对对比图

## ⚠️ 免责声明

本项目仅供学习和研究使用。外汇交易涉及高风险，可能导致本金损失。

- 周期分析基于历史数据，实际市场可能偏离预测
- 建议结合基本面分析和其他技术指标
- 严格执行止损，控制风险
- 本项目不对任何交易损失负责

## 📝 许可证

MIT License - 自由使用和修改

## 🔗 相关资源

- J.M. Hurst 原著：*"The Profit Magic of Stock Transaction Timing"*
- 周期分析理论：https://www.hurstcycles.com/

---

**祝交易顺利！ 📈**
