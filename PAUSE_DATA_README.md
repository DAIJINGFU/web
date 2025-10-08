# 停牌数据说明文档

## 📁 停牌数据目录

**位置**: `stockdata/stockdata/停牌数据/`

## 📋 数据格式要求

### 文件命名规则

- 每个股票一个 CSV 文件
- 文件名格式：`{股票代码}.csv`
- 示例：`000001.SZ.csv`, `600000.SH.csv`

### CSV 文件格式

#### 必需列（中文或英文均可）

| 中文列名     | 英文列名   | 数据类型   | 说明               | 示例         |
| ------------ | ---------- | ---------- | ------------------ | ------------ |
| 停牌开始日期 | start_date | 日期字符串 | YYYY-MM-DD 格式    | 2024-03-15   |
| 停牌结束日期 | end_date   | 日期字符串 | YYYY-MM-DD 格式    | 2024-03-20   |
| 停牌原因     | reason     | 文本       | 可选，说明停牌原因 | 重大资产重组 |

#### 支持的列名变体

系统会自动识别以下列名：

- 包含"开始"或"start"的列 → 停牌开始日期
- 包含"结束"或"end"的列 → 停牌结束日期
- 包含"原因"或"reason"的列 → 停牌原因

### 示例文件内容

**000001.SZ.csv**:

```csv
股票代码,停牌开始日期,停牌结束日期,停牌原因
000001.SZ,2024-03-15,2024-03-20,重大资产重组
000001.SZ,2024-06-10,2024-06-10,临时停牌
000001.SZ,2024-09-05,2024-09-15,股东大会停牌
```

**或使用英文列名**:

```csv
stock_code,start_date,end_date,reason
600000.SH,2024-01-10,2024-01-15,Major Asset Restructuring
600000.SH,2024-05-20,2024-05-20,Temporary Suspension
```

## 🔧 停牌检测逻辑

### 工作原理

1. **自动查找数据文件**

   - 系统会在以下路径查找停牌数据：
     - `停牌数据/{股票代码}.csv`
     - `pause_data/{股票代码}.csv`
     - `停牌数据/{股票代码前缀}.csv`（不带交易所后缀）

2. **停牌判断**

   - 如果交易日期在 `[停牌开始日期, 停牌结束日期]` 区间内，则认为停牌
   - 停牌期间买入/卖出订单会被阻止
   - 会在日志中记录：`[pause_check] BLOCK BUY/SELL {股票代码} paused on {日期}`

3. **容错处理**
   - 如果没有停牌数据文件，假设股票未停牌（不影响正常交易）
   - 如果数据文件格式错误，假设未停牌（避免误阻）

### 在策略中的表现

```python
def handle_data(context, data):
    stock = '000001.SZ'

    # 如果股票在停牌期间
    # order(stock, 100) 会被阻止
    # 订单状态会被标记为 'BlockedPaused'
    # 在 blocked_orders 中可以看到记录
```

## 📊 数据来源建议

### 如何获取停牌数据

1. **聚宽平台**

   ```python
   # 查询历史停牌数据
   from jqdatasdk import get_price
   paused_info = get_price('000001.XSHE',
                          start_date='2024-01-01',
                          end_date='2024-12-31',
                          fields=['paused'])
   ```

2. **Tushare**

   ```python
   import tushare as ts
   pro = ts.pro_api('your_token')
   df = pro.suspend_d(ts_code='000001.SZ',
                      start_date='20240101',
                      end_date='20241231')
   ```

3. **手动维护**
   - 从交易所公告获取
   - 深交所：http://www.szse.cn/
   - 上交所：http://www.sse.com.cn/

### 数据转换示例

```python
import pandas as pd

# 从聚宽获取的停牌数据转换
def convert_jq_pause_data(stock_code, start, end):
    # ... 获取聚宽停牌数据
    pause_df = ...  # DataFrame with 'date' and 'paused' columns

    # 找出停牌区间
    pause_periods = []
    in_pause = False
    start_date = None

    for idx, row in pause_df.iterrows():
        if row['paused'] and not in_pause:
            start_date = row['date']
            in_pause = True
        elif not row['paused'] and in_pause:
            pause_periods.append({
                '股票代码': stock_code,
                '停牌开始日期': start_date,
                '停牌结束日期': row['date'],
                '停牌原因': '停牌'
            })
            in_pause = False

    # 保存为CSV
    result_df = pd.DataFrame(pause_periods)
    result_df.to_csv(f'停牌数据/{stock_code}.csv', index=False)
```

## ⚠️ 注意事项

1. **日期格式统一**

   - 必须使用 YYYY-MM-DD 格式
   - 不支持其他日期格式（如 YYYYMMDD）

2. **停牌区间包含性**

   - 停牌开始日期和结束日期都包含在停牌期内
   - 例如：开始=2024-03-15，结束=2024-03-20
   - 则 2024-03-15 到 2024-03-20（含）都不能交易

3. **股票代码格式**

   - 支持 .SZ/.SH 和 .XSHE/.XSHG 两种格式
   - 系统会自动转换匹配

4. **性能考虑**
   - 每次下单都会读取停牌数据文件
   - 建议不要放置过大的停牌历史数据
   - 只保留回测期间相关的停牌记录

## 🧪 测试停牌功能

创建测试策略验证停牌检测：

```python
def initialize(context):
    g.security = '000001.SZ'
    set_option('log_level', 'debug')

def handle_data(context, data):
    # 尝试在停牌日买入（应该被阻止）
    if context.current_dt.date().isoformat() == '2024-03-15':
        order(g.security, 100)  # 应该被阻止
        print("尝试买入（停牌期间）")

    # 在非停牌日买入（应该成功）
    if context.current_dt.date().isoformat() == '2024-03-21':
        order(g.security, 100)  # 应该成功
        print("尝试买入（停牌结束后）")
```

查看日志应该看到：

```
[pause_check] BLOCK BUY 000001.SZ paused on 2024-03-15
```

## 📈 与 T+1 制度的结合

停牌检测和 T+1 制度是独立的两个检查：

1. **停牌检查**：阻止停牌期间的所有买卖
2. **T+1 检查**：阻止当日买入的股票当日卖出

两个检查都通过才能下单成功。

---

**文档版本**: v1.0  
**创建日期**: 2025-10-08  
**适用系统**: backtrader-new1 本地回测系统
