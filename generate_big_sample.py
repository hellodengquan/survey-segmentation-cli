import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

n = 5000

ids = [f"R{i+1:05d}" for i in range(n)]
ages = random.choices(["18-25", "26-35", "36-45", "46-55", "56+"], weights=[25, 35, 20, 12, 8], k=n)
genders = random.choices(["男", "女", "其他"], weights=[48, 48, 4], k=n)
regions = random.choices(["北京", "上海", "广州", "深圳", "成都", "杭州"], k=n)

satisfaction = ["非常不满意", "不满意", "一般", "满意", "非常满意"]
recommend = ["肯定不会", "可能不会", "不确定", "可能会", "肯定会"]
frequency = ["从不", "很少", "偶尔", "经常", "总是"]

rows = []
for i in range(n):
    dur = random.randint(60, 600)
    if i < 50:
        dur = random.randint(5, 45)
    elif i < 100:
        pass

    rows.append({
        "受访者 ID": ids[i],
        "年龄 ": ages[i],
        "Gender": genders[i],
        " 地区 ": regions[i],
        "答题时长(秒)": dur,
        "Q1_整体满意度": random.choices(satisfaction, weights=[5, 10, 25, 40, 20])[0],
        "Q2_推荐意愿": random.choices(recommend, weights=[5, 10, 20, 40, 25])[0],
        "Q3_使用频率": random.choices(frequency, weights=[10, 15, 25, 35, 15])[0],
        "Q4_是否购买": random.choices(["是", "否"], weights=[60, 40])[0],
        "Q5_界面满意度": random.choices(satisfaction, weights=[8, 12, 30, 35, 15])[0],
        "Q6_功能满意度": random.choices(recommend, weights=[10, 15, 25, 35, 15])[0],
        "Q7_体验评分": random.randint(1, 10),
        "Q8_性价比评分": random.randint(1, 10),
        "Q9_复购意愿": random.choices(frequency, weights=[5, 20, 30, 30, 15])[0],
    })

for i in range(100, 105):
    rows[i].update({
        "Q1_整体满意度": "一般",
        "Q2_推荐意愿": "一般",
        "Q3_使用频率": "一般",
        "Q5_界面满意度": "一般",
        "Q6_功能满意度": "不确定",
        "Q9_复购意愿": "偶尔",
        "Q7_体验评分": 5,
        "Q8_性价比评分": 5,
    })

df = pd.DataFrame(rows)
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "big_survey_mixed_columns.csv")
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"大文件已生成: {output_path}")
print(f"总行数: {len(df)}")
print(f"列名: {list(df.columns)}")
