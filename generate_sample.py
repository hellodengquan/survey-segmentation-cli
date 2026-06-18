import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

n_normal = 80
n_speeder = 5
n_straight = 5
n_pattern = 3
n_contradict = 3
n_total = n_normal + n_speeder + n_straight + n_pattern + n_contradict

ages = random.choices(["18-25", "26-35", "36-45", "46-55", "56+"], weights=[25, 35, 20, 12, 8], k=n_total)
genders = random.choices(["男", "女", "其他"], weights=[48, 48, 4], k=n_total)
regions = random.choices(["北京", "上海", "广州", "深圳", "成都", "杭州"], k=n_total)
educations = random.choices(["高中及以下", "大专", "本科", "硕士及以上"], weights=[10, 20, 45, 25], k=n_total)

satisfaction_opts = ["非常不满意", "不满意", "一般", "满意", "非常满意"]
recommend_opts = ["肯定不会", "可能不会", "不确定", "可能会", "肯定会"]
frequency_opts = ["从不", "很少", "偶尔", "经常", "总是"]
yes_no_opts = ["是", "否"]

rows = []

for i in range(n_total):
    rid = f"R{i + 1:03d}"
    age, gender, region, edu = ages[i], genders[i], regions[i], educations[i]

    if i >= n_normal + n_speeder + n_straight + n_pattern:
        duration = random.randint(120, 400)
    elif i >= n_normal + n_speeder + n_straight:
        duration = random.randint(120, 400)
    elif i >= n_normal + n_speeder:
        duration = random.randint(200, 500)
    elif i >= n_normal:
        duration = random.randint(5, 45)
    else:
        duration = random.randint(120, 600)

    if i < n_normal:
        q1 = random.choices(satisfaction_opts, weights=[5, 10, 25, 40, 20])[0]
        q2 = random.choices(recommend_opts, weights=[5, 10, 20, 40, 25])[0]
        q3 = random.choices(frequency_opts, weights=[10, 15, 25, 35, 15])[0]
        q4 = random.choices(yes_no_opts, weights=[60, 40])[0]
        q5 = random.choices(satisfaction_opts, weights=[8, 12, 30, 35, 15])[0]
        q6 = random.choices(recommend_opts, weights=[10, 15, 25, 35, 15])[0]
        q7 = random.randint(1, 10)
        q8 = random.randint(1, 10)
        q9 = random.choices(frequency_opts, weights=[5, 20, 30, 30, 15])[0]

    elif i < n_normal + n_speeder:
        q1 = random.choice(satisfaction_opts)
        q2 = random.choice(recommend_opts)
        q3 = random.choice(frequency_opts)
        q4 = random.choice(yes_no_opts)
        q5 = random.choice(satisfaction_opts)
        q6 = random.choice(recommend_opts)
        q7 = random.randint(1, 10)
        q8 = random.randint(1, 10)
        q9 = random.choice(frequency_opts)

    elif i < n_normal + n_speeder + n_straight:
        same = random.choice(satisfaction_opts)
        q1 = q2 = q3 = q5 = q6 = q9 = same
        q4 = random.choice(yes_no_opts)
        q7 = q8 = 5

    elif i < n_normal + n_speeder + n_straight + n_pattern:
        q1 = satisfaction_opts[0]
        q2 = satisfaction_opts[1]
        q3 = satisfaction_opts[2]
        q4 = yes_no_opts[0]
        q5 = satisfaction_opts[4]
        q6 = recommend_opts[4]
        q7 = 1
        q8 = 2
        q9 = frequency_opts[4]

    else:
        q1 = "非常满意"
        q2 = "肯定不会"
        q3 = "从不"
        q4 = "否"
        q5 = "非常不满意"
        q6 = "肯定会"
        q7 = 1
        q8 = 10
        q9 = "总是"

    rows.append({
        "respondent_id": rid,
        "age": age,
        "gender": gender,
        "region": region,
        "education": edu,
        "duration_seconds": duration,
        "Q1_整体满意度": q1,
        "Q2_推荐意愿": q2,
        "Q3_使用频率": q3,
        "Q4_是否购买": q4,
        "Q5_界面满意度": q5,
        "Q6_功能满意度": q6,
        "Q7_体验评分": q7,
        "Q8_性价比评分": q8,
        "Q9_复购意愿": q9,
    })

df = pd.DataFrame(rows)

sample_rows = [
    {"respondent_id": "R097", "age": "", "gender": "女", "region": "上海", "education": "本科", "duration_seconds": 0, "Q1_整体满意度": "", "Q2_推荐意愿": "", "Q3_使用频率": "", "Q4_是否购买": "", "Q5_界面满意度": "", "Q6_功能满意度": "", "Q7_体验评分": -1, "Q8_性价比评分": -1, "Q9_复购意愿": ""},
    {"respondent_id": "R098", "age": "26-35", "gender": "", "region": "", "education": "", "duration_seconds": 300, "Q1_整体满意度": "n/a", "Q2_推荐意愿": "N/A", "Q3_使用频率": "一般", "Q4_是否购买": "yes", "Q5_界面满意度": "no", "Q6_功能满意度": "一般", "Q7_体验评分": 7, "Q8_性价比评分": 6, "Q9_复购意愿": "偶尔"},
    {"respondent_id": "R099", "age": "36-45", "gender": "男", "region": "北京", "education": "硕士", "duration_seconds": 200, "Q1_整体满意度": "非常满意", "Q2_推荐意愿": "肯定会", "Q3_使用频率": "经常", "Q4_是否购买": "是", "Q5_界面满意度": "非常满意", "Q6_功能满意度": "肯定会", "Q7_体验评分": 9, "Q8_性价比评分": 9, "Q9_复购意愿": "总是"},
    {"respondent_id": "R001", "age": "26-35", "gender": "女", "region": "上海", "education": "本科", "duration_seconds": 350, "Q1_整体满意度": "满意", "Q2_推荐意愿": "可能会", "Q3_使用频率": "经常", "Q4_是否购买": "是", "Q5_界面满意度": "满意", "Q6_功能满意度": "可能会", "Q7_体验评分": 8, "Q8_性价比评分": 7, "Q9_复购意愿": "经常"},
]

for row in sample_rows:
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "sample_survey.csv")
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"示例数据已生成: {output_path}")
print(f"总行数: {len(df)}")
