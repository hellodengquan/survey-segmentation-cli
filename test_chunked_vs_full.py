import sys
import os
import json
import pandas as pd
import numpy as np
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from survey_analyzer.column_normalizer import AliasRegistry, normalize_column_name, normalize_columns, apply_column_normalization
from survey_analyzer.cleaner import clean_survey_data, clean_chunk, identify_column_types
from survey_analyzer.segmenter import segment_by_columns, auto_segment, get_segment_summary, IncrementalSegmentCounter
from survey_analyzer.stats import (
    compute_question_distribution, compute_cross_tabulation, compute_numeric_summary,
    compute_chunk_counts, merge_chunk_counts, IncrementalNumericStats,
)
from survey_analyzer.anomaly import (
    detect_anomalies, detect_chunk_anomalies, detect_global_anomalies, merge_anomaly_results,
    get_anomaly_summary,
)
from survey_analyzer.config import AppConfig, AnomalyThresholds, CleaningConfig, SegmentConfig


def compare_distributions(dist_full: dict[str, pd.DataFrame], dist_chunk: dict[str, pd.DataFrame]) -> list[str]:
    mismatches = []
    common_cols = set(dist_full.keys()) & set(dist_chunk.keys())

    for col in sorted(common_cols):
        df_f = dist_full[col].sort_values("选项").reset_index(drop=True)
        df_c = dist_chunk[col].sort_values("选项").reset_index(drop=True)

        if df_f.shape != df_c.shape:
            mismatches.append(f"[分布] 题目 '{col}' 行数不一致: 全量={df_f.shape[0]}, 分块={df_c.shape[0]}")
            continue

        for _, row_f in df_f.iterrows():
            option = row_f["选项"]
            row_c = df_c[df_c["选项"] == option]
            if row_c.empty:
                mismatches.append(f"[分布] 题目 '{col}' 分块缺少选项: {option}")
                continue

            count_f = int(row_f["人数"])
            count_c = int(row_c["人数"].values[0])
            if count_f != count_c:
                mismatches.append(f"[分布] 题目 '{col}' 选项 '{option}' 人数不一致: 全量={count_f}, 分块={count_c}")

            pct_f = float(str(row_f["占比（显示）"]).replace("%", ""))
            pct_c = float(str(row_c["占比（显示）"].values[0]).replace("%", ""))
            if abs(pct_f - pct_c) > 0.2:
                mismatches.append(f"[分布] 题目 '{col}' 选项 '{option}' 占比差异: 全量={pct_f}%, 分块={pct_c}%")

    only_full = set(dist_full.keys()) - set(dist_chunk.keys())
    only_chunk = set(dist_chunk.keys()) - set(dist_full.keys())
    if only_full:
        mismatches.append(f"[分布] 仅全量模式有: {only_full}")
    if only_chunk:
        mismatches.append(f"[分布] 仅分块模式有: {only_chunk}")

    return mismatches


def compare_anomalies(anomaly_full: pd.DataFrame, anomaly_chunk: pd.DataFrame) -> list[str]:
    mismatches = []

    if anomaly_full.empty and anomaly_chunk.empty:
        return mismatches

    if anomaly_full.empty != anomaly_chunk.empty:
        mismatches.append(f"[异常] 一方为空: 全量空={anomaly_full.empty}, 分块空={anomaly_chunk.empty}")
        return mismatches

    full_ids = set(zip(anomaly_full["受访者ID"], anomaly_full["异常类型"]))
    chunk_ids = set(zip(anomaly_chunk["受访者ID"], anomaly_chunk["异常类型"]))

    only_full = full_ids - chunk_ids
    only_chunk = chunk_ids - full_ids

    if only_full:
        mismatches.append(f"[异常] 仅全量检测到 {len(only_full)} 条: {sorted(only_full)[:5]}")
    if only_chunk:
        mismatches.append(f"[异常] 仅分块检测到 {len(only_chunk)} 条: {sorted(only_chunk)[:5]}")

    return mismatches


def run_comparison(data_path: str, chunk_size: int = 500):
    AliasRegistry.reset()
    config = AppConfig()

    print("=" * 70)
    print("分块 vs 全量 对照测试")
    print("=" * 70)
    print(f"数据文件: {data_path}")
    print(f"分块大小: {chunk_size} 行")
    print()

    all_mismatches = []

    # === 全量模式 ===
    print("▶ 运行全量模式...")
    AliasRegistry.reset()
    df_full = pd.read_csv(data_path, encoding="utf-8")
    norm_result = normalize_columns(list(df_full.columns))
    df_full = apply_column_normalization(df_full, norm_result)
    df_full = clean_survey_data(df_full, id_col=config.id_col, time_col=config.time_col, config=config.cleaning)
    col_types_full = identify_column_types(df_full, id_col=config.id_col)
    segment_cols_used = col_types_full.get("demographic", [])[:2]
    df_full = segment_by_columns(df_full, segment_cols_used)
    question_cols_full = col_types_full.get("question", [])
    dist_full = compute_question_distribution(df_full, question_cols_full)
    num_cols_full = [c for c in question_cols_full if pd.api.types.is_numeric_dtype(df_full[c])]
    num_summary_full = compute_numeric_summary(df_full, num_cols_full)
    segment_full = get_segment_summary(df_full)
    anomaly_full = detect_anomalies(df_full, question_cols_full, id_col=config.id_col, time_col=config.time_col, thresholds=config.anomaly)
    print(f"  全量: {len(df_full)} 行, {len(question_cols_full)} 题目, {len(anomaly_full)} 异常")

    # 确定 auto_segment 选择的分群列
    segment_cols_used = col_types_full.get("demographic", [])[:2]

    # === 分块模式 ===
    print("▶ 运行分块模式...")
    AliasRegistry.reset()
    cleaned_chunks = []
    chunk_anomaly_records = []
    total_rows_chunk = 0
    chunk_idx = 0

    for chunk in pd.read_csv(data_path, encoding="utf-8", chunksize=chunk_size):
        chunk_idx += 1
        norm_result_c = normalize_columns(list(chunk.columns))
        chunk = apply_column_normalization(chunk, norm_result_c)
        chunk = clean_chunk(chunk, config.cleaning, id_col=config.id_col)
        cleaned_chunks.append(chunk)

    full_df_chunk = pd.concat(cleaned_chunks, ignore_index=True)
    if config.id_col in full_df_chunk.columns:
        full_df_chunk = full_df_chunk.drop_duplicates(subset=[config.id_col])
    total_rows_chunk = len(full_df_chunk)

    col_types_chunk = identify_column_types(full_df_chunk, id_col=config.id_col)
    question_cols_chunk = col_types_chunk.get("question", [])

    full_df_chunk = segment_by_columns(full_df_chunk, segment_cols_used)
    segment_chunk_summary = get_segment_summary(full_df_chunk)

    dist_chunk = compute_question_distribution(full_df_chunk, question_cols_chunk)
    num_cols_chunk = [c for c in question_cols_chunk if pd.api.types.is_numeric_dtype(full_df_chunk[c])]
    num_summary_chunk = compute_numeric_summary(full_df_chunk, num_cols_chunk)

    anomaly_chunk = detect_anomalies(full_df_chunk, question_cols_chunk, id_col=config.id_col, time_col=config.time_col, thresholds=config.anomaly)
    print(f"  分块: {total_rows_chunk} 行 ({chunk_idx} 块), {len(question_cols_chunk)} 题目, {len(anomaly_chunk)} 异常")

    # === 增量分块统计（独立验证） ===
    print("▶ 运行增量统计模式...")
    AliasRegistry.reset()
    inc_cleaned_chunks = []

    for chunk in pd.read_csv(data_path, encoding="utf-8", chunksize=chunk_size):
        norm_result_c = normalize_columns(list(chunk.columns))
        chunk = apply_column_normalization(chunk, norm_result_c)
        chunk = clean_chunk(chunk, config.cleaning, id_col=config.id_col)
        inc_cleaned_chunks.append(chunk)

    inc_full = pd.concat(inc_cleaned_chunks, ignore_index=True)
    if config.id_col in inc_full.columns:
        inc_full = inc_full.drop_duplicates(subset=[config.id_col])

    inc_full = segment_by_columns(inc_full, segment_cols_used)

    inc_chunk_counts_list = []
    inc_numeric_stats = IncrementalNumericStats()
    inc_segment_counter = IncrementalSegmentCounter()
    inc_anomaly_records = []

    inc_chunk_size = max(chunk_size, len(inc_full) // 4) or chunk_size
    for i in range(0, len(inc_full), inc_chunk_size):
        chunk = inc_full.iloc[i:i + inc_chunk_size].copy()
        inc_segment_counter.update(chunk)

        col_types_c = identify_column_types(chunk, id_col=config.id_col)
        q_cols_c = col_types_c.get("question", [])

        inc_chunk_counts_list.append(compute_chunk_counts(chunk, q_cols_c))
        inc_numeric_stats.update(chunk, [c for c in q_cols_c if pd.api.types.is_numeric_dtype(chunk[c])])

        records = detect_chunk_anomalies(
            chunk, q_cols_c, id_col=config.id_col,
            time_col=config.time_col, thresholds=config.anomaly,
        )
        inc_anomaly_records.append(records)

    dist_inc = merge_chunk_counts(inc_chunk_counts_list, total_rows_chunk)
    global_anomaly = detect_global_anomalies(full_df_chunk, question_cols_chunk, id_col=config.id_col, thresholds=config.anomaly)
    anomaly_inc = merge_anomaly_results(inc_anomaly_records, global_anomaly)
    segment_inc_summary = inc_segment_counter.to_dataframe()
    num_summary_inc = inc_numeric_stats.to_dataframe()
    print(f"  增量: 分布 {len(dist_inc)} 题, 异常 {len(anomaly_inc)} 条")

    # === 对比 ===
    print()
    print("▶ ─────────────────────────────────────")
    print("▶ 对比 1: 全量模式 vs 拼合后分块模式")
    print("▶ ─────────────────────────────────────")
    print()

    print("  [1/6] 数据行数")
    if len(df_full) != total_rows_chunk:
        all_mismatches.append(f"数据行数不一致: 全量={len(df_full)}, 分块={total_rows_chunk}")
        print(f"    ❌ 全量={len(df_full)}, 分块={total_rows_chunk}")
    else:
        print(f"    ✅ 行数一致: {len(df_full)}")

    print("  [2/6] 题目列表")
    if set(question_cols_full) != set(question_cols_chunk):
        all_mismatches.append(f"题目列表不一致: 全量={question_cols_full}, 分块={question_cols_chunk}")
        print(f"    ❌ 全量={sorted(question_cols_full)}")
        print(f"    ❌ 分块={sorted(question_cols_chunk)}")
    else:
        print(f"    ✅ {len(question_cols_full)} 题目列表一致")

    print("  [3/6] 题目分布统计")
    dist_mismatches = compare_distributions(dist_full, dist_chunk)
    all_mismatches.extend(dist_mismatches)
    if dist_mismatches:
        for m in dist_mismatches:
            print(f"    ❌ {m}")
    else:
        checked = len(set(dist_full.keys()) & set(dist_chunk.keys()))
        print(f"    ✅ {checked} 题目分布完全一致")

    print("  [4/6] 分群统计")
    if segment_full.shape != segment_chunk_summary.shape:
        all_mismatches.append(f"分群行数不一致: 全量={segment_full.shape[0]}, 分块={segment_chunk_summary.shape[0]}")
        print(f"    ❌ 全量={segment_full.shape[0]} 组, 分块={segment_chunk_summary.shape[0]} 组")
    else:
        seg_ok = True
        for _, row_f in segment_full.iterrows():
            group = row_f["_群组"]
            row_c = segment_chunk_summary[segment_chunk_summary["_群组"] == group]
            if row_c.empty:
                seg_ok = False
                all_mismatches.append(f"分块缺少群组: {group}")
                continue
            if str(row_f["占比"]) != str(row_c["占比"].values[0]):
                seg_ok = False
                all_mismatches.append(f"群组 '{group}' 占比不一致: 全量={row_f['占比']}, 分块={row_c['占比'].values[0]}")
        if seg_ok:
            print(f"    ✅ {segment_full.shape[0]} 个群组完全一致")

    print("  [5/6] 异常检测")
    anomaly_mismatches = compare_anomalies(anomaly_full, anomaly_chunk)
    all_mismatches.extend(anomaly_mismatches)
    if anomaly_mismatches:
        for m in anomaly_mismatches:
            print(f"    ❌ {m}")
    else:
        print(f"    ✅ 异常检测结果完全一致 ({len(anomaly_full)} 条)")

    print("  [6/6] 数值统计")
    if num_summary_full.empty:
        print("    ⚠️  无数值列，跳过")
    elif num_summary_chunk.empty:
        all_mismatches.append("分块数值统计为空")
        print("    ❌ 分块数值统计为空")
    else:
        num_ok = True
        for col_name in num_summary_full["题目"].values:
            row_f = num_summary_full[num_summary_full["题目"] == col_name]
            row_c = num_summary_chunk[num_summary_chunk["题目"] == col_name]
            if row_c.empty or row_f.empty:
                continue
            rf = row_f.iloc[0]
            rc = row_c.iloc[0]
            for stat_col in ["有效数", "均值", "标准差", "最小值", "最大值"]:
                vf = rf[stat_col]
                vc = rc[stat_col]
                if isinstance(vf, (int, float)) and isinstance(vc, (int, float)):
                    if abs(vf - vc) > 0.1:
                        num_ok = False
                        all_mismatches.append(f"列 '{col_name}' {stat_col}: 全量={vf}, 分块={vc}")
        if num_ok:
            print(f"    ✅ {num_summary_full.shape[0]} 列数值统计一致")

    # === 对比2: 增量统计 vs 全量 ===
    print()
    print("▶ ─────────────────────────────────────")
    print("▶ 对比 2: 全量模式 vs 增量合并模式")
    print("▶ ─────────────────────────────────────")
    print()

    print("  [1/3] 增量分布 vs 全量分布")
    inc_dist_mm = compare_distributions(dist_full, dist_inc)
    all_mismatches.extend(inc_dist_mm)
    if inc_dist_mm:
        for m in inc_dist_mm:
            print(f"    ❌ {m}")
    else:
        checked = len(set(dist_full.keys()) & set(dist_inc.keys()))
        print(f"    ✅ {checked} 题目增量分布与全量一致")

    print("  [2/3] 增量异常 vs 全量异常")
    inc_anom_mm = compare_anomalies(anomaly_full, anomaly_inc)
    all_mismatches.extend(inc_anom_mm)
    if inc_anom_mm:
        for m in inc_anom_mm:
            print(f"    ❌ {m}")
    else:
        print(f"    ✅ 增量异常与全量一致 ({len(anomaly_full)} 条)")

    print("  [3/3] 增量数值统计 vs 全量数值统计")
    if num_summary_full.empty:
        print("    ⚠️  无数值列，跳过")
    elif num_summary_inc.empty:
        all_mismatches.append("增量数值统计为空")
        print("    ❌ 增量数值统计为空")
    else:
        inc_num_ok = True
        for col_name in num_summary_full["题目"].values:
            row_f = num_summary_full[num_summary_full["题目"] == col_name]
            row_c = num_summary_inc[num_summary_inc["题目"] == col_name]
            if row_c.empty or row_f.empty:
                continue
            rf = row_f.iloc[0]
            rc = row_c.iloc[0]
            for stat_col in ["有效数", "均值", "最小值", "最大值"]:
                vf = rf[stat_col]
                vc = rc[stat_col]
                if isinstance(vf, (int, float)) and isinstance(vc, (int, float)):
                    if abs(vf - vc) > 0.1:
                        inc_num_ok = False
                        all_mismatches.append(f"[增量] 列 '{col_name}' {stat_col}: 全量={vf}, 增量={vc}")
        if inc_num_ok:
            print(f"    ✅ 增量数值统计与全量一致")

    # === 汇总 ===
    print()
    print("=" * 70)
    if all_mismatches:
        print(f"❌ 对照测试未通过 — 共 {len(all_mismatches)} 项不一致:")
        for m in all_mismatches:
            print(f"   - {m}")
        return 1
    else:
        print("✅ 对照测试全部通过 — 全量/分块/增量三种模式结果完全一致")
        return 0


def test_alias_config():
    print("=" * 70)
    print("别名配置加载测试")
    print("=" * 70)

    AliasRegistry.reset()
    registry = AliasRegistry.get_instance()

    default_result = normalize_column_name("年龄")
    print(f"  内置默认: normalize_column_name('年龄') = '{default_result}'")
    assert default_result == "age", f"内置默认映射失败: {default_result}"
    print("  ✅ 内置默认映射正确")

    custom_config = {
        "chinese_english_aliases": {
            "受访者编号": "respondent_id",
            "受访者序号": "respondent_id",
            "年龄段": "age",
            "性别代码": "gender",
            "填写时长": "duration_seconds",
        },
        "canonical_names": {
            "respno": "respondent_id",
        }
    }

    custom_path = "/tmp/test_custom_aliases.json"
    with open(custom_path, "w", encoding="utf-8") as f:
        json.dump(custom_config, f, ensure_ascii=False, indent=2)

    AliasRegistry.reset()
    registry = AliasRegistry.get_instance()
    result = registry.reload_from_file(custom_path)

    assert result, "加载自定义别名文件失败"
    print(f"  ✅ 自定义别名文件加载成功")

    test_cases = {
        "受访者编号": "respondent_id",
        "受访者序号": "respondent_id",
        "性别代码": "gender",
        "填写时长": "duration_seconds",
        "年龄": "age",
        "地区": "region",
    }

    all_pass = True
    for input_name, expected in test_cases.items():
        actual = normalize_column_name(input_name)
        ok = actual == expected
        if not ok:
            all_pass = False
        print(f"  {'✅' if ok else '❌'} '{input_name}' → '{actual}' (期望: '{expected}')")

    assert normalize_column_name("地区") == "region", "reload 后内置别名丢失"
    print("  ✅ reload 后内置别名仍可用")

    AliasRegistry.reset()
    registry = AliasRegistry.get_instance()
    bad_result = registry.load_from_file("/nonexistent/file.json")
    assert not bad_result, "加载不存在的文件应该返回 False"
    print("  ✅ 不存在的文件正确返回 False")

    os.remove(custom_path)
    AliasRegistry.reset()

    if all_pass:
        print("✅ 别名配置加载测试全部通过")
    return 0 if all_pass else 1


SAMPLE_COLUMNS = [
    "respondent_id", "age", "gender", "region", "education", "duration_seconds",
    "Q1_整体满意度", "Q2_推荐意愿", "Q3_使用频率", "Q4_是否购买",
    "Q5_界面满意度", "Q6_功能满意度", "Q7_体验评分", "Q8_性价比评分", "Q9_复购意愿",
]

Q_OPTIONS = {
    "Q1_整体满意度": ["非常满意", "满意", "一般", "不满意", "非常不满意"],
    "Q2_推荐意愿": ["肯定会", "可能会", "不确定", "可能不会", "肯定不会"],
    "Q3_使用频率": ["总是", "经常", "偶尔", "很少", "从不"],
    "Q4_是否购买": ["是", "否"],
    "Q5_界面满意度": ["非常满意", "满意", "一般", "不满意", "非常不满意"],
    "Q6_功能满意度": ["肯定会", "可能会", "不确定", "可能不会", "肯定不会"],
    "Q7_体验评分": [str(x) for x in range(1, 11)],
    "Q8_性价比评分": [str(x) for x in range(1, 11)],
    "Q9_复购意愿": ["总是", "经常", "偶尔", "很少", "从不"],
}


def _generate_normal_row(rid: str, rng: np.random.Generator) -> dict:
    ages = ["18-25", "26-35", "36-45", "46-55", "55+"]
    genders = ["男", "女", "其他"]
    regions = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安"]
    educations = ["高中及以下", "大专", "本科", "硕士及以上"]

    row = {
        "respondent_id": rid,
        "age": rng.choice(ages),
        "gender": rng.choice(genders),
        "region": rng.choice(regions),
        "education": rng.choice(educations),
        "duration_seconds": int(rng.integers(100, 800)),
    }
    for q, opts in Q_OPTIONS.items():
        row[q] = rng.choice(opts)
    return row


def _generate_short_duration_row(rid: str, rng: np.random.Generator) -> dict:
    row = _generate_normal_row(rid, rng)
    row["duration_seconds"] = int(rng.integers(1, 9))
    return row


def _generate_missing_answers_row(rid: str, rng: np.random.Generator, missing_ratio: float = 0.6) -> dict:
    row = _generate_normal_row(rid, rng)
    q_cols = list(Q_OPTIONS.keys())
    n_missing = int(len(q_cols) * missing_ratio)
    miss_cols = rng.choice(q_cols, size=n_missing, replace=False)
    for col in miss_cols:
        row[col] = ""
    return row


def build_cross_chunk_duplicate_data(n_unique: int = 20) -> str:
    """
    构造跨 chunk 重复被试数据：每个被试有 2 行回答，所有第 1 次回答在前，所有第 2 次回答在后。
    当 chunk_size = n_unique 时，第 i 块和第 i+1 块包含同一被试的不同回答。
    """
    rng = np.random.default_rng(42)
    rows = []
    for i in range(1, n_unique + 1):
        rid = f"R{i:03d}"
        rows.append(_generate_normal_row(rid, rng))
    for i in range(1, n_unique + 1):
        rid = f"R{i:03d}"
        rows.append(_generate_normal_row(rid, rng))

    path = os.path.join(tempfile.gettempdir(), f"cross_chunk_dup_{n_unique}.csv")
    df = pd.DataFrame(rows, columns=SAMPLE_COLUMNS)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def _generate_straight_lining_row(rid: str, rng: np.random.Generator) -> dict:
    """生成直线答题（所有题选同一个选项）的行"""
    row = _generate_normal_row(rid, rng)
    single_option = rng.choice(["非常满意", "满意", "一般", "5"])
    for q in Q_OPTIONS:
        if "评分" in q:
            row[q] = "5"
        else:
            row[q] = single_option if single_option in Q_OPTIONS[q] else Q_OPTIONS[q][0]
    return row


def build_high_anomaly_data(n_rows: int = 100, anomaly_ratio: float = 0.8) -> str:
    """构造高异常占比数据（默认 80% 异常）"""
    rng = np.random.default_rng(7)
    rows = []
    n_anomaly = int(n_rows * anomaly_ratio)
    n_normal = n_rows - n_anomaly

    for i in range(n_anomaly):
        rid = f"R{i:05d}"
        anomaly_type = i % 3
        if anomaly_type == 0:
            rows.append(_generate_short_duration_row(rid, rng))
        elif anomaly_type == 1:
            rows.append(_generate_straight_lining_row(rid, rng))
        else:
            row = _generate_short_duration_row(rid, rng)
            row2 = _generate_straight_lining_row(rid, rng)
            row2["duration_seconds"] = row["duration_seconds"]
            rows.append(row2)

    for i in range(n_normal):
        rid = f"R{n_anomaly + i:05d}"
        rows.append(_generate_normal_row(rid, rng))

    rng.shuffle(rows)
    for idx, row in enumerate(rows):
        row["respondent_id"] = f"R{idx:05d}"

    path = os.path.join(tempfile.gettempdir(), f"high_anomaly_{n_rows}.csv")
    df = pd.DataFrame(rows, columns=SAMPLE_COLUMNS)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def build_small_dataset(n_rows: int = 10) -> str:
    """构造小型数据集（用于 chunk=1 等极端测试）"""
    rng = np.random.default_rng(99)
    rows = []
    for i in range(n_rows):
        rid = f"R{i:03d}"
        rows.append(_generate_normal_row(rid, rng))

    path = os.path.join(tempfile.gettempdir(), f"small_{n_rows}.csv")
    df = pd.DataFrame(rows, columns=SAMPLE_COLUMNS)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def build_heavy_duplicate_data(n_unique: int = 10, dup_per_id: int = 5) -> str:
    """
    构造大量重复数据：每个被试有 dup_per_id 行回答，打乱顺序。
    用于测试跨 chunk 去重的极端场景。
    """
    rng = np.random.default_rng(123)
    rows = []
    for i in range(n_unique):
        rid = f"R{i:03d}"
        for _ in range(dup_per_id):
            rows.append(_generate_normal_row(rid, rng))

    rng.shuffle(rows)
    path = os.path.join(tempfile.gettempdir(), f"heavy_dup_{n_unique}x{dup_per_id}.csv")
    df = pd.DataFrame(rows, columns=SAMPLE_COLUMNS)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def build_single_row_dataset() -> str:
    """构造只有 1 行的极端小数据集"""
    rng = np.random.default_rng(42)
    row = _generate_normal_row("R001", rng)
    path = os.path.join(tempfile.gettempdir(), "single_row.csv")
    df = pd.DataFrame([row], columns=SAMPLE_COLUMNS)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def test_extreme_chunk_sizes():
    """极端 chunk size 测试：1、等于行数、大于行数、单行数据"""
    print()
    print("=" * 70)
    print("边界测试：极端 chunk size")
    print("=" * 70)

    total_failures = 0
    expected_notes = []

    data_path = build_small_dataset(n_rows=10)
    test_cases = [
        ("chunk_size = 1（每行一块）", data_path, 1),
        ("chunk_size = 行数（一块正好）", data_path, 10),
        ("chunk_size > 行数（单块）", data_path, 100),
    ]

    for label, dp, cs in test_cases:
        print(f"\n--- {label} ---")
        rc = run_comparison(dp, chunk_size=cs)
        if rc != 0:
            total_failures += 1

    os.remove(data_path)

    single_path = build_single_row_dataset()
    print(f"\n--- 单行数据 + chunk_size=1 ---")
    rc = run_comparison(single_path, chunk_size=1)
    if rc != 0:
        total_failures += 1
    os.remove(single_path)

    print()
    print("=" * 70)
    if total_failures == 0:
        print("✅ 极端 chunk size 测试全部通过")
    else:
        print(f"❌ 极端 chunk size 测试：{total_failures} 组失败")

    if expected_notes:
        print("📝 预期差异说明:")
        for note in expected_notes:
            print(f"   - {note}")

    return total_failures


def test_cross_chunk_duplicates():
    """跨 chunk 边界重复被试测试"""
    print()
    print("=" * 70)
    print("边界测试：跨 chunk 边界重复被试")
    print("=" * 70)

    total_failures = 0
    expected_notes = []

    n_unique = 20
    data_path = build_cross_chunk_duplicate_data(n_unique=n_unique)

    print(f"场景1: {n_unique} 个唯一被试 × 2 行 = {n_unique * 2} 行")
    print(f"  chunk_size = {n_unique}（第一块全是首次回答，第二块全是第二次回答）")
    print()
    rc = run_comparison(data_path, chunk_size=n_unique)
    if rc != 0:
        total_failures += 1
    os.remove(data_path)

    n_unique2 = 10
    dup = 5
    data_path2 = build_heavy_duplicate_data(n_unique=n_unique2, dup_per_id=dup)
    print(f"\n场景2: {n_unique2} 个唯一被试 × {dup} 行 = {n_unique2 * dup} 行（重度重复）")
    print(f"  chunk_size = 7（每个被试的多行被分散在不同 chunk 中）")
    print()
    rc = run_comparison(data_path2, chunk_size=7)
    if rc != 0:
        total_failures += 1
    os.remove(data_path2)

    print()
    print("=" * 70)
    if total_failures == 0:
        print("✅ 跨 chunk 重复被试测试全部通过")
    else:
        print(f"❌ 跨 chunk 重复被试测试：{total_failures} 组失败")

    if expected_notes:
        print("📝 预期差异说明:")
        for note in expected_notes:
            print(f"   - {note}")

    return total_failures


def test_high_anomaly_ratio():
    """高异常占比测试：>70% 异常、100% 异常"""
    print()
    print("=" * 70)
    print("边界测试：高异常占比 (>70%)")
    print("=" * 70)

    total_failures = 0
    expected_notes = []

    data_path = build_high_anomaly_data(n_rows=100, anomaly_ratio=0.8)
    print(f"场景1: 100 行，约 80% 为异常答题")
    print(f"  chunk_size = 15")
    print()
    rc = run_comparison(data_path, chunk_size=15)
    if rc != 0:
        total_failures += 1
    os.remove(data_path)

    data_path2 = build_high_anomaly_data(n_rows=50, anomaly_ratio=1.0)
    print(f"\n场景2: 50 行，100% 全为异常答题")
    print(f"  chunk_size = 10")
    print()
    rc = run_comparison(data_path2, chunk_size=10)
    if rc != 0:
        total_failures += 1
    os.remove(data_path2)

    print()
    print("=" * 70)
    if total_failures == 0:
        print("✅ 高异常占比测试全部通过")
    else:
        print(f"❌ 高异常占比测试：{total_failures} 组失败")

    if expected_notes:
        print("📝 预期差异说明:")
        for note in expected_notes:
            print(f"   - {note}")

    return total_failures


def run_boundary_tests():
    """运行所有边界测试"""
    print()
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "边 界 测 试 套 件" + " " * 28 + "║")
    print("╚" + "═" * 68 + "╝")

    exit_code = 0
    exit_code |= test_extreme_chunk_sizes()
    exit_code |= test_cross_chunk_duplicates()
    exit_code |= test_high_anomaly_ratio()
    return exit_code


if __name__ == "__main__":
    exit_code = 0

    exit_code |= test_alias_config()

    sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data", "sample_survey.csv")
    if os.path.exists(sample_path):
        exit_code |= run_comparison(sample_path, chunk_size=30)

    big_path = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data", "big_survey_mixed_columns.csv"))
    if os.path.exists(big_path):
        exit_code |= run_comparison(big_path, chunk_size=500)

    exit_code |= run_boundary_tests()

    sys.exit(exit_code)
