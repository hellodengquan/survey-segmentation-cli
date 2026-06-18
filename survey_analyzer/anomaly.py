import pandas as pd
import numpy as np
import logging

from .config import AnomalyThresholds

logger = logging.getLogger(__name__)


def detect_anomalies(
    df: pd.DataFrame,
    question_cols: list[str],
    id_col: str = "respondent_id",
    time_col: str = "duration_seconds",
    thresholds: AnomalyThresholds = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = AnomalyThresholds()

    anomaly_records = []

    if id_col not in df.columns:
        df = df.copy()
        df[id_col] = range(1, len(df) + 1)

    valid_q_cols = [c for c in question_cols if c in df.columns]

    speed_anomalies = _detect_speeding(
        df, id_col, time_col, thresholds.speed_threshold_seconds
    )
    anomaly_records.extend(speed_anomalies)

    straight_anomalies = _detect_straight_lining(
        df, id_col, valid_q_cols,
        thresholds.straight_line_min_cols,
        thresholds.straight_line_ratio,
    )
    anomaly_records.extend(straight_anomalies)

    pattern_anomalies = _detect_pattern_answers(
        df, id_col, valid_q_cols, thresholds.pattern_min_cols
    )
    anomaly_records.extend(pattern_anomalies)

    if thresholds.contradiction_pairs:
        contra_anomalies = _detect_contradictions(
            df, id_col, thresholds.contradiction_pairs
        )
        anomaly_records.extend(contra_anomalies)

    outlier_anomalies = _detect_numeric_outliers(
        df, id_col, valid_q_cols,
        thresholds.numeric_outlier_iqr_multiplier,
        thresholds.numeric_outlier_min_rows,
    )
    anomaly_records.extend(outlier_anomalies)

    if not anomaly_records:
        logger.info("未检测到异常回答")
        return pd.DataFrame(columns=["受访者ID", "异常类型", "异常详情"])

    result = pd.DataFrame(anomaly_records)
    result = result.sort_values(["受访者ID", "异常类型"]).reset_index(drop=True)
    logger.info("共检测到 %d 条异常记录，涉及 %d 位受访者", len(result), result["受访者ID"].nunique())
    return result


def detect_chunk_anomalies(
    chunk: pd.DataFrame,
    question_cols: list[str],
    id_col: str = "respondent_id",
    time_col: str = "duration_seconds",
    thresholds: AnomalyThresholds = None,
) -> list[dict]:
    if thresholds is None:
        thresholds = AnomalyThresholds()

    anomaly_records = []

    if id_col not in chunk.columns:
        chunk = chunk.copy()
        chunk[id_col] = range(1, len(chunk) + 1)

    valid_q_cols = [c for c in question_cols if c in chunk.columns]

    anomaly_records.extend(_detect_speeding(chunk, id_col, time_col, thresholds.speed_threshold_seconds))
    anomaly_records.extend(_detect_straight_lining(chunk, id_col, valid_q_cols, thresholds.straight_line_min_cols, thresholds.straight_line_ratio))
    anomaly_records.extend(_detect_pattern_answers(chunk, id_col, valid_q_cols, thresholds.pattern_min_cols))

    if thresholds.contradiction_pairs:
        anomaly_records.extend(_detect_contradictions(chunk, id_col, thresholds.contradiction_pairs))

    return anomaly_records


def detect_global_anomalies(
    df: pd.DataFrame,
    question_cols: list[str],
    id_col: str = "respondent_id",
    thresholds: AnomalyThresholds = None,
) -> list[dict]:
    if thresholds is None:
        thresholds = AnomalyThresholds()

    anomaly_records = []
    valid_q_cols = [c for c in question_cols if c in df.columns]

    anomaly_records.extend(_detect_numeric_outliers(
        df, id_col, valid_q_cols,
        thresholds.numeric_outlier_iqr_multiplier,
        thresholds.numeric_outlier_min_rows,
    ))

    return anomaly_records


def merge_anomaly_results(chunk_records: list[list[dict]], global_records: list[dict] = None) -> pd.DataFrame:
    all_records = []
    for records in chunk_records:
        all_records.extend(records)
    if global_records:
        all_records.extend(global_records)

    if not all_records:
        logger.info("未检测到异常回答")
        return pd.DataFrame(columns=["受访者ID", "异常类型", "异常详情"])

    result = pd.DataFrame(all_records)
    result = result.drop_duplicates(subset=["受访者ID", "异常类型", "异常详情"])
    result = result.sort_values(["受访者ID", "异常类型"]).reset_index(drop=True)
    logger.info("共检测到 %d 条异常记录，涉及 %d 位受访者", len(result), result["受访者ID"].nunique())
    return result


def _detect_speeding(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    threshold: float,
) -> list[dict]:
    records = []
    if time_col not in df.columns:
        logger.info("未找到作答时长列 '%s'，跳过速度过快检测", time_col)
        return records

    speeders = df[df[time_col] < threshold]
    for _, row in speeders.iterrows():
        records.append({
            "受访者ID": row[id_col],
            "异常类型": "作答过快",
            "异常详情": f"作答时长仅 {row[time_col]:.0f} 秒（低于 {threshold:.0f} 秒阈值）",
        })
    logger.info("检测到 %d 条作答过快记录（阈值 %.0f 秒）", len(records), threshold)
    return records


def _detect_straight_lining(
    df: pd.DataFrame,
    id_col: str,
    question_cols: list[str],
    min_cols: int,
    ratio: float,
) -> list[dict]:
    records = []
    cat_cols = [c for c in question_cols if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in question_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    if len(cat_cols) >= min_cols:
        _check_straight_for_cols(df, id_col, cat_cols, min_cols, ratio, "类别题", records)

    if len(num_cols) >= min_cols:
        _check_straight_for_cols(df, id_col, num_cols, min_cols, ratio, "数值题", records)

    if len(cat_cols) < min_cols and len(num_cols) < min_cols and len(question_cols) >= min_cols:
        _check_straight_for_cols(df, id_col, question_cols, min_cols, ratio, "", records)

    logger.info("检测到 %d 条直线作答记录（≥%d 题，一致率≥%.0f%%）",
                len(records), min_cols, ratio * 100)
    return records


def _check_straight_for_cols(
    df: pd.DataFrame,
    id_col: str,
    cols: list[str],
    min_cols: int,
    ratio: float,
    label: str,
    records: list[dict],
):
    q_data = df[cols]
    for idx, row in q_data.iterrows():
        non_na = row.dropna()
        if len(non_na) < min_cols:
            continue

        mode = non_na.mode()
        if len(mode) == 0:
            continue
        most_common = mode.iloc[0]
        same_count = (non_na == most_common).sum()
        same_ratio = same_count / len(non_na)

        if same_ratio >= ratio:
            detail_label = f"（{label}）" if label else ""
            records.append({
                "受访者ID": df.loc[idx, id_col],
                "异常类型": "直线作答",
                "异常详情": f"{len(non_na)} 道题{detail_label}中有 {same_count} 道选择了相同答案「{most_common}」（一致率 {same_ratio:.0%}）",
            })


def _detect_pattern_answers(
    df: pd.DataFrame,
    id_col: str,
    question_cols: list[str],
    min_cols: int,
) -> list[dict]:
    records = []
    if len(question_cols) < min_cols:
        logger.info("题目数少于 %d，跳过规律性作答检测", min_cols)
        return records

    q_data = df[question_cols]
    for idx, row in q_data.iterrows():
        non_na = row.dropna()
        if len(non_na) < min_cols:
            continue

        vals = list(non_na.values)
        try:
            numeric_vals = [float(v) for v in vals]
        except (ValueError, TypeError):
            continue

        diffs = [numeric_vals[i + 1] - numeric_vals[i] for i in range(len(numeric_vals) - 1)]
        if len(set(diffs)) == 1 and diffs[0] != 0:
            records.append({
                "受访者ID": df.loc[idx, id_col],
                "异常类型": "规律性作答",
                "异常详情": f"连续 {len(numeric_vals)} 道题呈等差规律（步长={diffs[0]:.1f}）",
            })
    logger.info("检测到 %d 条规律性作答记录（≥%d 题）", len(records), min_cols)
    return records


def _detect_contradictions(
    df: pd.DataFrame,
    id_col: str,
    pairs: list,
) -> list[dict]:
    records = []
    for pair in pairs:
        if len(pair) == 3:
            col1, col2, contradiction_set = pair
        elif len(pair) == 2:
            col1, col2 = pair
            contradiction_set = {("非常满意", "肯定不会"), ("非常满意", "从不"),
                                 ("非常不满意", "肯定会"), ("非常不满意", "总是")}
        else:
            continue

        if col1 not in df.columns or col2 not in df.columns:
            logger.warning("矛盾检测: 列 '%s' 或 '%s' 不存在", col1, col2)
            continue

        for idx, row in df.iterrows():
            v1, v2 = row[col1], row[col2]
            if pd.isna(v1) or pd.isna(v2):
                continue

            pair_key = (str(v1), str(v2))
            reverse_key = (str(v2), str(v1))
            if pair_key in contradiction_set or reverse_key in contradiction_set:
                records.append({
                    "受访者ID": row[id_col],
                    "异常类型": "矛盾回答",
                    "异常详情": f"「{col1}」={v1} 与「{col2}」={v2} 存在逻辑矛盾",
                })
    logger.info("检测到 %d 条矛盾回答记录", len(records))
    return records


def _detect_numeric_outliers(
    df: pd.DataFrame,
    id_col: str,
    question_cols: list[str],
    iqr_multiplier: float,
    min_rows: int,
) -> list[dict]:
    records = []
    numeric_cols = [c for c in question_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < min_rows:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        outliers = df[(df[col] < lower) | (df[col] > upper)]
        for idx, row in outliers.iterrows():
            records.append({
                "受访者ID": row[id_col],
                "异常类型": "数值异常",
                "异常详情": f"「{col}」={row[col]:.2f}（正常范围 {lower:.2f}~{upper:.2f}，IQR×{iqr_multiplier}）",
            })
    logger.info("检测到 %d 条数值异常记录（IQR×%.1f，≥%d 行）",
                len(records), iqr_multiplier, min_rows)
    return records


def get_anomaly_summary(anomaly_df: pd.DataFrame) -> pd.DataFrame:
    if anomaly_df.empty:
        return pd.DataFrame(columns=["异常类型", "人次", "说明"])

    summary = (
        anomaly_df.groupby("异常类型")
        .agg(
            人次=("异常类型", "size"),
            示例=("异常详情", "first"),
        )
        .reset_index()
    )

    type_descriptions = {
        "作答过快": "受访者完成问卷的时间远低于正常水平，可能未认真阅读题目",
        "直线作答": "受访者对大量题目选择了相同答案，可能存在敷衍行为",
        "规律性作答": "受访者的答案呈明显等差规律，可能为随意填答",
        "矛盾回答": "受访者在不同题目中的回答存在逻辑矛盾",
        "数值异常": "受访者的数值回答远超正常范围",
    }

    summary["说明"] = summary["异常类型"].map(type_descriptions).fillna("需要研究员复核")
    summary = summary.sort_values("人次", ascending=False).reset_index(drop=True)
    return summary
