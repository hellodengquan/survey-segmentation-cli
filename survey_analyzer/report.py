import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_report(
    output_path: str,
    cleaned_df: pd.DataFrame,
    col_types: dict,
    segment_summary: pd.DataFrame,
    distributions: dict[str, pd.DataFrame],
    cross_tabs: dict[str, pd.DataFrame],
    anomaly_df: pd.DataFrame,
    anomaly_summary: pd.DataFrame,
    numeric_summary: pd.DataFrame,
) -> str:
    logger.info("开始生成报告: %s", output_path)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        _write_overview(writer, cleaned_df, col_types)
        _write_segment_summary(writer, segment_summary)
        _write_distributions(writer, distributions)
        _write_cross_tabs(writer, cross_tabs)
        _write_numeric_summary(writer, numeric_summary)
        _write_anomalies(writer, anomaly_df, anomaly_summary)
        _write_cleaned_data(writer, cleaned_df)

    logger.info("报告生成完成: %s", output_path)
    return output_path


def _write_overview(writer, df, col_types):
    overview_data = {
        "指标": [
            "报告生成时间",
            "有效受访者数",
            "总列数",
            "人口统计学列数",
            "问卷题目列数",
            "时长/时间列数",
            "其他列数",
        ],
        "值": [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            len(df),
            len(df.columns),
            len(col_types.get("demographic", [])),
            len(col_types.get("question", [])),
            len(col_types.get("timing", [])),
            len(col_types.get("other", [])),
        ],
    }
    overview_df = pd.DataFrame(overview_data)
    overview_df.to_excel(writer, sheet_name="总览", index=False)
    logger.info("已写入「总览」工作表")


def _write_segment_summary(writer, segment_summary):
    if segment_summary is not None and not segment_summary.empty:
        segment_summary.to_excel(writer, sheet_name="分群汇总", index=False)
        logger.info("已写入「分群汇总」工作表")


def _write_distributions(writer, distributions):
    if not distributions:
        return

    all_rows = []
    for col, dist_df in distributions.items():
        temp = dist_df.copy()
        temp.insert(0, "题目", col)
        all_rows.append(temp)

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_excel(writer, sheet_name="题目分布", index=False)
    logger.info("已写入「题目分布」工作表，共 %d 题", len(distributions))


def _write_cross_tabs(writer, cross_tabs):
    if not cross_tabs:
        return

    all_rows = []
    for col, cross_df in cross_tabs.items():
        temp = cross_df.copy()
        temp.insert(0, "题目", col)
        all_rows.append(temp)

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_excel(writer, sheet_name="交叉分析", index=False)
    logger.info("已写入「交叉分析」工作表，共 %d 题", len(cross_tabs))


def _write_numeric_summary(writer, numeric_summary):
    if numeric_summary is not None and not numeric_summary.empty:
        numeric_summary.to_excel(writer, sheet_name="数值统计", index=False)
        logger.info("已写入「数值统计」工作表")


def _write_anomalies(writer, anomaly_df, anomaly_summary):
    if anomaly_summary is not None and not anomaly_summary.empty:
        anomaly_summary.to_excel(writer, sheet_name="异常汇总", index=False)
        logger.info("已写入「异常汇总」工作表")

    if anomaly_df is not None and not anomaly_df.empty:
        anomaly_df.to_excel(writer, sheet_name="异常明细", index=False)
        logger.info("已写入「异常明细」工作表，共 %d 条", len(anomaly_df))


def _write_cleaned_data(writer, df):
    display_df = df.copy()
    if "_群组" in display_df.columns:
        display_df = display_df.drop(columns=["_群组"])
    display_df.to_excel(writer, sheet_name="清洗后数据", index=False)
    logger.info("已写入「清洗后数据」工作表")
