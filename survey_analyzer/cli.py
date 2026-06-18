import argparse
import sys
import logging
import os

import pandas as pd

from .cleaner import clean_survey_data, identify_column_types
from .segmenter import segment_by_columns, auto_segment, get_segment_summary
from .stats import compute_question_distribution, compute_cross_tabulation, compute_numeric_summary
from .anomaly import detect_anomalies, get_anomaly_summary
from .report import generate_report


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="survey-analyzer",
        description="问卷数据分析工具 —— 清洗、分群、统计、异常检测一步到位",
    )

    parser.add_argument(
        "input",
        help="问卷数据文件路径（支持 .csv / .xlsx / .xls）",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="输出报告路径（默认: 与输入同目录，文件名加 _report.xlsx 后缀）",
    )
    parser.add_argument(
        "--id-col",
        default="respondent_id",
        help="受访者ID列名（默认: respondent_id）",
    )
    parser.add_argument(
        "--time-col",
        default="duration_seconds",
        help="作答时长列名（默认: duration_seconds）",
    )
    parser.add_argument(
        "--segment-cols",
        nargs="+",
        default=None,
        help="手动指定分群列名（空格分隔），不指定则自动识别人口统计学列",
    )
    parser.add_argument(
        "--speed-threshold",
        type=float,
        default=60,
        help="作答过快阈值（秒），低于此值标记为异常（默认: 60）",
    )
    parser.add_argument(
        "--straight-ratio",
        type=float,
        default=0.8,
        help="直线作答判定比率（默认: 0.8，即80%%题目选同一答案则判定）",
    )
    parser.add_argument(
        "--empty-threshold",
        type=float,
        default=0.7,
        help="行缺失率阈值，超过则删除该行（默认: 0.7）",
    )
    parser.add_argument(
        "--no-auto-clean",
        action="store_true",
        help="跳过自动数据清洗",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="输出详细日志",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="CSV文件编码（默认: utf-8）",
    )

    return parser


def read_data(filepath: str, encoding: str = "utf-8") -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return pd.read_csv(filepath, encoding=encoding)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，请使用 .csv / .xlsx / .xls")


def resolve_output_path(input_path: str, output: str = None) -> str:
    if output:
        return output
    base, _ = os.path.splitext(input_path)
    return f"{base}_report.xlsx"


def run(args: argparse.Namespace):
    setup_logging(args.verbose)
    logger = logging.getLogger("survey_analyzer")

    logger.info("=" * 60)
    logger.info("问卷数据分析工具 启动")
    logger.info("=" * 60)

    logger.info("读取数据: %s", args.input)
    df = read_data(args.input, encoding=args.encoding)
    logger.info("原始数据: %d 行 × %d 列", len(df), len(df.columns))

    if not args.no_auto_clean:
        logger.info("── 步骤 1/5: 数据清洗 ──")
        df = clean_survey_data(
            df,
            id_col=args.id_col,
            time_col=args.time_col,
            empty_threshold=args.empty_threshold,
        )
    else:
        logger.info("── 跳过数据清洗 ──")

    logger.info("── 步骤 2/5: 列类型识别 ──")
    col_types = identify_column_types(df, id_col=args.id_col)
    for k, v in col_types.items():
        if v:
            logger.info("  %s: %s", k, v)

    logger.info("── 步骤 3/5: 用户分群 ──")
    if args.segment_cols:
        df = segment_by_columns(df, args.segment_cols)
    else:
        df = auto_segment(df, col_types)
    segment_summary = get_segment_summary(df)

    question_cols = col_types.get("question", [])
    numeric_q_cols = [c for c in question_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_q_cols = [c for c in question_cols if c not in numeric_q_cols]

    logger.info("── 步骤 4/5: 题目分布统计 ──")
    distributions = compute_question_distribution(df, question_cols)
    cross_tabs = compute_cross_tabulation(df, question_cols)
    num_summary = compute_numeric_summary(df, numeric_q_cols)

    logger.info("── 步骤 5/5: 异常回答检测 ──")
    anomaly_df = detect_anomalies(
        df,
        question_cols=question_cols,
        id_col=args.id_col,
        time_col=args.time_col,
        speed_threshold_seconds=args.speed_threshold,
        straight_line_ratio=args.straight_ratio,
    )
    anomaly_summary = get_anomaly_summary(anomaly_df)

    output_path = resolve_output_path(args.input, args.output)
    generate_report(
        output_path=output_path,
        cleaned_df=df,
        col_types=col_types,
        segment_summary=segment_summary,
        distributions=distributions,
        cross_tabs=cross_tabs,
        anomaly_df=anomaly_df,
        anomaly_summary=anomaly_summary,
        numeric_summary=num_summary,
    )

    logger.info("=" * 60)
    logger.info("分析完成！报告已保存至: %s", os.path.abspath(output_path))
    logger.info("=" * 60)

    print(f"\n✅ 分析完成！报告已保存至: {os.path.abspath(output_path)}")
    print(f"   有效受访者: {len(df)} 人")
    print(f"   分析题目数: {len(question_cols)} 题")
    print(f"   分群数: {segment_summary['_群组'].nunique() if '_群组' in segment_summary.columns else len(segment_summary)} 个")
    if not anomaly_df.empty:
        print(f"   异常回答: {anomaly_df['受访者ID'].nunique()} 位受访者需复核")
    else:
        print(f"   异常回答: 未检测到")


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"❌ 参数错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ 运行出错: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
