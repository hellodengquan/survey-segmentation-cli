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
from .schema_validator import SchemaValidator, ValidationResult
from .config import load_config, get_available_survey_types, AppConfig


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    survey_types = get_available_survey_types()

    parser = argparse.ArgumentParser(
        prog="survey-analyzer",
        description="问卷数据分析工具 —— 清洗、分群、统计、异常检测一步到位",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
可用的问卷类型预设 (--survey-type):
  {', '.join(survey_types)}

支持的配置文件格式:
  - JSON (.json)
  - TOML (.toml, .tml)  - 需 Python 3.11+
""",
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

    basic_group = parser.add_argument_group("基本配置")
    basic_group.add_argument(
        "--id-col",
        default=None,
        help="受访者ID列名（默认: respondent_id）",
    )
    basic_group.add_argument(
        "--time-col",
        default=None,
        help="作答时长列名（默认: duration_seconds）",
    )
    basic_group.add_argument(
        "--segment-cols",
        nargs="+",
        default=None,
        help="手动指定分群列名（空格分隔），不指定则自动识别人口统计学列",
    )
    basic_group.add_argument(
        "--survey-type",
        choices=survey_types,
        default=None,
        help="选择问卷类型预设，自动加载相应阈值配置",
    )
    basic_group.add_argument(
        "--config",
        default=None,
        help="配置文件路径，支持 .json 或 .toml 格式",
    )
    basic_group.add_argument(
        "--encoding",
        default="utf-8",
        help="CSV文件编码（默认: utf-8，可设 auto 自动检测）",
    )

    anomaly_group = parser.add_argument_group("异常检测阈值")
    anomaly_group.add_argument(
        "--speed-threshold",
        type=float,
        default=None,
        help="作答过快阈值（秒），低于此值标记为异常",
    )
    anomaly_group.add_argument(
        "--straight-ratio",
        type=float,
        default=None,
        help="直线作答判定比率（0-1），如 0.8 表示80%%题目选同一答案则判定",
    )
    anomaly_group.add_argument(
        "--straight-min-cols",
        type=int,
        default=None,
        help="直线作答最少题目数，低于此数量则不检测",
    )
    anomaly_group.add_argument(
        "--pattern-min-cols",
        type=int,
        default=None,
        help="规律性作答最少题目数",
    )
    anomaly_group.add_argument(
        "--outlier-iqr",
        type=float,
        default=None,
        help="数值异常的 IQR 乘数（默认: 3.0）",
    )
    anomaly_group.add_argument(
        "--outlier-min-rows",
        type=int,
        default=None,
        help="数值异常检测的最少数据行数",
    )

    cleaning_group = parser.add_argument_group("数据清洗配置")
    cleaning_group.add_argument(
        "--empty-threshold",
        type=float,
        default=None,
        help="行缺失率阈值（0-1），超过则删除该行",
    )
    cleaning_group.add_argument(
        "--no-auto-clean",
        action="store_true",
        help="跳过自动数据清洗",
    )
    cleaning_group.add_argument(
        "--no-trim",
        action="store_true",
        help="不清理字符串的前后空白",
    )
    cleaning_group.add_argument(
        "--no-lowercase",
        action="store_true",
        help="不将字符串转为小写",
    )
    cleaning_group.add_argument(
        "--no-standardize-yesno",
        action="store_true",
        help="不自动标准化 是/否 类答案",
    )

    segment_group = parser.add_argument_group("分群配置")
    segment_group.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="自动分群的最大群组数（默认: 15）",
    )
    segment_group.add_argument(
        "--min-categories",
        type=int,
        default=None,
        help="分群列的最少类别数（默认: 2）",
    )
    segment_group.add_argument(
        "--max-categories",
        type=int,
        default=None,
        help="分群列的最大类别数（默认: 8）",
    )

    report_group = parser.add_argument_group("报告配置")
    report_group.add_argument(
        "--no-cross-tabs",
        action="store_true",
        help="报告中不包含交叉分析表",
    )
    report_group.add_argument(
        "--no-numeric-summary",
        action="store_true",
        help="报告中不包含数值统计",
    )
    report_group.add_argument(
        "--no-anomalies",
        action="store_true",
        help="报告中不包含异常检测结果",
    )
    report_group.add_argument(
        "--no-raw-data",
        action="store_true",
        help="报告中不包含清洗后的数据",
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="跳过 Schema 校验（不推荐）",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="输出详细日志",
    )

    return parser


def _build_cli_overrides(args: argparse.Namespace) -> dict:
    overrides = {}

    if args.speed_threshold is not None:
        overrides["speed_threshold_seconds"] = args.speed_threshold
    if args.straight_ratio is not None:
        overrides["straight_line_ratio"] = args.straight_ratio
    if args.straight_min_cols is not None:
        overrides["straight_line_min_cols"] = args.straight_min_cols
    if args.pattern_min_cols is not None:
        overrides["pattern_min_cols"] = args.pattern_min_cols
    if args.outlier_iqr is not None:
        overrides["numeric_outlier_iqr_multiplier"] = args.outlier_iqr
    if args.outlier_min_rows is not None:
        overrides["numeric_outlier_min_rows"] = args.outlier_min_rows

    if args.empty_threshold is not None:
        overrides["empty_threshold"] = args.empty_threshold
    if args.no_auto_clean:
        overrides["drop_empty_rows"] = False
    if args.no_trim:
        overrides["trim_whitespace"] = False
    if args.no_lowercase:
        overrides["lowercase_strings"] = False
    if args.no_standardize_yesno:
        overrides["standardize_yes_no"] = False

    if args.max_groups is not None:
        overrides["max_groups"] = args.max_groups
    if args.min_categories is not None:
        overrides["min_categories"] = args.min_categories
    if args.max_categories is not None:
        overrides["max_categories"] = args.max_categories

    if args.no_cross_tabs:
        overrides["include_cross_tabs"] = False
    if args.no_numeric_summary:
        overrides["include_numeric_summary"] = False
    if args.no_anomalies:
        overrides["include_anomalies"] = False
    if args.no_raw_data:
        overrides["include_raw_data"] = False

    if args.id_col is not None:
        overrides["id_col"] = args.id_col
    if args.time_col is not None:
        overrides["time_col"] = args.time_col
    if args.encoding is not None:
        overrides["encoding"] = args.encoding

    if args.segment_cols is not None:
        overrides["segment_cols"] = args.segment_cols

    return overrides


def _print_validation_result(result: ValidationResult) -> None:
    if result.warnings:
        print("\n⚠️  校验警告:")
        for w in result.warnings:
            print(f"   - {w}")
        print()

    if not result.valid:
        print("\n❌  校验失败，请解决以下问题后重试:")
        for e in result.errors:
            print(f"   - {e}")
        print()


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

    cli_overrides = _build_cli_overrides(args)

    logger.info("=" * 60)
    logger.info("问卷数据分析工具 启动")
    logger.info("=" * 60)

    if not args.skip_validation:
        logger.info("── 前置校验: Schema 与编码检查 ──")
        encoding_to_check = args.encoding if args.encoding else "utf-8"
        validation = SchemaValidator.full_validation(
            args.input,
            encoding=encoding_to_check,
        )
        _print_validation_result(validation)

        if not validation.valid:
            logger.error("数据校验未通过，已终止")
            sys.exit(2)

        encoding_to_use = validation.detected_encoding
        has_fallback = any(w.code == "ENCODING_FALLBACK" for w in validation.warnings)
        if args.encoding and args.encoding.lower() != "auto" and not has_fallback:
            encoding_to_use = args.encoding
        logger.info("使用编码: %s", encoding_to_use)
        cli_overrides["encoding"] = encoding_to_use
    else:
        logger.info("── 跳过 Schema 校验 ──")
        encoding_to_use = args.encoding

    logger.info("加载配置...")
    config = load_config(
        config_file=args.config,
        survey_type=args.survey_type,
        cli_overrides=cli_overrides,
    )
    logger.info("使用问卷类型预设: %s", config.survey_type)

    logger.info("读取数据: %s", args.input)
    df = read_data(args.input, encoding=config.encoding)
    logger.info("原始数据: %d 行 × %d 列", len(df), len(df.columns))

    if not args.no_auto_clean:
        logger.info("── 步骤 1/5: 数据清洗 ──")
        df = clean_survey_data(
            df,
            id_col=config.id_col,
            time_col=config.time_col,
            config=config.cleaning,
        )
    else:
        logger.info("── 跳过数据清洗 ──")

    logger.info("── 步骤 2/5: 列类型识别 ──")
    col_types = identify_column_types(df, id_col=config.id_col)
    for k, v in col_types.items():
        if v:
            logger.info("  %s: %s", k, v)

    logger.info("── 步骤 3/5: 用户分群 ──")
    if config.segment.segment_cols:
        df = segment_by_columns(df, config.segment.segment_cols)
    else:
        df = auto_segment(df, col_types, config.segment)
    segment_summary = get_segment_summary(df)

    question_cols = col_types.get("question", [])
    numeric_q_cols = [c for c in question_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    logger.info("── 步骤 4/5: 题目分布统计 ──")
    distributions = compute_question_distribution(df, question_cols)
    cross_tabs = compute_cross_tabulation(df, question_cols) if config.report.include_cross_tabs else {}
    num_summary = compute_numeric_summary(df, numeric_q_cols) if config.report.include_numeric_summary else pd.DataFrame()

    logger.info("── 步骤 5/5: 异常回答检测 ──")
    if config.report.include_anomalies:
        anomaly_df = detect_anomalies(
            df,
            question_cols=question_cols,
            id_col=config.id_col,
            time_col=config.time_col,
            thresholds=config.anomaly,
        )
        anomaly_summary = get_anomaly_summary(anomaly_df)
    else:
        anomaly_df = pd.DataFrame()
        anomaly_summary = pd.DataFrame()

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
    group_count = segment_summary["_群组"].nunique() if "_群组" in segment_summary.columns else len(segment_summary)
    print(f"   分群数: {group_count} 个")
    print(f"   问卷类型: {config.survey_type}")
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
        print(f"\n❌ 文件未找到: {e}", file=sys.stderr)
        print("   请检查文件路径是否正确\n", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ 参数错误: {e}", file=sys.stderr)
        print("   使用 --help 查看可用参数\n", file=sys.stderr)
        sys.exit(1)
    except UnicodeDecodeError as e:
        print(f"\n❌ 编码错误: 无法使用指定编码读取文件", file=sys.stderr)
        print(f"   尝试使用 --encoding auto 自动检测，或手动指定正确的编码格式", file=sys.stderr)
        print(f"   错误详情: {e}\n", file=sys.stderr)
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"\n❌ 数据解析错误: CSV 文件格式不正确", file=sys.stderr)
        print(f"   请检查文件是否损坏，或分隔符是否正确", file=sys.stderr)
        print(f"   错误详情: {e}\n", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  用户中断", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 运行出错: {type(e).__name__}: {e}", file=sys.stderr)
        print("   如果问题持续，请检查数据格式或联系技术支持\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
