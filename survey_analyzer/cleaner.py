import pandas as pd
import re
import logging

from .config import CleaningConfig

logger = logging.getLogger(__name__)


def clean_survey_data(
    df: pd.DataFrame,
    id_col: str = "respondent_id",
    time_col: str = "duration_seconds",
    config: CleaningConfig = None,
    **kwargs,
) -> pd.DataFrame:
    if config is None:
        config = CleaningConfig(
            drop_empty_rows=kwargs.get("drop_empty_rows", True),
            empty_threshold=kwargs.get("empty_threshold", 0.7),
            trim_whitespace=kwargs.get("trim_whitespace", True),
            lowercase_strings=kwargs.get("lowercase_strings", True),
            fill_na_numeric=kwargs.get("fill_na_numeric", -1),
        )

    logger.info("开始清洗问卷数据，原始行数: %d，列数: %d", len(df), len(df.columns))

    df = df.copy()

    if config.trim_whitespace:
        str_cols = df.select_dtypes(include=["object"]).columns
        for col in str_cols:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        logger.info("已清理字符串列的前后空白，涉及 %d 列", len(str_cols))

    if config.lowercase_strings:
        str_cols = df.select_dtypes(include=["object"]).columns
        for col in str_cols:
            df[col] = df[col].apply(lambda x: x.lower() if isinstance(x, str) else x)
        logger.info("已将字符串列转为小写，涉及 %d 列", len(str_cols))

    if config.standardize_yes_no:
        df = _standardize_yes_no(df)
    df = _standardize_na_values(df)

    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(config.fill_na_numeric)

    if config.drop_empty_rows:
        before = len(df)
        df = _drop_mostly_empty(df, threshold=config.empty_threshold)
        logger.info("删除缺失率超过 %.0f%% 的行: %d -> %d", config.empty_threshold * 100, before, len(df))

    if id_col in df.columns:
        df = df.drop_duplicates(subset=[id_col])
        logger.info("按 %s 去重后行数: %d", id_col, len(df))

    logger.info("清洗完成，最终行数: %d，列数: %d", len(df), len(df.columns))
    return df


def _standardize_yes_no(df: pd.DataFrame) -> pd.DataFrame:
    yes_pattern = re.compile(r"^(yes|是|y|true|1|对|同意)$", re.IGNORECASE)
    no_pattern = re.compile(r"^(no|否|n|false|0|错|不同意)$", re.IGNORECASE)

    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 10:
            matches_yes = sum(1 for v in unique_vals if yes_pattern.match(str(v)))
            matches_no = sum(1 for v in unique_vals if no_pattern.match(str(v)))
            if matches_yes > 0 and matches_no > 0 and (matches_yes + matches_no) == len(unique_vals):
                df[col] = df[col].apply(
                    lambda x: "是" if yes_pattern.match(str(x)) else ("否" if no_pattern.match(str(x)) else x)
                )
                logger.info("列 '%s' 已标准化为 是/否", col)
    return df


def _standardize_na_values(df: pd.DataFrame) -> pd.DataFrame:
    na_patterns = [
        "nan", "none", "n/a", "na", "null", "无", "不适用",
        "不知道", "不清楚", "未填写", "-", "--", "…", "",
    ]
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].apply(lambda x: pd.NA if isinstance(x, str) and x.strip().lower() in na_patterns else x)
    return df


def _drop_mostly_empty(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    non_na_ratio = df.notna().sum(axis=1) / len(df.columns)
    return df[non_na_ratio >= (1 - threshold)]


def identify_column_types(df: pd.DataFrame, id_col: str = "respondent_id") -> dict:
    col_types = {"id": [], "demographic": [], "question": [], "timing": [], "other": []}

    demo_keywords = ["age", "gender", "region", "education", "income", "occupation", "年龄", "性别", "地区", "学历", "收入", "职业"]
    time_keywords = ["duration", "duration_seconds", "start_time", "end_time", "submit_time", "时长", "开始时间", "结束时间", "提交时间"]

    def _keyword_match(col_lower, keywords):
        for kw in keywords:
            if kw in col_lower:
                if len(kw) <= 3:
                    import re
                    if re.search(rf'\b{re.escape(kw)}\b', col_lower):
                        return True
                else:
                    return True
        return False

    for col in df.columns:
        col_lower = col.lower()
        if col == id_col:
            col_types["id"].append(col)
        elif _keyword_match(col_lower, time_keywords):
            col_types["timing"].append(col)
        elif _keyword_match(col_lower, demo_keywords):
            col_types["demographic"].append(col)
        elif df[col].dtype in ["object", "string"] or df[col].nunique() <= 20:
            col_types["question"].append(col)
        else:
            col_types["other"].append(col)

    logger.info("列类型识别结果: %s", {k: len(v) for k, v in col_types.items()})
    return col_types
