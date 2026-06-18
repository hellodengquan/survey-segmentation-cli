import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_question_distribution(df: pd.DataFrame, question_cols: list[str]) -> dict[str, pd.DataFrame]:
    results = {}
    for col in question_cols:
        if col not in df.columns:
            logger.warning("列 '%s' 不存在，跳过", col)
            continue

        counts = df[col].value_counts(dropna=False)
        total = len(df)

        dist = pd.DataFrame({
            "选项": counts.index.astype(str),
            "人数": counts.values,
        })
        dist["占比"] = (dist["人数"] / total * 100).round(1)
        dist["占比（显示）"] = dist["占比"].astype(str) + "%"
        dist["累计占比"] = dist["占比"].cumsum().round(1).astype(str) + "%"
        dist = dist.reset_index(drop=True)

        na_count = int(df[col].isna().sum())
        if na_count > 0:
            dist.loc[len(dist)] = {
                "选项": "（缺失值）",
                "人数": na_count,
                "占比": round(na_count / total * 100, 1),
                "占比（显示）": f"{round(na_count / total * 100, 1)}%",
                "累计占比": "",
            }

        results[col] = dist
        logger.info("题目 '%s' 分布统计完成，共 %d 个选项", col, len(dist))

    return results


def compute_cross_tabulation(
    df: pd.DataFrame,
    question_cols: list[str],
    segment_col: str = "_群组",
) -> dict[str, pd.DataFrame]:
    if segment_col not in df.columns:
        logger.warning("分群列 '%s' 不存在，跳过交叉分析", segment_col)
        return {}

    results = {}
    for col in question_cols:
        if col not in df.columns:
            continue

        ct = pd.crosstab(df[segment_col], df[col], margins=True, margins_name="合计")
        ct_pct = pd.crosstab(df[segment_col], df[col], normalize="index").round(3) * 100

        cross = ct.copy().astype(str)
        for c in ct.columns:
            if c == "合计":
                continue
            cross[c] = ct[c].astype(str) + " (" + ct_pct[c].round(1).astype(str) + "%)"

        cross = cross.reset_index()
        cross = cross.rename(columns={segment_col: "群组"})
        results[col] = cross
        logger.info("题目 '%s' 交叉分析完成", col)

    return results


def compute_numeric_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()

    valid_cols = [c for c in numeric_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not valid_cols:
        return pd.DataFrame()

    stats = df[valid_cols].describe().T
    stats = stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    stats.columns = ["有效数", "均值", "标准差", "最小值", "25%分位", "中位数", "75%分位", "最大值"]
    stats = stats.round(2)
    stats.index.name = "题目"
    stats = stats.reset_index()
    return stats
