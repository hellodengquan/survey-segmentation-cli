import pandas as pd
import logging
from collections import Counter

from .config import SegmentConfig

logger = logging.getLogger(__name__)


def segment_by_columns(df: pd.DataFrame, segment_cols: list[str]) -> pd.DataFrame:
    if not segment_cols:
        logger.warning("未指定分群列，将整体作为一个群组")
        df = df.copy()
        df["_群组"] = "全部受访者"
        return df

    missing = [c for c in segment_cols if c not in df.columns]
    if missing:
        raise ValueError(f"分群列不存在于数据中: {missing}")

    df = df.copy()

    def _build_label(row):
        parts = []
        for col in segment_cols:
            parts.append(f"{col}={row[col]}")
        return " | ".join(parts)

    df["_群组"] = df.apply(_build_label, axis=1)
    logger.info("按 %s 分群，共产生 %d 个群组", segment_cols, df["_群组"].nunique())
    return df


def auto_segment(df: pd.DataFrame, col_types: dict, config: SegmentConfig = None) -> pd.DataFrame:
    if config is None:
        config = SegmentConfig()

    demo_cols = col_types.get("demographic", [])
    if not demo_cols:
        logger.info("未识别到人口统计学列，将整体作为一个群组")
        df = df.copy()
        df["_群组"] = "全部受访者"
        return df

    candidates = []
    for col in demo_cols:
        nunique = df[col].nunique()
        if config.min_categories <= nunique <= config.max_categories:
            candidates.append((col, nunique))

    candidates.sort(key=lambda x: x[1])

    selected = []
    estimated_groups = 1
    for col, nunique in candidates:
        if estimated_groups * nunique > config.max_groups:
            break
        selected.append(col)
        estimated_groups *= nunique
        if len(selected) >= 2:
            break

    if not selected:
        logger.info("人口统计学列均不适合分群（类别过多或过少），将整体作为一个群组")
        df = df.copy()
        df["_群组"] = "全部受访者"
        return df

    return segment_by_columns(df, selected)


def segment_chunk(chunk: pd.DataFrame, segment_cols: list[str]) -> pd.DataFrame:
    if not segment_cols:
        chunk = chunk.copy()
        chunk["_群组"] = "全部受访者"
        return chunk
    return segment_by_columns(chunk, segment_cols)


def get_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "_群组" not in df.columns:
        raise ValueError("数据尚未分群，请先调用 segment_by_columns 或 auto_segment")

    summary = (
        df.groupby("_群组")
        .size()
        .reset_index(name="人数")
    )
    total = summary["人数"].sum()
    summary["占比"] = (summary["人数"] / total * 100).round(1).astype(str) + "%"
    summary = summary.sort_values("人数", ascending=False).reset_index(drop=True)
    logger.info("分群汇总:\n%s", summary.to_string(index=False))
    return summary


class IncrementalSegmentCounter:
    def __init__(self):
        self._counter: Counter = Counter()
        self._total: int = 0

    def update(self, chunk: pd.DataFrame, segment_col: str = "_群组"):
        if segment_col not in chunk.columns:
            self._counter["全部受访者"] += len(chunk)
            self._total += len(chunk)
            return

        counts = chunk[segment_col].value_counts()
        for group, count in counts.items():
            self._counter[str(group)] += count
        self._total += len(chunk)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._counter:
            return pd.DataFrame(columns=["_群组", "人数", "占比"])

        rows = []
        for group, count in self._counter.most_common():
            rows.append({
                "_群组": group,
                "人数": count,
                "占比": f"{round(count / self._total * 100, 1)}%",
            })

        return pd.DataFrame(rows)
