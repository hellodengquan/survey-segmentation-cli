import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


CHINESE_ENGLISH_ALIASES = {
    "受访者id": "respondent_id",
    "受访者_id": "respondent_id",
    "回答者id": "respondent_id",
    "回答者_id": "respondent_id",
    "用户id": "respondent_id",
    "用户_id": "respondent_id",
    "年龄": "age",
    "性别": "gender",
    "地区": "region",
    "城市": "region",
    "省份": "region",
    "学历": "education",
    "教育程度": "education",
    "收入": "income",
    "薪资": "income",
    "薪水": "income",
    "职业": "occupation",
    "工作": "occupation",
    "作答时长": "duration_seconds",
    "答题时间": "duration_seconds",
    "答题时长": "duration_seconds",
    "用时": "duration_seconds",
    "时长": "duration_seconds",
    "答题时长秒": "duration_seconds",
    "作答时长秒": "duration_seconds",
    "开始时间": "start_time",
    "结束时间": "end_time",
    "提交时间": "submit_time",
}

CANONICAL_NAMES = {
    "respondent_id": "respondent_id",
    "resp_id": "respondent_id",
    "respid": "respondent_id",
    "respondentid": "respondent_id",
    "user_id": "respondent_id",
    "userid": "respondent_id",
    "uid": "respondent_id",
    "duration_seconds": "duration_seconds",
    "duration": "duration_seconds",
    "durationsec": "duration_seconds",
    "time_spent": "duration_seconds",
    "timespent": "duration_seconds",
    "elapsed": "duration_seconds",
    "age": "age",
    "gender": "gender",
    "sex": "gender",
    "region": "region",
    "city": "region",
    "province": "region",
    "area": "region",
    "education": "education",
    "edu": "education",
    "income": "income",
    "salary": "income",
    "occupation": "occupation",
    "job": "occupation",
    "start_time": "start_time",
    "starttime": "start_time",
    "end_time": "end_time",
    "endtime": "end_time",
    "submit_time": "submit_time",
    "submittime": "submit_time",
}


@dataclass
class NormalizationResult:
    mapping: dict[str, str] = field(default_factory=dict)
    renamed_columns: list[tuple[str, str]] = field(default_factory=list)
    ambiguous: list[tuple[str, list[str]]] = field(default_factory=list)


def normalize_column_name(col: str) -> str:
    normalized = col.strip()
    normalized = re.sub(r'[\s\u3000]+', '_', normalized)
    normalized = re.sub(r'[-—_]+', '_', normalized)
    normalized = re.sub(r'[()（）\[\]【】]', '', normalized)
    normalized = normalized.strip('_')
    normalized = normalized.lower()

    if normalized in CHINESE_ENGLISH_ALIASES:
        return CHINESE_ENGLISH_ALIASES[normalized]

    parts = normalized.split('_')
    chinese_parts = []
    for part in parts:
        if part in CHINESE_ENGLISH_ALIASES:
            chinese_parts.append(CHINESE_ENGLISH_ALIASES[part])
        else:
            chinese_parts.append(part)

    if chinese_parts != parts:
        rejoined = '_'.join(chinese_parts)
        if rejoined in CANONICAL_NAMES:
            return CANONICAL_NAMES[rejoined]
        return rejoined

    if normalized in CANONICAL_NAMES:
        return CANONICAL_NAMES[normalized]

    return normalized


def normalize_columns(columns: list[str]) -> NormalizationResult:
    result = NormalizationResult()

    canonical_to_originals: dict[str, list[str]] = {}
    for col in columns:
        canonical = normalize_column_name(col)
        result.mapping[col] = canonical
        if canonical not in canonical_to_originals:
            canonical_to_originals[canonical] = []
        canonical_to_originals[canonical].append(col)

    for original, canonical in result.mapping.items():
        if original != canonical:
            result.renamed_columns.append((original, canonical))

    for canonical, originals in canonical_to_originals.items():
        if len(originals) > 1:
            result.ambiguous.append((canonical, originals))

    if result.renamed_columns:
        logger.info("列名归一化: %s",
                    ", ".join(f"「{o}」→「{c}」" for o, c in result.renamed_columns))

    if result.ambiguous:
        for canonical, originals in result.ambiguous:
            logger.warning("多个列名归一化为同一标准名「%s」: %s",
                          canonical, ", ".join(f"「{o}」" for o in originals))

    return result


def apply_column_normalization(df: "pd.DataFrame", norm_result: NormalizationResult) -> "pd.DataFrame":
    if not norm_result.renamed_columns and not norm_result.ambiguous:
        return df

    df = df.copy()

    for canonical, originals in norm_result.ambiguous:
        if len(originals) > 1:
            keep_col = originals[0]
            for dup_col in originals[1:]:
                df[keep_col] = df[keep_col].fillna(df[dup_col])
                df = df.drop(columns=[dup_col])
                logger.info("合并同名列: 「%s」合并到「%s」", dup_col, keep_col)

    rename_map = {}
    for original, canonical in norm_result.mapping.items():
        if original in df.columns and original != canonical:
            if canonical in rename_map.values():
                continue
            rename_map[original] = canonical

    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info("已重命名 %d 列", len(rename_map))

    return df


def resolve_col_name(requested: str, available_columns: list[str]) -> str | None:
    requested_norm = normalize_column_name(requested)
    for col in available_columns:
        if normalize_column_name(col) == requested_norm:
            return col
    if requested in available_columns:
        return requested
    return None
