import re
import json
import os
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_BUILTIN_CHINESE_ENGLISH_ALIASES = {
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

_BUILTIN_CANONICAL_NAMES = {
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


class AliasRegistry:
    _instance = None

    def __init__(self):
        self._chinese_english_aliases = dict(_BUILTIN_CHINESE_ENGLISH_ALIASES)
        self._canonical_names = dict(_BUILTIN_CANONICAL_NAMES)

    @classmethod
    def get_instance(cls) -> "AliasRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    @property
    def chinese_english_aliases(self) -> dict[str, str]:
        return self._chinese_english_aliases

    @property
    def canonical_names(self) -> dict[str, str]:
        return self._canonical_names

    def load_from_file(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            logger.warning("别名配置文件不存在: %s，使用内置默认别名", filepath)
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error("别名配置文件 JSON 解析失败: %s", e)
            return False
        except Exception as e:
            logger.error("读取别名配置文件失败: %s", e)
            return False

        custom_aliases = data.get("chinese_english_aliases", {})
        custom_canonical = data.get("canonical_names", {})

        if not isinstance(custom_aliases, dict) or not isinstance(custom_canonical, dict):
            logger.error("别名配置文件格式错误: chinese_english_aliases 和 canonical_names 必须是字典")
            return False

        self._chinese_english_aliases.update(custom_aliases)
        self._canonical_names.update(custom_canonical)

        logger.info("已从 %s 加载别名配置: 中英文别名 %d 条，标准名映射 %d 条",
                    filepath, len(custom_aliases), len(custom_canonical))
        return True

    def reload_from_file(self, filepath: str) -> bool:
        self._chinese_english_aliases = dict(_BUILTIN_CHINESE_ENGLISH_ALIASES)
        self._canonical_names = dict(_BUILTIN_CANONICAL_NAMES)
        return self.load_from_file(filepath)


def load_default_aliases():
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_aliases.json")
    registry = AliasRegistry.get_instance()
    if os.path.exists(default_path):
        registry.load_from_file(default_path)
    return registry


@dataclass
class NormalizationResult:
    mapping: dict[str, str] = field(default_factory=dict)
    renamed_columns: list[tuple[str, str]] = field(default_factory=list)
    ambiguous: list[tuple[str, list[str]]] = field(default_factory=list)


def normalize_column_name(col: str) -> str:
    registry = AliasRegistry.get_instance()

    normalized = col.strip()
    normalized = re.sub(r'[\s\u3000]+', '_', normalized)
    normalized = re.sub(r'[-—_]+', '_', normalized)
    normalized = re.sub(r'[()（）\[\]【】]', '', normalized)
    normalized = normalized.strip('_')
    normalized = normalized.lower()

    if normalized in registry.chinese_english_aliases:
        return registry.chinese_english_aliases[normalized]

    parts = normalized.split('_')
    mapped_parts = []
    for part in parts:
        if part in registry.chinese_english_aliases:
            mapped_parts.append(registry.chinese_english_aliases[part])
        else:
            mapped_parts.append(part)

    if mapped_parts != parts:
        rejoined = '_'.join(mapped_parts)
        if rejoined in registry.canonical_names:
            return registry.canonical_names[rejoined]
        return rejoined

    if normalized in registry.canonical_names:
        return registry.canonical_names[normalized]

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
