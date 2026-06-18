import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

logger = logging.getLogger(__name__)

try:
    import tomllib
except ImportError:
    tomllib = None


@dataclass
class AnomalyThresholds:
    speed_threshold_seconds: float = 60.0
    straight_line_min_cols: int = 5
    straight_line_ratio: float = 0.8
    pattern_min_cols: int = 4
    numeric_outlier_iqr_multiplier: float = 3.0
    numeric_outlier_min_rows: int = 10
    contradiction_pairs: list = field(default_factory=list)


@dataclass
class CleaningConfig:
    trim_whitespace: bool = True
    lowercase_strings: bool = True
    standardize_yes_no: bool = True
    drop_empty_rows: bool = True
    empty_threshold: float = 0.7
    fill_na_numeric: float = -1.0


@dataclass
class SegmentConfig:
    auto_segment: bool = True
    segment_cols: list[str] = field(default_factory=list)
    max_groups: int = 15
    min_categories: int = 2
    max_categories: int = 8


@dataclass
class ReportConfig:
    include_cross_tabs: bool = True
    include_numeric_summary: bool = True
    include_anomalies: bool = True
    include_raw_data: bool = True


@dataclass
class AppConfig:
    anomaly: AnomalyThresholds = field(default_factory=AnomalyThresholds)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    survey_type: str = "default"
    id_col: str = "respondent_id"
    time_col: str = "duration_seconds"
    encoding: str = "utf-8"


PRESET_CONFIGS = {
    "default": AppConfig(),

    "likert7": AppConfig(
        survey_type="likert7",
        anomaly=AnomalyThresholds(
            straight_line_min_cols=3,
            straight_line_ratio=0.85,
            pattern_min_cols=5,
            speed_threshold_seconds=90.0,
        ),
        cleaning=CleaningConfig(
            standardize_yes_no=False,
        ),
    ),

    "multiple_choice": AppConfig(
        survey_type="multiple_choice",
        anomaly=AnomalyThresholds(
            straight_line_min_cols=6,
            straight_line_ratio=0.75,
            speed_threshold_seconds=45.0,
        ),
    ),

    "satisfaction": AppConfig(
        survey_type="satisfaction",
        anomaly=AnomalyThresholds(
            straight_line_min_cols=4,
            straight_line_ratio=0.8,
            speed_threshold_seconds=60.0,
        ),
    ),

    "market_research": AppConfig(
        survey_type="market_research",
        anomaly=AnomalyThresholds(
            straight_line_min_cols=6,
            straight_line_ratio=0.7,
            speed_threshold_seconds=120.0,
        ),
        segment=SegmentConfig(
            max_groups=20,
        ),
    ),

    "psychological": AppConfig(
        survey_type="psychological",
        anomaly=AnomalyThresholds(
            straight_line_min_cols=8,
            straight_line_ratio=0.9,
            pattern_min_cols=6,
            speed_threshold_seconds=180.0,
            numeric_outlier_iqr_multiplier=2.5,
        ),
    ),
}


def load_config(
    config_file: Optional[str] = None,
    survey_type: Optional[str] = None,
    cli_overrides: Optional[dict] = None,
) -> AppConfig:
    config = PRESET_CONFIGS.get("default", AppConfig())

    if survey_type and survey_type in PRESET_CONFIGS:
        config = PRESET_CONFIGS[survey_type]
        logger.info("已加载问卷类型预设: %s", survey_type)
    elif survey_type:
        logger.warning("未找到问卷类型预设 '%s'，使用默认配置。可用类型: %s",
                      survey_type, ", ".join(PRESET_CONFIGS.keys()))

    if config_file:
        file_config = _load_from_file(config_file)
        if file_config:
            config = _merge_configs(config, file_config)
            logger.info("已加载配置文件: %s", config_file)

    if cli_overrides:
        config = _apply_cli_overrides(config, cli_overrides)
        logger.debug("已应用命令行参数覆盖")

    return config


def _load_from_file(filepath: str) -> Optional[dict]:
    if not os.path.exists(filepath):
        logger.warning("配置文件不存在: %s", filepath)
        return None

    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext in (".toml", ".tml"):
            return _load_toml(filepath)
        elif ext == ".json":
            return _load_json(filepath)
        else:
            logger.error("不支持的配置文件格式: %s", ext)
            return None
    except Exception as e:
        logger.error("加载配置文件失败: %s", e)
        return None


def _load_toml(filepath: str) -> dict:
    if tomllib is None:
        raise ImportError("需要 Python 3.11+ 才能解析 TOML 配置文件，请改用 JSON 格式或升级 Python")

    with open(filepath, "rb") as f:
        return tomllib.load(f)


def _load_json(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _merge_configs(base: AppConfig, override: dict) -> AppConfig:
    result = AppConfig()

    result.anomaly = _merge_dataclass(base.anomaly, override.get("anomaly", {}))
    result.cleaning = _merge_dataclass(base.cleaning, override.get("cleaning", {}))
    result.segment = _merge_dataclass(base.segment, override.get("segment", {}))
    result.report = _merge_dataclass(base.report, override.get("report", {}))

    if "survey_type" in override:
        result.survey_type = override["survey_type"]
    else:
        result.survey_type = base.survey_type

    if "id_col" in override:
        result.id_col = override["id_col"]
    else:
        result.id_col = base.id_col

    if "time_col" in override:
        result.time_col = override["time_col"]
    else:
        result.time_col = base.time_col

    if "encoding" in override:
        result.encoding = override["encoding"]
    else:
        result.encoding = base.encoding

    return result


def _merge_dataclass(base_obj, override_dict: dict):
    result = type(base_obj)()
    for field_name in base_obj.__dataclass_fields__:
        base_value = getattr(base_obj, field_name)
        override_value = override_dict.get(field_name, base_value)
        setattr(result, field_name, override_value)
    return result


def _apply_cli_overrides(config: AppConfig, overrides: dict) -> AppConfig:
    result = AppConfig()

    import copy
    result.anomaly = copy.deepcopy(config.anomaly)
    result.cleaning = copy.deepcopy(config.cleaning)
    result.segment = copy.deepcopy(config.segment)
    result.report = copy.deepcopy(config.report)
    result.survey_type = config.survey_type
    result.id_col = config.id_col
    result.time_col = config.time_col
    result.encoding = config.encoding

    anomaly_fields = set(f.name for f in result.anomaly.__dataclass_fields__.values())
    cleaning_fields = set(f.name for f in result.cleaning.__dataclass_fields__.values())
    segment_fields = set(f.name for f in result.segment.__dataclass_fields__.values())
    report_fields = set(f.name for f in result.report.__dataclass_fields__.values())

    for key, value in overrides.items():
        if value is None:
            continue

        if key == "survey_type":
            continue

        if key in ("id_col", "time_col", "encoding"):
            setattr(result, key, value)
        elif key in anomaly_fields:
            setattr(result.anomaly, key, value)
        elif key in cleaning_fields:
            setattr(result.cleaning, key, value)
        elif key in segment_fields:
            setattr(result.segment, key, value)
        elif key in report_fields:
            setattr(result.report, key, value)
        elif key == "segment_cols" and value:
            result.segment.segment_cols = value
            result.segment.auto_segment = False
        elif key == "no_auto_clean" and value:
            result.cleaning.drop_empty_rows = False

    return result


def get_available_survey_types() -> list[str]:
    return list(PRESET_CONFIGS.keys())


def save_config_to_file(config: AppConfig, filepath: str) -> bool:
    ext = os.path.splitext(filepath)[1].lower()

    try:
        config_dict = {
            "survey_type": config.survey_type,
            "id_col": config.id_col,
            "time_col": config.time_col,
            "encoding": config.encoding,
            "anomaly": asdict(config.anomaly),
            "cleaning": asdict(config.cleaning),
            "segment": asdict(config.segment),
            "report": asdict(config.report),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            if ext == ".json":
                json.dump(config_dict, f, ensure_ascii=False, indent=2)
            elif ext in (".toml", ".tml"):
                f.write(_dict_to_toml(config_dict))
            else:
                logger.error("不支持的配置文件格式: %s", ext)
                return False

        logger.info("配置已保存到: %s", filepath)
        return True
    except Exception as e:
        logger.error("保存配置文件失败: %s", e)
        return False


def _dict_to_toml(data: dict, indent: int = 0) -> str:
    lines = []
    prefix = "  " * indent

    for key, value in data.items():
        if isinstance(value, dict):
            if indent == 0:
                lines.append(f"\n[{key}]")
            else:
                parent = ".".join([""] * indent)
                lines.append(f"\n[{parent}{key}]")
            lines.append(_dict_to_toml(value, indent + 1))
        elif isinstance(value, list):
            items = []
            for item in value:
                if isinstance(item, str):
                    items.append(f'"{item}"')
                elif isinstance(item, bool):
                    items.append("true" if item else "false")
                else:
                    items.append(str(item))
            lines.append(f"{prefix}{key} = [{', '.join(items)}]")
        elif isinstance(value, str):
            lines.append(f'{prefix}{key} = "{value}"')
        elif isinstance(value, bool):
            lines.append(f"{prefix}{key} = {'true' if value else 'false'}")
        else:
            lines.append(f"{prefix}{key} = {value}")

    return "\n".join(lines)
