import os
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .column_normalizer import normalize_column_name, normalize_columns, NormalizationResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    code: str
    message: str
    severity: str = "error"
    suggestion: str = ""

    def __str__(self):
        return f"[{self.severity.upper()}] {self.message}" + (f"（建议：{self.suggestion}）" if self.suggestion else "")


@dataclass
class ValidationResult:
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    detected_encoding: str = "utf-8"
    column_normalization: Optional[NormalizationResult] = None

    def __bool__(self):
        return self.valid


class SchemaValidator:
    ENCODINGS_TO_TRY = ["utf-8-sig", "utf-8", "gbk", "gb2312", "gb18030", "latin-1"]

    REQUIRED_COLUMNS = {
        "respondent_id": "受访者唯一标识",
    }

    RECOGNIZED_COLUMN_PATTERNS = {
        "respondent_id": [r"respondent[_-]?id", r"resp[_-]?id", r"受访者id", r"回答者id", r"用户id", r"user[_-]?id"],
        "duration_seconds": [r"duration", r"time[_-]?spent", r"作答时长", r"答题时间", r"用时", r"时长"],
        "age": [r"\bage\b", r"年龄"],
        "gender": [r"\bgender\b", r"\bsex\b", r"性别"],
        "region": [r"\bregion\b", r"\bcity\b", r"\bprovince\b", r"地区", r"城市", r"省份"],
        "education": [r"\beducation\b", r"\bedu\b", r"学历", r"教育程度"],
        "income": [r"\bincome\b", r"\bsalary\b", r"收入", r"薪资"],
        "occupation": [r"\boccupation\b", r"\bjob\b", r"职业", r"工作"],
    }

    QUESTION_PREFIX_PATTERNS = [r"^Q\d+", r"^q\d+", r"^\d+\.", r"^第\d+题"]

    @classmethod
    def full_validation(
        cls,
        filepath: str,
        expected_columns: Optional[list[str]] = None,
        encoding: str = "utf-8",
        min_rows: int = 10,
    ) -> ValidationResult:
        result = ValidationResult(valid=True)

        file_errors = cls._check_file_exists(filepath)
        if file_errors:
            result.errors.extend(file_errors)
            result.valid = False
            return result

        encoding_result = cls._detect_encoding(filepath, encoding)
        result.detected_encoding = encoding_result.detected_encoding
        result.errors.extend(encoding_result.errors)
        result.warnings.extend(encoding_result.warnings)

        if not encoding_result.valid:
            result.valid = False
            return result

        import pandas as pd

        try:
            df = pd.read_csv(filepath, encoding=result.detected_encoding, nrows=min_rows)
        except Exception as e:
            result.errors.append(ValidationError(
                code="PARSE_ERROR",
                message=f"无法解析 CSV 文件: {str(e)}",
                suggestion="请检查文件是否损坏，或尝试使用其他编码格式",
            ))
            result.valid = False
            return result

        row_issues = cls._check_min_rows(filepath, result.detected_encoding, min_rows)
        for issue in row_issues:
            if issue.severity == "error":
                result.errors.append(issue)
            else:
                result.warnings.append(issue)

        column_errors, column_warnings, norm_result = cls._validate_columns(df, expected_columns)
        result.errors.extend(column_errors)
        result.warnings.extend(column_warnings)
        result.column_normalization = norm_result

        if norm_result.renamed_columns:
            for original, canonical in norm_result.renamed_columns:
                result.warnings.append(ValidationError(
                    code="COLUMN_NORMALIZED",
                    message=f"列名归一化: 「{original}」→「{canonical}」",
                    suggestion="建议将数据源列名改为标准名称以避免歧义",
                    severity="warning",
                ))

        if norm_result.ambiguous:
            for canonical, originals in norm_result.ambiguous:
                if len(originals) > 1:
                    result.warnings.append(ValidationError(
                        code="AMBIGUOUS_NORMALIZATION",
                        message=f"多个列名归一化为同一标准名「{canonical}」: {', '.join('「' + o + '」' for o in originals)}",
                        suggestion="将合并这些列的数据，建议修改列名以避免混淆",
                        severity="warning",
                    ))

        result.valid = len(result.errors) == 0
        return result

    @classmethod
    def _check_file_exists(cls, filepath: str) -> list[ValidationError]:
        errors = []
        if not os.path.exists(filepath):
            errors.append(ValidationError(
                code="FILE_NOT_FOUND",
                message=f"文件不存在: {filepath}",
                suggestion="请检查文件路径是否正确",
            ))
            return errors

        if not os.path.isfile(filepath):
            errors.append(ValidationError(
                code="NOT_A_FILE",
                message=f"路径不是文件: {filepath}",
                suggestion="请提供正确的文件路径",
            ))
            return errors

        ext = os.path.splitext(filepath)[1].lower()
        if ext not in (".csv", ".xlsx", ".xls"):
            errors.append(ValidationError(
                code="UNSUPPORTED_FORMAT",
                message=f"不支持的文件格式: {ext}",
                suggestion="请使用 .csv、.xlsx 或 .xls 格式的文件",
            ))

        return errors

    @classmethod
    def _detect_encoding(cls, filepath: str, requested_encoding: str) -> ValidationResult:
        result = ValidationResult(valid=True)

        if os.path.splitext(filepath)[1].lower() in (".xlsx", ".xls"):
            result.detected_encoding = "utf-8"
            return result

        encodings_to_check = []
        if requested_encoding and requested_encoding.lower() != "auto":
            encodings_to_check = [requested_encoding] + cls.ENCODINGS_TO_TRY
        else:
            encodings_to_check = cls.ENCODINGS_TO_TRY

        for encoding in encodings_to_check:
            try:
                with open(filepath, "r", encoding=encoding) as f:
                    f.readline()
                    f.read()
                result.detected_encoding = encoding
                if encoding != requested_encoding and requested_encoding and requested_encoding.lower() != "auto":
                    result.warnings.append(ValidationError(
                        code="ENCODING_FALLBACK",
                        message=f"指定编码 {requested_encoding} 读取失败，已自动切换为 {encoding}",
                        suggestion="如果频繁出现此问题，请确认文件实际编码格式",
                    ))
                return result
            except UnicodeDecodeError:
                continue
            except Exception:
                continue

        result.valid = False
        result.errors.append(ValidationError(
            code="ENCODING_ERROR",
            message=f"无法识别文件编码，尝试了: {', '.join(encodings_to_check)}",
            suggestion="请将文件另存为 UTF-8 格式，或使用 --encoding 参数手动指定编码",
        ))
        return result

    @classmethod
    def _check_min_rows(cls, filepath: str, encoding: str, min_rows: int) -> list[ValidationError]:
        errors = []
        ext = os.path.splitext(filepath)[1].lower()

        import pandas as pd

        if ext == ".csv":
            df = pd.read_csv(filepath, encoding=encoding)
        else:
            df = pd.read_excel(filepath)

        if len(df) < min_rows:
            errors.append(ValidationError(
                code="INSUFFICIENT_ROWS",
                message=f"数据行数过少: {len(df)} 行（建议至少 {min_rows} 行）",
                suggestion="数据量太小可能导致统计结果不稳定，请确认数据是否完整",
                severity="warning",
            ))

        return errors

    @classmethod
    def _validate_columns(
        cls,
        df: "pd.DataFrame",
        expected_columns: Optional[list[str]] = None,
    ) -> tuple[list[ValidationError], list[ValidationError], NormalizationResult]:
        errors = []
        warnings = []
        actual_columns = list(df.columns)

        norm_result = normalize_columns(actual_columns)
        normalized_to_original = {}
        for original, canonical in norm_result.mapping.items():
            if canonical not in normalized_to_original:
                normalized_to_original[canonical] = original

        for required_col, description in cls.REQUIRED_COLUMNS.items():
            matches = cls._find_matching_columns(required_col, actual_columns)
            if not matches:
                matches = cls._find_matching_columns_normalized(required_col, norm_result)
            if not matches:
                similar = cls._find_similar_columns(required_col, actual_columns)
                similar_norm = cls._find_similar_normalized(required_col, norm_result)
                all_similar = list(dict.fromkeys(similar + similar_norm))
                suggestion = f"请添加 '{required_col}' 列（{description}）"
                if all_similar:
                    suggestion += f"，是否是列名拼写错误？相似列名: {', '.join('「' + s + '」' for s in all_similar)}"
                errors.append(ValidationError(
                    code="MISSING_REQUIRED_COLUMN",
                    message=f"缺少必填列: '{required_col}'（{description}）",
                    suggestion=suggestion,
                ))

        if expected_columns:
            for exp_col in expected_columns:
                if exp_col not in actual_columns and normalize_column_name(exp_col) not in norm_result.mapping.get(exp_col, exp_col):
                    norm_match = [o for o, c in norm_result.mapping.items() if c == normalize_column_name(exp_col)]
                    if not norm_match:
                        similar = cls._find_similar_columns(exp_col, actual_columns)
                        suggestion = "请确认列名是否正确"
                        if similar:
                            suggestion += f"，相似列名: {', '.join('「' + s + '」' for s in similar)}"
                        errors.append(ValidationError(
                            code="MISSING_EXPECTED_COLUMN",
                            message=f"缺少期望的列: '{exp_col}'",
                            suggestion=suggestion,
                        ))

        recognized_cols = set()
        for std_name, patterns in cls.RECOGNIZED_COLUMN_PATTERNS.items():
            for actual_col in actual_columns:
                if cls._column_matches_patterns(actual_col, patterns):
                    recognized_cols.add(actual_col)

        for original, canonical in norm_result.mapping.items():
            if canonical in cls.RECOGNIZED_COLUMN_PATTERNS or canonical in {"respondent_id", "duration_seconds", "start_time", "end_time", "submit_time"}:
                recognized_cols.add(original)

        question_cols = [
            col for col in actual_columns
            if any(re.match(p, col) for p in cls.QUESTION_PREFIX_PATTERNS)
            or any(re.match(p, normalize_column_name(col)) for p in cls.QUESTION_PREFIX_PATTERNS)
        ]
        recognized_cols.update(question_cols)

        unrecognized = [col for col in actual_columns if col not in recognized_cols]
        if unrecognized:
            warnings.append(ValidationError(
                code="UNRECOGNIZED_COLUMNS",
                message=f"未识别用途的列: {', '.join('「' + c + '」' for c in unrecognized)}",
                suggestion="这些列将自动作为问卷题目处理，如果是人口统计学属性，请重命名为 age、gender 等标准名称",
                severity="warning",
            ))

        empty_cols = [col for col in actual_columns if df[col].isna().all()]
        if empty_cols:
            warnings.append(ValidationError(
                code="EMPTY_COLUMNS",
                message=f"完全为空的列: {', '.join(empty_cols)}",
                suggestion="这些列将被保留但不会产生有效统计，建议检查数据源",
                severity="warning",
            ))

        duplicate_cols = []
        seen = set()
        for col in actual_columns:
            lower = col.lower().replace(" ", "").replace("-", "").replace("_", "")
            if lower in seen:
                duplicate_cols.append(col)
            seen.add(lower)
        if duplicate_cols:
            errors.append(ValidationError(
                code="DUPLICATE_COLUMNS",
                message=f"存在重复列名（不区分大小写/空格/下划线）: {', '.join(duplicate_cols)}",
                suggestion="请重命名重复的列",
            ))

        return errors, warnings, norm_result

    @classmethod
    def _find_matching_columns(cls, std_name: str, actual_columns: list[str]) -> list[str]:
        patterns = cls.RECOGNIZED_COLUMN_PATTERNS.get(std_name, [])
        return [col for col in actual_columns if cls._column_matches_patterns(col, patterns)]

    @classmethod
    def _find_matching_columns_normalized(cls, std_name: str, norm_result: NormalizationResult) -> list[str]:
        canonical = normalize_column_name(std_name)
        return [original for original, c in norm_result.mapping.items() if c == canonical]

    @classmethod
    def _column_matches_patterns(cls, col_name: str, patterns: list[str]) -> bool:
        col_lower = col_name.lower().replace(" ", "").replace("-", "_")
        for pattern in patterns:
            if re.search(pattern, col_lower, re.IGNORECASE):
                return True
        return False

    @classmethod
    def _find_similar_columns(cls, target: str, candidates: list[str], max_distance: int = 3) -> list[str]:
        similar = []
        target_norm = normalize_column_name(target)
        target_lower = target.lower().replace(" ", "").replace("-", "").replace("_", "")

        for candidate in candidates:
            cand_norm = normalize_column_name(candidate)
            if target_norm == cand_norm and target != candidate:
                similar.append(candidate)
                continue

            cand_lower = candidate.lower().replace(" ", "").replace("-", "").replace("_", "")
            distance = cls._levenshtein_distance(target_lower, cand_lower)
            if distance <= max_distance and distance > 0:
                similar.append(candidate)

        for candidate in candidates:
            cand_lower = candidate.lower().replace(" ", "").replace("-", "").replace("_", "")
            if target_lower in cand_lower or cand_lower in target_lower:
                if candidate not in similar:
                    similar.append(candidate)

        return similar[:5]

    @classmethod
    def _find_similar_normalized(cls, target: str, norm_result: NormalizationResult) -> list[str]:
        target_norm = normalize_column_name(target)
        return [original for original, canonical in norm_result.mapping.items()
                if canonical == target_norm and original != target]

    @staticmethod
    def _levenshtein_distance(a: str, b: str) -> int:
        if len(a) < len(b):
            return SchemaValidator._levenshtein_distance(b, a)
        if len(b) == 0:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
            prev = curr
        return prev[-1]
