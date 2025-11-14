import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

DATA_DIR = Path('data')
MONTHS = {
    'jan': 1,
    'january': 1,
    'feb': 2,
    'february': 2,
    'mar': 3,
    'march': 3,
    'apr': 4,
    'april': 4,
    'may': 5,
    'jun': 6,
    'june': 6,
    'jul': 7,
    'july': 7,
    'aug': 8,
    'august': 8,
    'sep': 9,
    'sept': 9,
    'september': 9,
    'oct': 10,
    'october': 10,
    'nov': 11,
    'november': 11,
    'dec': 12,
    'december': 12,
}


def clean_text(value: Optional[str]) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text == '-':
        return None
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


DEPARTMENT_ALIASES: dict[str, str] = {
    'meta': 'คลินิก Meta',
    'meta clinic': 'คลินิก Meta',
    'คลินิก meta': 'คลินิก Meta',
    'เบาหวาน': 'คลินิก Meta',
    'med': 'OPD Med',
    'opd med': 'OPD Med',
    'burn u.': 'Burn Unit',
    'burn u': 'Burn Unit',
    'burn unit': 'Burn Unit',
    '22a': '22A (Stroke Unit)',
    '22 stroke': '22A (Stroke Unit)',
    'stroke': '22A (Stroke Unit)',
    'metha plus': 'Meta Plus',
    'meta plus': 'Meta Plus',
    'เคมีบำบัด และโรคเลือด': 'เคมีบำบัด',
    'เคมีบำบัดและโรคเลือด': 'เคมีบำบัด',
}


def normalize_department_display(name: Optional[str]) -> Optional[str]:
    if not name:
        return name
    normalized = name
    for character, replacement in (
        ('\ufeff', ''),
        ('\u200b', ''),
        ('\u200c', ''),
        ('\u200d', ''),
        ('\u202f', ' '),
        ('\xa0', ' '),
    ):
        normalized = normalized.replace(character, replacement)
    normalized = re.sub(r'\s+', ' ', normalized, flags=re.UNICODE).strip()
    if not normalized:
        return None
    alias = DEPARTMENT_ALIASES.get(normalized.casefold())
    if alias:
        return alias
    return normalized


def department_group_key(name: Optional[str]) -> Tuple[str, str]:
    display = normalize_department_display(name) or ''
    key = display.casefold()
    return key, display


def parse_month_year(path: Path) -> tuple[int, int, int]:
    """Return (year_be, year_ad, month)."""
    name = path.stem.lower()
    month = None
    for key, val in MONTHS.items():
        if re.search(rf'\b{key}\b', name):
            month = val
            break
    if month is None:
        raise ValueError(f'Could not determine month from {path}')

    year_two_digit = None
    match = re.search(r'(\d{2})', name)
    if match:
        year_two_digit = int(match.group(1))

    if year_two_digit is None:
        # Try parent directory e.g. "4-68"
        match = re.search(r'(\d{2})', path.parent.name)
        if match:
            year_two_digit = int(match.group(1))

    if year_two_digit is None:
        raise ValueError(f'Could not determine year from {path}')

    year_be = 2500 + year_two_digit
    year_ad = year_be - 543
    return year_be, year_ad, month


@dataclass
class DepartmentScore:
    department: str
    average_score: float
    survey_count: Optional[int]
    file_path: str
    month: int
    year_be: int
    year_ad: int
    recommendation_percent: Optional[float] = None


@dataclass
class Aggregate:
    department: str
    average_score: float
    survey_count: int
    month_count: int
    recommendation_percent: Optional[float]


def parse_excel(path: Path) -> List[DepartmentScore]:
    year_be, year_ad, month = parse_month_year(path)

    try:
        df = pd.read_excel(path, sheet_name='Total Table', header=None)
    except ValueError:
        df = pd.read_excel(path, header=None)

    headers: Optional[List[Optional[str]]] = None
    counts_row: Optional[pd.Series] = None
    pending_scores: dict[int, DepartmentScore] = {}
    results: List[DepartmentScore] = []

    for _, row in df.iterrows():
        first_cell = clean_text(row.iloc[0])
        if first_cell == 'หัวข้อ':
            headers = [clean_text(val) for val in row.tolist()]
            counts_row = None
        elif headers and first_cell == 'จำนวนใบประเมิน':
            counts_row = row
        elif headers and first_cell == 'ระดับคะแนนเฉลี่ย':
            pending_scores.clear()
            for col_idx, dept in enumerate(headers[1:], start=1):
                dept_name = clean_text(dept)
                if not dept_name:
                    continue
                dept_name = normalize_department_display(dept_name)
                if not dept_name:
                    continue
                val = row.iloc[col_idx]
                if pd.isna(val):
                    continue
                if isinstance(val, str) and val.strip() == '-':
                    continue
                try:
                    score = float(val)
                except (TypeError, ValueError):
                    continue
                survey_count: Optional[int] = None
                if counts_row is not None:
                    count_val = counts_row.iloc[col_idx]
                    if pd.notna(count_val) and not (isinstance(count_val, str) and count_val.strip() == '-'):
                        try:
                            survey_count = int(float(count_val))
                        except (TypeError, ValueError):
                            survey_count = None
                score_entry = DepartmentScore(
                    department=dept_name,
                    average_score=score,
                    survey_count=survey_count,
                    file_path=str(path.relative_to(DATA_DIR)),
                    month=month,
                    year_be=year_be,
                    year_ad=year_ad,
                )
                results.append(score_entry)
                pending_scores[col_idx] = score_entry
        elif (
            headers
            and first_cell
            and first_cell.startswith('การแนะนำญาติหรือคนรู้จักให้มาใช้บริการที่โรงพยาบาล')
        ):
            for col_idx, _ in enumerate(headers[1:], start=1):
                score_entry = pending_scores.get(col_idx)
                if not score_entry:
                    continue
                val = row.iloc[col_idx]
                if pd.isna(val) or (isinstance(val, str) and val.strip() == '-'):
                    continue
                try:
                    recommendation = float(val) / 100.0
                except (TypeError, ValueError):
                    continue
                score_entry.recommendation_percent = recommendation
            headers = None
            counts_row = None
            pending_scores.clear()
    return results


def load_all_scores(data_dir: Path) -> List[DepartmentScore]:
    scores: List[DepartmentScore] = []
    for path in sorted(data_dir.rglob('*.xlsx')):
        scores.extend(parse_excel(path))
    return scores


def aggregate_scores(scores: Iterable[DepartmentScore], *, group_key) -> List[Aggregate]:
    buckets: dict[str, dict[str, float]] = {}
    meta: dict[str, dict[str, int]] = {}
    labels: dict[str, str] = {}

    for score in scores:
        key, label = group_key(score)
        if key not in labels and label:
            labels[key] = label
        if key not in buckets:
            buckets[key] = {
                'weighted': 0.0,
                'weight': 0.0,
                'simple_total': 0.0,
                'count': 0,
                'months': set(),
                'rec_weighted': 0.0,
                'rec_weight': 0.0,
                'rec_simple_total': 0.0,
                'rec_simple_count': 0,
            }
            meta[key] = {'survey_count': 0}
        bucket = buckets[key]
        weight = score.survey_count if score.survey_count and score.survey_count > 0 else None
        if weight:
            bucket['weighted'] += score.average_score * weight
            bucket['weight'] += weight
            meta[key]['survey_count'] += weight
            if score.recommendation_percent is not None:
                bucket['rec_weighted'] += score.recommendation_percent * weight
                bucket['rec_weight'] += weight
        else:
            bucket['simple_total'] += score.average_score
            bucket['count'] += 1
            if score.recommendation_percent is not None:
                bucket['rec_simple_total'] += score.recommendation_percent
                bucket['rec_simple_count'] += 1
        bucket['months'].add((score.year_ad, score.month))

    aggregates: List[Aggregate] = []
    for key, bucket in buckets.items():
        total_weight = bucket['weight']
        if total_weight > 0:
            avg = bucket['weighted'] / total_weight
            survey_count = meta[key]['survey_count']
            if bucket['rec_weight'] > 0:
                recommendation = bucket['rec_weighted'] / bucket['rec_weight']
            elif bucket['rec_simple_count'] > 0:
                recommendation = bucket['rec_simple_total'] / bucket['rec_simple_count']
            else:
                recommendation = float('nan')
        else:
            count = bucket['count']
            avg = bucket['simple_total'] / count if count else float('nan')
            survey_count = count
            if bucket['rec_simple_count'] > 0:
                recommendation = bucket['rec_simple_total'] / bucket['rec_simple_count']
            else:
                recommendation = float('nan')
        aggregates.append(
            Aggregate(
                department=labels.get(key, key),
                average_score=avg,
                survey_count=survey_count,
                month_count=len(bucket['months']),
                recommendation_percent=recommendation,
            )
        )
    return aggregates


def aggregate_by_department(scores: Iterable[DepartmentScore]) -> List[Aggregate]:
    return aggregate_scores(scores, group_key=lambda s: department_group_key(s.department))


def aggregate_by_year(scores: Iterable[DepartmentScore]) -> dict[int, List[Aggregate]]:
    by_year: dict[int, List[DepartmentScore]] = defaultdict(list)
    for score in scores:
        by_year[score.year_ad].append(score)
    return {year: aggregate_by_department(year_scores) for year, year_scores in by_year.items()}


def to_serializable(scores: List[DepartmentScore], yearly: dict[int, List[Aggregate]], overall: List[Aggregate]) -> dict:
    return {
        'raw_scores': [score.__dict__ for score in scores],
        'yearly': {
            str(year): [agg.__dict__ for agg in aggs]
            for year, aggs in sorted(yearly.items())
        },
        'overall': [agg.__dict__ for agg in overall],
    }


def main() -> None:
    scores = load_all_scores(DATA_DIR)
    yearly = aggregate_by_year(scores)
    overall = aggregate_by_department(scores)

    output = to_serializable(scores, yearly, overall)
    output_path = Path('analysis/results.json')
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print(f'Parsed {len(scores)} department-month scores across {len(yearly)} years.')
    for year, aggs in sorted(yearly.items()):
        print(f'Year {year}: {len(aggs)} departments')
    print(f'Overall unique departments: {len(overall)}')


if __name__ == '__main__':
    main()
