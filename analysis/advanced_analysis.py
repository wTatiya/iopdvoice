import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RESULTS_PATH = Path('analysis/results.json')
DOCX_DIR = Path('analysis/docx_text')
OUTPUT_ANALYSIS_PATH = Path('analysis/advanced_insights.json')
OUTPUT_DASHBOARD_PATH = Path('docs/data/advanced_insights.json')

THAI_MONTHS = {
    'มกราคม': 1,
    'กุมภาพันธ์': 2,
    'มีนาคม': 3,
    'เมษายน': 4,
    'พฤษภาคม': 5,
    'มิถุนายน': 6,
    'กรกฎาคม': 7,
    'กรกฏาคม': 7,  # Alternate spelling observed in filenames
    'สิงหาคม': 8,
    'กันยายน': 9,
    'ตุลาคม': 10,
    'พฤศจิกายน': 11,
    'ธันวาคม': 12,
}

POSITIVE_KEYWORDS = [
    'ดี',
    'ยอดเยี่ยม',
    'เยี่ยม',
    'ขอบคุณ',
    'สะอาด',
    'รวดเร็ว',
    'ประทับใจ',
    'น่ารัก',
    'อบอุ่น',
    'บริการดี',
    'พูดจาดี',
    'น่ายกย่อง',
]

NEGATIVE_KEYWORDS = [
    'ช้า',
    'รอ',
    'ล่าช้า',
    'ไม่ดี',
    'ไม่สะดวก',
    'ไม่พอ',
    'แออัด',
    'แก้ไข',
    'ปรับปรุง',
    'เสีย',
    'ขาด',
    'แย่',
    'ร้องเรียน',
]

THEME_KEYWORDS = {
    'wait_time': ['รอ', 'คิว', 'ช้า', 'ล่าช้า', 'นาน', 'ติดขัด'],
    'attitude': ['ยิ้ม', 'ไพเราะ', 'เอาใจใส่', 'พูดจา', 'บริการดี', 'ห่วงใย'],
    'communication': ['อธิบาย', 'แจ้ง', 'ข้อมูล', 'สื่อสาร', 'คำแนะนำ', 'บอก'],
    'process': ['ระบบ', 'ขั้นตอน', 'เอกสาร', 'ลงทะเบียน', 'เวร', 'ประสาน'],
    'facilities': ['ห้องน้ำ', 'ที่นั่ง', 'เก้าอี้', 'แอร์', 'เครื่อง', 'เตียง', 'สถานที่', 'จอดรถ'],
    'staffing': ['เจ้าหน้าที่', 'บุคลากร', 'พนักงาน', 'คน', 'เพิ่มคน', 'ขาดคน'],
}

TIME_OF_DAY_KEYWORDS = {
    'morning': ['เช้า'],
    'afternoon': ['บ่าย'],
    'evening': ['เย็น'],
    'night': ['กลางคืน', 'ดึก', 'เวรดึก'],
}

DEMOGRAPHIC_KEYWORDS = {
    'elderly': ['ผู้สูงอายุ', 'คนแก่', 'ผู้เฒ่า'],
    'children': ['เด็ก', 'ลูก', 'กุมาร'],
    'pregnancy': ['หญิงตั้งครรภ์', 'คนท้อง', 'แม่ตั้งครรภ์', 'คุณแม่'],
    'caregiver': ['ญาติ', 'ผู้ดูแล', 'ผู้ปกครอง'],
    'disability': ['ผู้พิการ', 'คนพิการ', 'วีลแชร์'],
    'foreign': ['ต่างชาติ', 'ต่างประเทศ'],
}

SERVICE_GROUP_RULES: List[tuple[str, List[str]]] = [
    ('Pediatrics', ['PED', 'NICU', 'PICU', 'กุมาร', 'เด็ก']),
    ('Maternity', ['ANC', 'OB', 'สูติ', 'นรี', 'ห้องคลอด', 'LDR']),
    ('Adult Inpatient', ['24A', '25A', '25B', '26A', '26B', 'WARD', 'W', 'IPD']),
    ('Emergency', ['ER', 'ฉุกเฉิน']),
    ('Surgical', ['ศัลยกรรม', 'ORTHO', 'ศัลย', 'ผ่าตัด']),
    ('General Medicine', ['MED', 'เวชปฏิบัติ', 'OPD', 'นอกเวลา', 'Meta']),
    ('Specialty Clinic', ['ผิวหนัง', 'จักษุ', 'หัวใจ', 'ตา', 'ENT', 'HBO', 'ฝังเข็ม']),
]


@dataclass
class Comment:
    period: str
    text: str
    themes: List[str]
    sentiment: str
    time_refs: List[str]
    demographics: List[str]


def load_results() -> pd.DataFrame:
    data = json.loads(RESULTS_PATH.read_text())
    df = pd.DataFrame(data['raw_scores'])
    df['period'] = pd.to_datetime(
        dict(year=df['year_ad'], month=df['month'], day=1),
        errors='coerce',
    )
    df['survey_weight'] = df['survey_count'].apply(
        lambda val: float(val) if pd.notna(val) and val > 0 else np.nan
    )
    return df


def read_comment_files() -> List[Comment]:
    comments: List[Comment] = []
    for path in DOCX_DIR.glob('*.txt'):
        text = path.read_text(encoding='utf-8')
        period = extract_period_from_text(text) or extract_period_from_filename(path.name)
        if not period:
            continue
        for line in extract_comment_lines(text.splitlines()):
            normalized = line.strip()
            if not normalized:
                continue
            themes = detect_categories(normalized, THEME_KEYWORDS)
            sentiment = detect_sentiment(normalized)
            time_refs = detect_categories(normalized, TIME_OF_DAY_KEYWORDS)
            demographics = detect_categories(normalized, DEMOGRAPHIC_KEYWORDS)
            comments.append(
                Comment(
                    period=period,
                    text=normalized,
                    themes=themes,
                    sentiment=sentiment,
                    time_refs=time_refs,
                    demographics=demographics,
                )
            )
    return comments


def extract_comment_lines(lines: Iterable[str]) -> Iterable[str]:
    ignored_prefixes = (
        'สรุปการประเมิน',
        'จำนวนแบบสำรวจ',
        'เกณฑ์การประเมิน',
        'คะแนนเฉลี่ย',
        'คำชม',
        'ข้อเสนอแนะ',
        'หมายเหตุ',
        'ค่าเฉลี่ย',
    )
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped == '-' or stripped.startswith('('):
            continue
        if any(stripped.startswith(prefix) for prefix in ignored_prefixes):
            continue
        yield stripped


def extract_period_from_text(text: str) -> Optional[str]:
    match = re.search(r'เดือน\s*([ก-๙]+)\s*(\d{4})', text)
    if not match:
        return None
    month_label, year_text = match.groups()
    month = THAI_MONTHS.get(month_label)
    if not month:
        return None
    year_be = int(year_text)
    year_ad = year_be - 543
    return f"{year_ad:04d}-{month:02d}"


def extract_period_from_filename(name: str) -> Optional[str]:
    match = re.search(r'เดือน([ก-๙]+)\s*(\d{4})', name)
    if not match:
        return None
    month_label, year_text = match.groups()
    month = THAI_MONTHS.get(month_label)
    if not month:
        return None
    year_be = int(year_text)
    year_ad = year_be - 543
    return f"{year_ad:04d}-{month:02d}"


def detect_categories(text: str, keyword_map: dict[str, List[str]]) -> List[str]:
    lowered = text.lower()
    matches = []
    for key, words in keyword_map.items():
        if any(word in lowered for word in words):
            matches.append(key)
    return matches


def detect_sentiment(text: str) -> str:
    lowered = text.lower()
    positive = any(keyword in lowered for keyword in POSITIVE_KEYWORDS)
    negative = any(keyword in lowered for keyword in NEGATIVE_KEYWORDS)
    if positive and not negative:
        return 'positive'
    if negative and not positive:
        return 'negative'
    if positive and negative:
        return 'mixed'
    return 'neutral'


def assign_service_group(department: str) -> str:
    name = department.upper()
    for label, tokens in SERVICE_GROUP_RULES:
        if any(token.upper() in name for token in tokens):
            return label
    if 'IPD' in name:
        return 'Adult Inpatient'
    if 'OPD' in name:
        return 'General Medicine'
    return 'Other'


def compute_trends(df: pd.DataFrame) -> dict:
    periods = sorted(df['period'].dropna().unique())
    period_labels = [period.strftime('%Y-%m') for period in periods]

    series = []
    for department, group in df.groupby('department'):
        ordered = group.sort_values('period')
        points = []
        for row in ordered.itertuples(index=False):
            if pd.isna(row.period):
                continue
            points.append(
                {
                    'period': row.period.strftime('%Y-%m'),
                    'average_score': round(float(row.average_score), 3),
                    'survey_count': int(row.survey_count) if not pd.isna(row.survey_count) else None,
                }
            )
        if points:
            series.append(
                {
                    'department': department,
                    'mean_score': round(float(ordered['average_score'].mean()), 3),
                    'months_observed': len(points),
                    'series': points,
                }
            )
    return {
        'periods': period_labels,
        'departments': sorted(series, key=lambda item: item['department'].lower()),
    }


def compute_volatility(df: pd.DataFrame) -> dict:
    stats_rows = []
    for department, group in df.groupby('department'):
        ordered = group.sort_values('period')
        scores = ordered['average_score'].astype(float)
        if len(scores.dropna()) < 3:
            continue
        std = float(scores.std(ddof=0))
        mean = float(scores.mean())
        coef = float(std / mean) if mean else float('nan')
        filled_scores = scores.ffill().bfill()
        trend = float(
            np.polyfit(
                np.arange(len(filled_scores)),
                filled_scores.to_numpy(dtype=float),
                1,
            )[0]
        )
        last = float(scores.iloc[-1])
        stats_rows.append(
            {
                'department': department,
                'volatility_index': round(std, 4),
                'coefficient_of_variation': round(coef, 4) if not math.isnan(coef) else None,
                'score_range': round(float(scores.max() - scores.min()), 3),
                'trend_per_month': round(float(trend), 4),
                'last_score': round(last, 3),
                'observations': len(scores),
            }
        )
    ordered = sorted(stats_rows, key=lambda item: item['volatility_index'], reverse=True)
    median = float(np.median([row['volatility_index'] for row in stats_rows])) if stats_rows else float('nan')
    return {
        'summary': {
            'departments_evaluated': len(stats_rows),
            'median_volatility': round(median, 4) if not math.isnan(median) else None,
        },
        'most_unstable': ordered[:15],
        'most_stable': sorted(stats_rows, key=lambda item: item['volatility_index'])[:15],
    }


def compute_weighted_scores(df: pd.DataFrame) -> pd.DataFrame:
    def weighted_average(group: pd.DataFrame) -> pd.Series:
        weights = group['survey_weight']
        scores = group['average_score']
        if weights.notna().any():
            weighted_sum = (scores * weights.fillna(0)).sum()
            total_weight = weights.fillna(0).sum()
            avg = weighted_sum / total_weight if total_weight > 0 else scores.mean()
            weight = total_weight
        else:
            avg = scores.mean()
            weight = len(group)
        return pd.Series(
            {
                'average_score': avg,
                'survey_total': weight,
                'months_observed': group['period'].nunique(),
            }
        )

    aggregated = (
        df.groupby('department', as_index=False)
        .apply(weighted_average, include_groups=False)
        .reset_index(drop=True)
    )
    return aggregated


def identify_high_performance(aggregated: pd.DataFrame, volatility: dict) -> List[dict]:
    volatility_map = {row['department']: row for row in volatility['most_unstable'] + volatility['most_stable']}
    records = []
    for row in aggregated.itertuples(index=False):
        vol = volatility_map.get(row.department)
        vol_index = vol['volatility_index'] if vol else None
        if row.average_score >= 4.6 and row.months_observed >= 6:
            records.append(
                {
                    'department': row.department,
                    'average_score': round(float(row.average_score), 3),
                    'survey_total': int(round(row.survey_total)),
                    'months_observed': int(row.months_observed),
                    'volatility_index': round(float(vol_index), 4) if vol_index is not None else None,
                }
            )
    return sorted(records, key=lambda item: (-item['average_score'], item['volatility_index'] or 0))[:15]


def identify_critical_gaps(aggregated: pd.DataFrame, volatility: dict) -> List[dict]:
    volatility_map = {row['department']: row for row in volatility['most_unstable'] + volatility['most_stable']}
    bottom = aggregated.sort_values('average_score').head(15)
    gaps = []
    for row in bottom.itertuples(index=False):
        vol = volatility_map.get(row.department)
        gaps.append(
            {
                'department': row.department,
                'average_score': round(float(row.average_score), 3),
                'survey_total': int(round(row.survey_total)),
                'volatility_index': round(float(vol['volatility_index']), 4) if vol else None,
                'months_observed': int(row.months_observed),
            }
        )
    return gaps


def summarize_comments(comments: List[Comment]) -> dict:
    themes_counter = Counter()
    sentiment_counter = Counter()
    time_counter = Counter()
    demographic_counter = Counter()
    monthly_theme = defaultdict(lambda: Counter())
    monthly_sentiment = defaultdict(lambda: Counter())

    for comment in comments:
        if comment.sentiment:
            sentiment_counter[comment.sentiment] += 1
            monthly_sentiment[comment.period][comment.sentiment] += 1
        if comment.themes:
            for theme in comment.themes:
                themes_counter[theme] += 1
                monthly_theme[comment.period][theme] += 1
        if comment.time_refs:
            for ref in comment.time_refs:
                time_counter[ref] += 1
        if comment.demographics:
            for key in comment.demographics:
                demographic_counter[key] += 1

    themes_over_time = []
    for period, counts in sorted(monthly_theme.items()):
        total = sum(counts.values())
        themes_over_time.append(
            {
                'period': period,
                'themes': {theme: counts[theme] for theme in THEME_KEYWORDS.keys()},
                'total_mentions': total,
            }
        )

    sentiments_over_time = []
    for period, counts in sorted(monthly_sentiment.items()):
        total = sum(counts.values())
        sentiments_over_time.append(
            {
                'period': period,
                'sentiment_counts': dict(counts),
                'total': total,
            }
        )

    return {
        'total_comments': len(comments),
        'themes': dict(themes_counter),
        'sentiment': dict(sentiment_counter),
        'themes_over_time': themes_over_time,
        'sentiment_over_time': sentiments_over_time,
        'time_of_day': dict(time_counter),
        'demographics': dict(demographic_counter),
    }


def wait_time_vs_satisfaction(df: pd.DataFrame, comments: List[Comment]) -> dict:
    monthly_scores = (
        df.groupby('period')
        .apply(
            lambda group: np.average(
                group['average_score'],
                weights=group['survey_weight'].fillna(1),
            ),
            include_groups=False,
        )
        .dropna()
    )
    wait_counts = defaultdict(lambda: {'mentions': 0, 'total': 0})
    for comment in comments:
        wait_counts[comment.period]['total'] += 1
        if 'wait_time' in comment.themes:
            wait_counts[comment.period]['mentions'] += 1
    records = []
    for period, score in monthly_scores.items():
        label = period.strftime('%Y-%m')
        counts = wait_counts[label]
        total = counts['total']
        share = counts['mentions'] / total if total else 0.0
        records.append(
            {
                'period': label,
                'satisfaction': round(float(score), 3),
                'wait_share': round(float(share), 4),
                'wait_mentions': counts['mentions'],
                'comment_total': total,
            }
        )
    if len(records) >= 2:
        df_records = pd.DataFrame(records)
        pearson_r, p_value = stats.pearsonr(df_records['wait_share'], df_records['satisfaction'])
    else:
        pearson_r, p_value = float('nan'), float('nan')
    return {
        'series': records,
        'pearson_r': round(float(pearson_r), 4) if not math.isnan(pearson_r) else None,
        'p_value': round(float(p_value), 4) if not math.isnan(p_value) else None,
    }


def benchmark_top_bottom(aggregated: pd.DataFrame, volatility: dict) -> dict:
    ordered = aggregated.sort_values('average_score', ascending=False)
    top = ordered.head(10)
    bottom = ordered.tail(10)

    def stats_for(group: pd.DataFrame) -> dict:
        return {
            'mean_score': round(float(group['average_score'].mean()), 3),
            'mean_volatility': round(float(group.get('volatility_index', pd.Series(dtype=float)).mean()), 4)
            if 'volatility_index' in group
            else None,
            'mean_survey_total': round(float(group['survey_total'].mean()), 1),
            'mean_months_observed': round(float(group['months_observed'].mean()), 1),
        }

    volatility_map = {row['department']: row['volatility_index'] for row in volatility['most_unstable'] + volatility['most_stable']}

    top_enriched = top.assign(volatility_index=top['department'].map(volatility_map))
    bottom_enriched = bottom.assign(volatility_index=bottom['department'].map(volatility_map))

    top_stats = stats_for(top_enriched)
    bottom_stats = stats_for(bottom_enriched)
    return {
        'top10': {
            'departments': top_enriched[['department', 'average_score', 'survey_total', 'volatility_index']]
            .round({'average_score': 3})
            .to_dict(orient='records'),
            'stats': top_stats,
        },
        'bottom10': {
            'departments': bottom_enriched[['department', 'average_score', 'survey_total', 'volatility_index']]
            .round({'average_score': 3})
            .to_dict(orient='records'),
            'stats': bottom_stats,
        },
        'delta': {
            'score_gap': round(top_stats['mean_score'] - bottom_stats['mean_score'], 3),
            'survey_gap': round(top_stats['mean_survey_total'] - bottom_stats['mean_survey_total'], 1),
        },
    }


def time_of_day_summary(comment_summary: dict) -> List[dict]:
    totals = comment_summary['time_of_day']
    total_mentions = sum(totals.values()) or 1
    return [
        {
            'time': key,
            'mentions': totals.get(key, 0),
            'share': round(totals.get(key, 0) / total_mentions, 4),
        }
        for key in TIME_OF_DAY_KEYWORDS.keys()
    ]


def demographic_disparity(df: pd.DataFrame, comment_summary: dict) -> dict:
    aggregated = compute_weighted_scores(df)
    aggregated['service_group'] = aggregated['department'].apply(assign_service_group)
    group_stats = (
        aggregated.groupby('service_group')
        .agg(
            average_score=('average_score', 'mean'),
            survey_total=('survey_total', 'sum'),
            departments=('department', 'nunique'),
        )
        .reset_index()
    )
    group_stats['average_score'] = group_stats['average_score'].round(3)
    group_stats['survey_total'] = group_stats['survey_total'].round(0).astype(int)

    comment_counts = [
        {
            'group': key,
            'mentions': int(comment_summary['demographics'].get(key, 0)),
        }
        for key in DEMOGRAPHIC_KEYWORDS.keys()
    ]

    return {
        'service_groups': group_stats.to_dict(orient='records'),
        'comment_mentions': comment_counts,
    }


def predictive_risk_model(df: pd.DataFrame) -> dict:
    df_sorted = df.sort_values(['department', 'period'])
    records = []
    for department, group in df_sorted.groupby('department'):
        group = group.dropna(subset=['period']).sort_values('period')
        group = group[['period', 'average_score']].copy()
        group['lag1'] = group['average_score'].shift(1)
        group['lag2'] = group['average_score'].shift(2)
        group['lag3'] = group['average_score'].shift(3)
        group['trend'] = group['average_score'] - group['lag1']
        group['rolling_mean3'] = group['average_score'].rolling(3).mean()
        group['rolling_std3'] = group['average_score'].rolling(3).std()
        group['target'] = (group['average_score'].shift(-1) < group['average_score'] - 0.05).astype(float)
        group['department'] = department
        records.append(group)
    if not records:
        return {'high_risk_departments': [], 'next_period': compute_next_period(df['period'].max())}

    feature_df = pd.concat(records).dropna(subset=['lag1', 'lag2', 'lag3', 'trend', 'rolling_mean3', 'rolling_std3'])

    train_df = feature_df.dropna(subset=['target'])
    if train_df.empty:
        return {'high_risk_departments': [], 'next_period': None}

    features = ['lag1', 'lag2', 'lag3', 'trend', 'rolling_mean3', 'rolling_std3']
    pipeline = Pipeline(
        steps=[
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, class_weight='balanced')),
        ]
    )
    pipeline.fit(train_df[features], train_df['target'])

    latest_records = []
    for department, group in feature_df.groupby('department'):
        last_row = group.sort_values('period').iloc[-1]
        latest_records.append(
            {
                'department': department,
                'features': last_row[features].astype(float),
                'last_period': last_row['period'].strftime('%Y-%m'),
                'last_score': float(last_row['average_score']),
            }
        )

    predictions = []
    for item in latest_records:
        feature_values = item['features']
        if feature_values.isna().any():
            continue
        probability = float(
            pipeline.predict_proba(
                pd.DataFrame([feature_values.to_dict()], columns=features)
            )[0, 1]
        )
        predictions.append(
            {
                'department': item['department'],
                'probability_of_decline': round(probability, 4),
                'last_period': item['last_period'],
                'last_score': round(item['last_score'], 3),
            }
        )

    predictions.sort(key=lambda row: row['probability_of_decline'], reverse=True)
    next_period = compute_next_period(df_sorted['period'].max())
    return {
        'high_risk_departments': predictions[:15],
        'next_period': next_period,
    }


def compute_next_period(period: Optional[pd.Timestamp]) -> Optional[str]:
    if pd.isna(period) or period is None:
        return None
    next_month = (period.to_period('M') + 1).to_timestamp()
    return next_month.strftime('%Y-%m')


def build_insights() -> dict:
    df = load_results()
    comments = read_comment_files()
    trends = compute_trends(df)
    volatility = compute_volatility(df)
    aggregated = compute_weighted_scores(df)
    high_performance = identify_high_performance(aggregated, volatility)
    critical_gaps = identify_critical_gaps(aggregated, volatility)
    comment_summary = summarize_comments(comments)
    wait_corr = wait_time_vs_satisfaction(df, comments)
    benchmarking = benchmark_top_bottom(aggregated, volatility)
    time_summary = time_of_day_summary(comment_summary)
    disparity = demographic_disparity(df, comment_summary)
    risk = predictive_risk_model(df)

    overall_average = round(float(aggregated['average_score'].mean()), 3)
    overall_volatility = round(float(aggregated['average_score'].std(ddof=0)), 4)

    return {
        'metadata': {
            'departments': int(df['department'].nunique()),
            'observations': int(len(df)),
            'overall_average_score': overall_average,
            'overall_score_std': overall_volatility,
            'comment_count': int(comment_summary['total_comments']),
        },
        'trends': trends,
        'volatility': volatility,
        'high_performance': high_performance,
        'critical_gaps': critical_gaps,
        'systemic_themes': {
            'themes': comment_summary['themes'],
            'sentiment_totals': comment_summary['sentiment'],
            'time_of_day': time_summary,
            'demographic_mentions': disparity['comment_mentions'],
        },
        'sentiment_over_time': comment_summary['sentiment_over_time'],
        'themes_over_time': comment_summary['themes_over_time'],
        'wait_time_correlation': wait_corr,
        'benchmarking': benchmarking,
        'predictive_risk': risk,
        'demographic_disparity': disparity,
        'weighted_scores': aggregated.round({'average_score': 3}).to_dict(orient='records'),
    }


def write_output(data: dict) -> None:
    OUTPUT_ANALYSIS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    OUTPUT_DASHBOARD_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def main() -> None:
    insights = build_insights()
    write_output(insights)
    print(f'Wrote {OUTPUT_ANALYSIS_PATH} and {OUTPUT_DASHBOARD_PATH}')


if __name__ == '__main__':
    main()
