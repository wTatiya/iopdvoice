import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

RESULTS_PATH = Path('analysis/results.json')
OUTPUT_PATH = Path('analysis/summary.json')


def load_results() -> pd.DataFrame:
    data = json.loads(RESULTS_PATH.read_text())
    return pd.DataFrame(data['raw_scores'])


def summarize(df: pd.DataFrame) -> dict:
    df = df.copy()
    df['year'] = df['year_ad']
    df['month'] = df['month']

    summary = {
        'metadata': build_metadata(df),
        'overall': build_rankings(df),
        'years': {},
        'months': {},
    }

    for year, group in df.groupby('year'):
        summary['years'][str(year)] = build_rankings(group)

    for (year, month), group in df.groupby(['year', 'month']):
        key = f"{year:04d}-{month:02d}"
        summary['months'][key] = build_rankings(group)

    return summary


def build_rankings(df: pd.DataFrame) -> dict:
    grouped = df.groupby('department').apply(weighted_average, include_groups=False).reset_index()
    grouped = grouped.sort_values('average_score', ascending=False)
    top10 = grouped.head(10).to_dict(orient='records')
    bottom10 = grouped.tail(10).sort_values('average_score').to_dict(orient='records')
    grouped['rank'] = range(1, len(grouped) + 1)
    return {
        'top10': top10,
        'bottom10': bottom10,
        'department_count': len(grouped),
        'average_of_averages': grouped['average_score'].mean(),
    }


def weighted_average(group: pd.DataFrame) -> pd.Series:
    has_weight = group['survey_count'].notna() & (group['survey_count'] > 0)
    if has_weight.any():
        weighted_sum = (group.loc[has_weight, 'average_score'] * group.loc[has_weight, 'survey_count']).sum()
        weight = group.loc[has_weight, 'survey_count'].sum()
        if weight > 0:
            avg = weighted_sum / weight
        else:
            avg = group['average_score'].mean()
        survey_total = int(weight)
    else:
        avg = group['average_score'].mean()
        survey_total = int(group['average_score'].count())
    return pd.Series({'average_score': avg, 'survey_total': survey_total})


def build_metadata(df: pd.DataFrame) -> dict:
    periods = sorted({(row.year, row.month) for row in df[['year', 'month']].itertuples(index=False)})
    period_labels = [f"{year:04d}-{month:02d}" for year, month in periods]
    total_surveys = (
        df.loc[df['survey_count'].notna() & (df['survey_count'] > 0), 'survey_count']
        .astype(float)
        .sum()
    )
    weighted_avg = weighted_average(df.assign(department='__overall__'))['average_score']
    return {
        'department_month_records': int(len(df)),
        'unique_departments': int(df['department'].nunique()),
        'years': sorted(df['year'].unique().tolist()),
        'periods': period_labels,
        'total_reported_surveys': float(total_surveys),
        'overall_average_score': float(weighted_avg),
    }


def main() -> None:
    df = load_results()
    summary = summarize(df)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Wrote {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
