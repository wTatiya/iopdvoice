import json
import math
from pathlib import Path

import pandas as pd
from json.encoder import (
    _make_iterencode,
    encode_basestring,
    encode_basestring_ascii,
)

RESULTS_PATH = Path('analysis/results.json')
OUTPUT_PATH = Path('analysis/summary.json')
DOCS_OUTPUT_PATH = Path('docs/data/summary.json')


class TwoDecimalEncoder(json.JSONEncoder):
    def iterencode(self, o, _one_shot: bool = False):
        if self.check_circular:
            markers = {}
        else:
            markers = None

        if self.ensure_ascii:
            _encoder = encode_basestring_ascii
        else:
            _encoder = encode_basestring

        def floatstr(
            value,
            allow_nan=self.allow_nan,
            _inf=float('inf'),
            _neginf=-float('inf'),
        ):
            if not allow_nan:
                if math.isnan(value) or math.isinf(value):
                    raise ValueError(
                        f'Out of range float values are not JSON compliant: {value!r}'
                    )
            return format(value, '.2f')

        _iterencode = _make_iterencode(
            markers,
            self.default,
            _encoder,
            self.indent,
            floatstr,
            self.key_separator,
            self.item_separator,
            self.sort_keys,
            self.skipkeys,
            _one_shot,
        )
        return _iterencode(o, 0)


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
    grouped = (
        df.groupby('department')
        .apply(weighted_average, include_groups=False)
        .reset_index()
    )

    mean_score = grouped['average_score'].mean()
    std_dev = grouped['average_score'].std(ddof=0)
    if pd.isna(std_dev) or std_dev == 0:
        grouped['sigma_diff'] = 0.0
    else:
        grouped['sigma_diff'] = (grouped['average_score'] - mean_score) / std_dev

    top10 = (
        grouped.sort_values('sigma_diff', ascending=False)
        .head(10)
        .to_dict(orient='records')
    )
    bottom10 = (
        grouped.sort_values('sigma_diff', ascending=True)
        .head(10)
        .to_dict(orient='records')
    )
    grouped = grouped.sort_values('sigma_diff', ascending=False)
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
        weights = group.loc[has_weight, 'survey_count'].astype(float)
        weighted_sum = (group.loc[has_weight, 'average_score'] * weights).sum()
        weight = weights.sum()
        if weight > 0:
            avg = weighted_sum / weight
        else:
            avg = group['average_score'].mean()
        survey_total = int(weight)
        if 'recommendation_percent' in group:
            recommendation = weighted_mean(group.loc[has_weight, 'recommendation_percent'], weights)
        else:
            recommendation = None
    else:
        avg = group['average_score'].mean()
        survey_total = int(group['average_score'].count())
        if 'recommendation_percent' in group:
            recommendation = weighted_mean(group['recommendation_percent'])
        else:
            recommendation = None
    return pd.Series(
        {
            'average_score': avg,
            'survey_total': survey_total,
            'recommendation_percent': recommendation,
        }
    )


def weighted_mean(values: pd.Series, weights: pd.Series | None = None) -> float | None:
    valid = values.notna()
    if not valid.any():
        return None
    if weights is not None:
        valid_weights = weights[valid]
        total_weight = valid_weights.sum()
        if total_weight > 0:
            return float((values[valid] * valid_weights).sum() / total_weight)
    return float(values[valid].mean())


def build_metadata(df: pd.DataFrame) -> dict:
    periods = sorted({(row.year, row.month) for row in df[['year', 'month']].itertuples(index=False)})
    period_labels = [f"{year:04d}-{month:02d}" for year, month in periods]
    total_surveys = (
        df.loc[df['survey_count'].notna() & (df['survey_count'] > 0), 'survey_count']
        .astype(float)
        .sum()
    )
    overall_metrics = weighted_average(df.assign(department='__overall__'))
    weighted_avg = overall_metrics['average_score']
    recommendation = overall_metrics.get('recommendation_percent')
    return {
        'department_month_records': int(len(df)),
        'unique_departments': int(df['department'].nunique()),
        'years': sorted(df['year'].unique().tolist()),
        'periods': period_labels,
        'total_reported_surveys': float(total_surveys),
        'overall_average_score': float(weighted_avg),
        'overall_recommendation_percentage': None
        if recommendation is None or pd.isna(recommendation)
        else float(recommendation),
    }


def main() -> None:
    df = load_results()
    summary = summarize(df)
    payload = json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
        cls=TwoDecimalEncoder,
    )
    OUTPUT_PATH.write_text(payload)
    DOCS_OUTPUT_PATH.write_text(payload)
    print(f'Wrote {OUTPUT_PATH}')
    print(f'Wrote {DOCS_OUTPUT_PATH}')


if __name__ == '__main__':
    main()
