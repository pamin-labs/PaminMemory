"""Render the documented tables from retained numerical evidence."""

def allocation_table(records):
    lines = ['| Case / metric | Before | After | Absolute difference | Percentage change |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in records:
        for label, key in [('requests', 'calls'), ('bytes requested', 'bytes_requested')]:
            before, after = row['before_' + key], row['after_' + key]
            lines.append(f"| {row['case']} / {label} | {before:,} | {after:,} | {after-before:+,} | {(after/before-1)*100:+.2f}% |")
    return '\n'.join(lines)

def runtime_table(summary):
    lines = ['| Metric | Before | Candidate | Absolute difference | Percentage change |',
             '| --- | ---: | ---: | ---: | ---: |']
    count = summary['pairs_per_round'] * summary['rounds']
    lines.append(f'| Candidate score checks per arm | {count:,} | {count:,}, exactly equal | 0 | 0% |')
    fields = [('Rank p50', 'rank_p50', 6, 's'), ('Rank p95', 'rank_p95', 6, 's'),
              ('Median 60-query rank-call wall sum', 'median_rank_wall_total', 6, 's'),
              ('Median whole-process CPU user+system', 'median_process_cpu_seconds', 2, 's'),
              ('Median peak process RSS', 'median_peak_process_rss_bytes', 0, 'B')]
    for label, key, decimals, unit in fields:
        before, after = summary['baseline'][key], summary['candidate'][key]
        def value(number, signed=False):
            if decimals == 0:
                return f'{number:+,.0f}' if signed else f'{number:,.0f}'
            return format(number, ('+' if signed else '') + f'.{decimals}f')
        lines.append(f'| {label} | {value(before)} {unit} | {value(after)} {unit} | {value(after-before, True)} {unit} | {(after/before-1)*100:+.2f}% |')
    for label in ('Whole search latency', 'Model-only steady RSS', 'Isolated total disk'):
        lines.append(f'| {label} | N/A | N/A | N/A | N/A |')
    return '\n'.join(lines)
