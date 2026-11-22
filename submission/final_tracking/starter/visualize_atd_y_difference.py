"""Y-only original/corrected ATD differences, shared normalizations and viewer."""
from pathlib import Path
import base64
import hashlib
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/broadcast_atd_from_csv_v1'
ANALYSIS = BASE / 'coordinate_difference_analysis'
OUT = ANALYSIS / 'y_only_normalized_view'
os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'mplcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save(name, value):
    (OUT / name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                     allow_nan=False) + '\n', encoding='utf-8')


def read_y(path):
    values = []
    with path.open('rb') as stream:
        for frame, line in enumerate(stream):
            assert int(line.split(b':', 1)[0]) == frame + 1
            values.append(float(line.rstrip(b'\r\n').rsplit(b':', 1)[1].split(b',')[1]))
    return np.asarray(values) * 68.


def stats(dy):
    if not len(dy):
        return dict(frames=0)
    absolute = np.abs(dy)
    return dict(frames=len(dy), signed_min_m=dy.min(), signed_max_m=dy.max(),
                signed_mean_bias_m=dy.mean(), mean_absolute_difference_m=absolute.mean(),
                median_absolute_difference_m=np.median(absolute),
                rmse_y_m=np.sqrt(np.mean(dy ** 2)),
                absolute_p90_m=np.quantile(absolute, .90),
                absolute_p95_m=np.quantile(absolute, .95),
                absolute_p99_m=np.quantile(absolute, .99),
                absolute_max_m=absolute.max(),
                signed_central_95_percent_empirical_range_m=np.quantile(dy, [.025, .975]),
                rmse_y_pitch_normalized=np.sqrt(np.mean(dy ** 2)) / 68.,
                absolute_p95_pitch_normalized=np.quantile(absolute, .95) / 68.,
                signed_min_pitch_normalized=dy.min() / 68.,
                signed_max_pitch_normalized=dy.max() / 68.)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ANALYSIS / 'per_frame_coordinate_difference.csv'
    original = ROOT / 'data/broadcast/BAR-ATH_broadcast_atd.txt'
    corrected = BASE / 'BAR-ATH_broadcast_atd.txt'
    xml = ROOT / 'data/broadcast/BAR-ATH_broadcast_atd.xml'
    paths = [source, original, corrected, xml]
    before_hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    previous = json.loads((ANALYSIS / 'coordinate_difference_summary.json').read_text())
    for p in (original, corrected):
        assert before_hashes[str(p.relative_to(ROOT))] == previous['source_hashes'][str(p.relative_to(ROOT))]
    xmlroot = ET.parse(xml).getroot()
    assert float(xmlroot.findtext('.//FieldSize/Height')) == 68.
    assert float(xmlroot.findtext('.//FieldSize/Width')) == 105.
    d = pd.read_csv(source, float_precision='round_trip', low_memory=False)
    assert len(d) == 164488 and np.array_equal(d.frame, np.arange(len(d)))
    old_y = read_y(original)
    new_y = read_y(corrected)
    assert np.array_equal(old_y, d.original_y_m.to_numpy(), equal_nan=True)
    assert np.array_equal(new_y, d.corrected_y_m.to_numpy(), equal_nan=True)
    comparable = np.isfinite(old_y) & np.isfinite(new_y)
    assert np.array_equal(comparable, d.comparable.to_numpy(bool))
    dy = new_y - old_y
    dy[~comparable] = np.nan
    assert np.array_equal(dy, d.dy_m.to_numpy(), equal_nan=True)
    absolute = np.abs(dy)
    max_abs = float(np.nanmax(absolute))
    min_abs = float(np.nanmin(absolute))
    assert max_abs > min_abs and min_abs == 0.
    amplitude = (absolute - min_abs) / (max_abs - min_abs)
    pitch_normalized = dy / 68.
    changed = d.changed_existing_xy.to_numpy(bool)
    changed_y = comparable & (dy != 0)
    definitions = {'ALL_COMPARABLE': comparable, 'CHANGED_EXISTING_XY': changed,
                   'CHANGED_EXISTING_Y': changed_y, 'RETAINED_EXISTING': comparable & ~changed,
                   'CHANGED_FIRST_HALF': changed & d.half.eq(1).to_numpy(),
                   'CHANGED_SECOND_HALF': changed & d.half.eq(2).to_numpy()}
    cohorts = {label: stats(dy[mask]) for label, mask in definitions.items()}
    for record in cohorts.values():
        if record['frames']:
            record['absolute_p95_minmax_normalized'] = record['absolute_p95_m'] / max_abs
            record['rmse_y_minmax_normalized'] = record['rmse_y_m'] / max_abs
    result = d[['frame', 'atd_frame_id', 'video_time', 'time_sec', 'half',
                'original_y_m', 'corrected_y_m', 'comparable', 'changed_existing_xy',
                'newly_filled_original_missing', 'both_missing', 'source_kind',
                'source_is_prediction', 'comparison_status']].copy()
    result['original_y_pitch_normalized'] = old_y / 68.
    result['corrected_y_pitch_normalized'] = new_y / 68.
    result['delta_y_m'] = dy
    result['absolute_delta_y_m'] = absolute
    result['signed_delta_y_over_68'] = pitch_normalized
    result['absolute_delta_y_over_68'] = absolute / 68.
    result['absolute_delta_y_minmax_0_1'] = amplitude
    result['changed_existing_y'] = changed_y
    result.to_csv(OUT / 'y_difference_all_frames.csv', index=False)
    pd.DataFrame([dict(cohort=k, **{a: b for a, b in v.items() if not isinstance(b, (list, np.ndarray))})
                  for k, v in cohorts.items()]).to_csv(OUT / 'y_summary_by_cohort.csv', index=False)
    result.loc[comparable].nlargest(100, 'absolute_delta_y_m').to_csv(
        OUT / 'largest_100_y_differences.csv', index=False)
    n = previous['video_frames']
    peak = int(np.nanargmax(absolute[:n]))
    assert n == 164487 and not comparable[-1]
    metadata = dict(frames=n, ATD_rows=len(d), fps=25, default_frame=peak,
                    pitch_y_m=68., global_abs_min_m=min_abs, global_abs_max_m=max_abs,
                    comparable_frames=int(comparable.sum()), changed_existing_xy_frames=int(changed.sum()),
                    changed_existing_y_frames=int(changed_y.sum()),
                    filled_missing_excluded=int(d.newly_filled_original_missing.sum()),
                    both_missing_excluded=int(d.both_missing.sum()), cohorts=cohorts,
                    maximum_frame=dict(frame=peak, atd_frame_id=peak + 1,
                                       video_time=d.video_time.iloc[peak],
                                       original_y_m=old_y[peak], corrected_y_m=new_y[peak],
                                       delta_y_m=dy[peak], absolute_delta_y_m=absolute[peak],
                                       source_kind=d.source_kind.iloc[peak]),
                    formulas=dict(signed='delta_y_m=corrected_y_m-original_y_m',
                                  pitch_normalized='delta_y_m/68',
                                  minmax_absolute='abs(delta_y_m)/global_max_abs_delta_y_m'),
                    comparison_is_ground_truth_error=False, airborne_only_subset=False,
                    missing_comparisons_are_NaN=True, clipped_outside_pitch_coordinates=False,
                    input_source_hashes=before_hashes,
                    normalization_scope='One fixed scale for every comparable frame',
                    html_float_storage='float64, original_y_m, corrected_y_m, delta_y_m',
                    extra_final_ATD_row_in_CSV_but_not_HTML=True)
    sample = np.column_stack([old_y[:n], new_y[:n], dy[:n]]).astype('<f8')
    status = np.where(comparable[:n], np.where(changed[:n], 1, 2),
                      np.where(d.newly_filled_original_missing.to_numpy(bool)[:n], 3, 0)).astype('u1')
    source_ids = d.source_kind.iloc[:n].map({'ORIGINAL_BROADCAST_ATD': 0,
        'CSV_ORIGINAL_SNP_OBSERVATION': 1, 'CSV_YOLO26X_OBSERVATION': 2,
        'EXISTING_CSV_PREDICTION': 3}).to_numpy('u1')
    template = (ROOT / 'starter/templates/atd_y_difference.template.html').read_text(encoding='utf-8')
    html = template.replace('__METADATA__', json.dumps(clean(metadata), ensure_ascii=False, separators=(',', ':')))
    for key, array in [('ARRAY', sample), ('STATUS', status), ('SOURCE', source_ids)]:
        html = html.replace('__' + key + '_DATA__', base64.b64encode(array.tobytes()).decode('ascii'))
    (OUT / 'y_difference_interactive.html').write_text(html, encoding='utf-8')
    fontpath = Path('/System/Library/Fonts/AppleSDGothicNeo.ttc')
    if fontpath.exists():
        font_manager.fontManager.addfont(str(fontpath))
        plt.rcParams['font.family'] = [font_manager.FontProperties(fname=str(fontpath)).get_name(),
                                      'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    a = cohorts['ALL_COMPARABLE']
    c = cohorts['CHANGED_EXISTING_XY']
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), constrained_layout=True)
    f = np.arange(n)
    axes[0].plot(f, pitch_normalized[:n], lw=.45, color='#2773b9', rasterized=True)
    axes[0].axhline(0, color='#667085', lw=.7)
    low, high = np.asarray(c['signed_central_95_percent_empirical_range_m']) / 68.
    axes[0].axhspan(low, high, color='#de8a27', alpha=.13,
                   label=f'변경 프레임 ΔY의 중앙 95% 범위: {low:.4f} ~ {high:.4f}')
    axes[0].set(xlim=(0, n - 1), ylabel='ΔY / 68m', xlabel='원본 영상 프레임 (0부터)',
                title='Y 방향 차이 · ΔY = 보정 Y − 기존 Y / 양수·음수 방향 보존')
    axes[0].secondary_yaxis('right', functions=(lambda x: x * 68., lambda x: x / 68.)).set_ylabel('ΔY (m)')
    axes[0].legend(loc='lower left', fontsize=10)
    axes[1].plot(f, amplitude[:n], lw=.45, color='#2773b9', rasterized=True)
    axes[1].axhline(c['absolute_p95_m'] / max_abs, color='#d67e13', ls='--', lw=1,
                   label=f'변경 프레임 |ΔY| P95: {c["absolute_p95_m"]:.3f}m = {c["absolute_p95_m"] / max_abs:.4f}')
    axes[1].scatter([peak], [1], s=22, color='#ce3655', zorder=5)
    axes[1].annotate(f'f{peak} · {max_abs:.3f}m → 1.0000', (peak, 1),
                     xytext=(-135, -26), textcoords='offset points')
    axes[1].set(xlim=(0, n - 1), ylim=(-.02, 1.05), ylabel='|ΔY| 전체 공통 정규화 [0–1]',
                xlabel='원본 영상 프레임 (0부터)',
                title=f'Y 차이 크기 · 0m → 0 / 최대 {max_abs:.6f}m → 1 · 결측은 공백')
    axes[1].secondary_yaxis('right', functions=(lambda x: x * max_abs,
                                               lambda x: x / max_abs)).set_ylabel('|ΔY| (m)')
    axes[1].legend(loc='upper left', fontsize=10)
    for mask, label, color in [(comparable, '전체 비교 가능 (유지된 0 포함)', '#7c8795'),
                               (changed, '실제 XY 변경 프레임', '#2773b9')]:
        values = np.sort(absolute[mask])
        axes[2].plot(values / max_abs, np.arange(1, len(values) + 1) / len(values) * 100,
                     color=color, lw=1.6, label=f'{label}: {len(values):,}')
    axes[2].axhline(95, color='#d67e13', ls='--', lw=.9)
    axes[2].set(xlim=(-.005, 1.005), ylim=(0, 101), xlabel='|ΔY| 정규화 [0–1]',
                ylabel='누적 프레임 비율 (%)',
                title=f'변경 프레임 Y RMSE {c["rmse_y_m"]:.3f}m · MAE {c["mean_absolute_difference_m"]:.3f}m · P95 {c["absolute_p95_m"]:.3f}m')
    axes[2].secondary_xaxis('top', functions=(lambda x: x * max_abs,
                                             lambda x: x / max_abs)).set_xlabel('|ΔY| (m)')
    axes[2].legend(loc='lower right', fontsize=10)
    for ax in axes:
        ax.grid(alpha=.18)
    fig.suptitle('기존 ATD ↔ 보정 ATD: 프레임별 Y좌표 차이와 정규화', fontsize=17)
    fig.savefig(OUT / 'y_difference_normalized.png', dpi=180)
    fig.savefig(OUT / 'y_difference_normalized.pdf')
    plt.close(fig)
    assert np.nanmin(amplitude) == 0. and np.nanmax(amplitude) == 1.
    assert np.isnan(amplitude[~comparable]).all() and np.isnan(pitch_normalized[~comparable]).all()
    assert np.allclose(pitch_normalized[comparable] * 68., dy[comparable], atol=1e-12)
    assert np.allclose(amplitude[comparable] * max_abs, absolute[comparable], atol=1e-12)
    assert np.array_equal(np.frombuffer(sample.tobytes(), dtype='<f8').reshape(-1, 3),
                          sample, equal_nan=True)
    assert np.isclose(a['rmse_y_m'], previous['cohorts']['ALL_COMPARABLE']['rms_dy_m'], atol=1e-12)
    assert np.isclose(c['rmse_y_m'], previous['cohorts']['CHANGED_EXISTING_ONLY']['rms_dy_m'], atol=1e-12)
    assert all(sha(ROOT / p) == signature for p, signature in before_hashes.items())
    save('y_normalization_summary.json', metadata)
    save('verification.json', dict(passed=True, actual_TXT_y_matches_existing_CSV=True,
        all_ATD_rows_in_CSV=len(d), video_frames_in_HTML=n, delta_y_recomputed_from_actual_TXT=True,
        signed_and_absolute_normalization_invert_to_meters=True, missing_is_not_zero=True,
        previous_RMSE_y_reproduced=True, HTML_float64_exact_round_trip=True,
        original_corrected_ATD_and_analysis_sources_unchanged=True,
        common_pitch_width_m=68., no_airborne_subset_or_accuracy_claim=True))
    (OUT / 'README.md').write_text(f'''# ATD Y좌표 차이만 비교

[프레임별 인터랙티브 화면](y_difference_interactive.html) · [정규화 그래프](y_difference_normalized.png) · [PDF](y_difference_normalized.pdf) · [전체 프레임 CSV](y_difference_all_frames.csv)

기존 ATD와 보정 ATD의 **경기장 Y좌표**를 비교합니다. 높이 Z나 화면의 픽셀 Y가 아닙니다.
공중볼 여부를 새로 분류하거나 공중볼 구간만 선별한 결과가 아닙니다.
공중볼의 평면 투영 오차는 카메라 위치에 따라 X와 Y 모두에 생길 수 있습니다.
이 분석에서는 요청에 따라 Y 차이만 추출했습니다. 실제 위치 정답이 없어 위치 정확도 개선은 평가하지 않습니다.

- 방향을 포함한 차이: `ΔY = 보정 Y − 기존 Y`.
- 경기장 폭 기준 정규화: `ΔY/68`. 모든 프레임에 같은 68m를 적용하고 부호를 유지합니다.
- 크기 0–1 정규화: `|ΔY|/{max_abs:.12f}`. 0은 0m, 1은 관측된 최대 {max_abs:.6f}m입니다.
- 두 번째 정규화도 전체 비교 가능한 프레임에 하나의 고정 척도를 사용합니다. 프레임별로 다른 최솟값/최댓값을 적용하지 않습니다.
- 경기장 밖 좌표는 자르지 않습니다. ΔY/68은 ±1을 넘을 수 있습니다.
- 통계의 중앙 95% 범위는 ΔY의 2.5~97.5백분위, |ΔY| P95는 절대 차이의 95백분위입니다. 신뢰구간이 아닙니다.

전체 비교 가능 {a['frames']:,}프레임: Y RMSE {a['rmse_y_m']:.6f}m, |ΔY| P95 {a['absolute_p95_m']:.6f}m.
실제 XY 변경 {c['frames']:,}프레임: Y RMSE {c['rmse_y_m']:.6f}m, |ΔY| P95 {c['absolute_p95_m']:.6f}m.
ΔY 전체 범위는 {a['signed_min_m']:.6f}~{a['signed_max_m']:.6f}m입니다.
유지된 원본 좌표 {int((comparable & ~changed).sum()):,}프레임은 차이 0으로 별도 구분합니다.
원본 결측 보완 {int(d.newly_filled_original_missing.sum()):,}프레임과 양쪽 결측 {int(d.both_missing.sum()):,}프레임의 차이는 NaN입니다.

HTML은 원본 영상의 {n:,}프레임 전부를 내장한 로컬 파일이며 서버·외부 라이브러리가 필요 없습니다.
축소 화면에서는 각 화면 열의 최솟값·최댓값을 보존하고, 확대하면 각 프레임을 표시합니다.
슬라이더·숫자 입력·방향키로 이동하며 선택 프레임의 기존 Y와 보정 Y를 비교합니다.
선택적으로 기존 전체 비교 MP4와 연결됩니다. 원본/보정 ATD, Homography, 기존 그래프, 기존 영상은 변경하지 않았습니다.
CSV에는 ATD {len(d):,}행을 모두 보존하며 영상 밖 마지막 1행은 HTML·그래프에서 제외했습니다.
''', encoding='utf-8')
    files = ['y_difference_interactive.html', 'y_difference_all_frames.csv',
             'y_difference_normalized.png', 'y_difference_normalized.pdf',
             'y_summary_by_cohort.csv', 'y_normalization_summary.json',
             'verification.json', 'largest_100_y_differences.csv', 'README.md']
    save('artifact_integrity.json', {name: sha(OUT / name) for name in files})
    print(json.dumps(clean(dict(all=a, changed=c, peak=metadata['maximum_frame'],
                               output=str(OUT))), ensure_ascii=False))


if __name__ == '__main__':
    main()
