#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hydro-Logics: reproducible onset-pH estimator and synthetic/CSV evaluation.

Derived from the simulation supplied by the research team (D.V. and J.S.;
original header declares GNU GPL v3). Technical review for Jose Alfredo Sanchez,
2026-10-16. Distributed as a derivative under GNU GPL v3. This is NOT experimental validation or a HER
curve-level electrochemical simulator. Use --help for command-line arguments.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis

R_GAS = 8.31446261815324
F_CONST = 96485.33212
BUFFERS = (2.0, 4.0, 7.0, 10.0, 12.0)


def slope(T_c: float) -> float:
    """Absolute value of the ideal Nernst slope [V / pH] at T [deg C]."""
    T = float(T_c) + 273.15
    if not 250 < T < 450:
        raise ValueError(f"Temperatura no admisible: {T_c} °C")
    return math.log(10.0) * R_GAS * T / F_CONST


@dataclass
class Config:
    k_true_V: float = -0.3200
    temperature_C: float = 25.0
    sigma_ref_V: float = 0.015
    sigma_ohmic_residual_V: float = 0.015
    sigma_measurement_V: float = 0.002
    interference_probability: float = 0.05
    interference_min_V: float = 0.020
    interference_max_V: float = 0.030
    q_ph2: float = 0.0
    prior_pH: float = 7.0
    prior_variance_ph2: float = 1e6

    def gaussian_variance_V2(self) -> float:
        return (self.sigma_ref_V**2 + self.sigma_ohmic_residual_V**2
                + self.sigma_measurement_V**2)

    def interference_variance_V2(self) -> float:
        a, b = self.interference_min_V, self.interference_max_V
        return self.interference_probability * (a * a + a * b + b * b) / 3

    def check(self) -> None:
        slope(self.temperature_C)
        for name in ('sigma_ref_V', 'sigma_ohmic_residual_V', 'sigma_measurement_V',
                     'interference_min_V', 'interference_max_V', 'q_ph2'):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} debe ser >= 0")
        if not 0 <= self.interference_probability <= 1:
            raise ValueError('interference_probability fuera de [0,1]')
        if self.interference_max_V < self.interference_min_V:
            raise ValueError('interference_max_V debe ser >= interference_min_V')
        if self.prior_variance_ph2 <= 0:
            raise ValueError('prior_variance_ph2 debe ser > 0')


def generate_onset(ph: float, cfg: Config, rng: np.random.Generator,
                   interference: bool = True, temperature_C: float | None = None) -> float:
    """Synthetic onset [V]. Perturbations are independent PER READING.

    Important: zero-mean residual iR is NOT a physical uncompensated i*Ru drop.
    For actual samples model current, Ru, surface kinetics and temporal drift.
    """
    temp = cfg.temperature_C if temperature_C is None else temperature_C
    v = cfg.k_true_V - slope(temp) * ph
    v += rng.normal(0, cfg.sigma_ref_V)
    v += rng.normal(0, cfg.sigma_ohmic_residual_V)
    v += rng.normal(0, cfg.sigma_measurement_V)
    if interference and rng.random() < cfg.interference_probability:
        v += rng.choice((-1.0, 1.0)) * rng.uniform(
            cfg.interference_min_V, cfg.interference_max_V)
    return float(v)


def estimate_calibration(rows: list[dict]) -> tuple[float, float, dict]:
    """Fit fixed physical slope and estimate intercept across independent buffers.

    Each buffer is first averaged (avoids overweighting many replicates).
    uK is the across-buffer standard error of the fitted intercept, NOT k=2.
    """
    buffers: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        ph = float(r['pH_reference'])
        t = float(r['temperature_C'])
        v = float(r['Vonset_V'])
        if not all(map(math.isfinite, (ph, t, v))):
            raise ValueError('Dato no finito en calibración')
        buffers[round(ph, 6)].append(v + slope(t) * ph)
    if len(buffers) < 3:
        raise ValueError('Se requieren >=3 pH de buffer diferentes para calibrar')
    means = np.array([np.mean(values) for values in buffers.values()])
    k_est = float(np.mean(means))
    u_k = float(np.std(means, ddof=1) / math.sqrt(len(means)))
    summary = {'n_buffers': len(buffers), 'n_readings': len(rows),
               'buffers_pH': sorted(buffers), 'method': 'fixed-Nernst-slope; equal-buffer weight'}
    return k_est, u_k, summary


class PHKalman:
    """Scalar Kalman observer: y = K_cal - S(T) * pH + epsilon.

    Measurement variance is expressed in V^2; calibration uncertainty is a
    shared systematic term, propagated separately once after the final update.
    A single-state filter does not track correlated reference-electrode drift.
    """
    def __init__(self, k_cal_V: float, config: Config):
        self.k_cal = float(k_cal_V)
        self.cfg = config
        self.x = float(config.prior_pH)
        self.p = float(config.prior_variance_ph2)
        self.r = config.gaussian_variance_V2() + config.interference_variance_V2()
        if self.r <= 0:
            self.r = 1e-16  # numeric floor for zero-noise self-test

    def update(self, onset_V: float, temperature_C: float) -> tuple[float, float]:
        p_pred = self.p + self.cfg.q_ph2
        h = -slope(temperature_C)  # CRITICAL: negative observation derivative
        innovation = float(onset_V) - (self.k_cal + h * self.x)
        innov_var = h * h * p_pred + self.r
        gain = p_pred * h / innov_var  # negative for positive Nernst slope
        self.x += gain * innovation
        # Joseph form: numerically nonnegative covariance
        self.p = (1 - gain * h)**2 * p_pred + gain**2 * self.r
        return float(self.x), float(self.p)


def assess_readings(observations: list[dict], k_cal: float, u_k: float,
                    cfg: Config) -> dict:
    if not observations:
        raise ValueError('No hay mediciones para el filtro')
    f = PHKalman(k_cal, cfg)
    for obs in observations:
        f.update(float(obs['Vonset_V']), float(obs['temperature_C']))
    # Single shared calibration offset appears ONCE, not once per reading.
    # Approx. valid for equal temperatures, same K_cal used throughout.
    t_values = [float(o['temperature_C']) for o in observations]
    s_effective = float(np.mean([slope(t) for t in t_values]))
    u_filter = math.sqrt(max(0.0, f.p))
    u_total = math.sqrt(f.p + (u_k / s_effective)**2)
    return {'pH_est': f.x, 'u_filter_pH': u_filter,
            'u_calibration_pH': abs(u_k / s_effective),
            'u_combined_pH': u_total, 'U95_approx_pH': 2 * u_total,
            'n_readings': len(observations)}


def synthetic_calibration(cfg: Config, rng: np.random.Generator) -> list[dict]:
    return [dict(pH_reference=ph, temperature_C=cfg.temperature_C,
                 Vonset_V=generate_onset(ph, cfg, rng, interference=False))
            for ph in BUFFERS for _ in range(3)]


def run_synthetic(cfg: Config, rng: np.random.Generator, trials: int,
                  replicates: int) -> tuple[list[dict], dict, dict]:
    if trials < 2 or replicates < 1:
        raise ValueError('trials >= 2 y replicates >= 1')
    # Calibration and evaluations draw independent onset measurements.
    cal_rows = synthetic_calibration(cfg, rng)
    k_est, u_k, cal_info = estimate_calibration(cal_rows)
    levels = np.linspace(2.0, 12.0, 21)
    records = []
    for ph in levels:
        for trial in range(trials):
            obs = [dict(Vonset_V=generate_onset(float(ph), cfg, rng),
                        temperature_C=cfg.temperature_C) for _ in range(replicates)]
            result = assess_readings(obs, k_est, u_k, cfg)
            result.update(sample_id=f'sim_ph{ph:.1f}_trial{trial:03d}',
                          pH_reference=float(ph), role='synthetic',
                          temperature_C=cfg.temperature_C)
            records.append(result)
    cal_info.update(K_cal_estimated_V=k_est, u_K_standard_V=u_k,
                    K_cal_true_V=cfg.k_true_V,
                    K_offset_V=k_est - cfg.k_true_V)
    return records, cal_info, dict(mode='synthetic', n_levels=len(levels),
                                   n_trials_per_level=trials,
                                   readings_per_trial=replicates,
                                   n_simulated_test_readings=len(records) * replicates)


def load_onset_csv(input_path: Path) -> list[dict]:
    with input_path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        needed = {'sample_id', 'role', 'Vonset_V', 'temperature_C'}
        if not needed.issubset(set(reader.fieldnames or [])):
            raise ValueError(f'Faltan columnas: {sorted(needed - set(reader.fieldnames or []))}')
        result = []
        for n, row in enumerate(reader, 2):
            role = str(row['role']).strip().lower()
            if role not in ('calibration', 'test'):
                raise ValueError(f'Fila {n}: role debe ser calibration o test')
            if not row['sample_id'].strip():
                raise ValueError(f'Fila {n}: sample_id vacío')
            for col in ('Vonset_V', 'temperature_C'):
                val = float(row[col]);
                if not math.isfinite(val):
                    raise ValueError(f'Fila {n}: valor no finito {col}')
            row['role'] = role
            row['Vonset_V'] = float(row['Vonset_V'])
            row['temperature_C'] = float(row['temperature_C'])
            txt = (row.get('pH_reference') or '').strip()
            row['pH_reference'] = float(txt) if txt else None
            result.append(row)
    if not result:
        raise ValueError('CSV vacío')
    return result


def run_real(cfg: Config, rows: list[dict]) -> tuple[list[dict], dict, dict]:
    calibration = [r for r in rows if r['role'] == 'calibration']
    test = [r for r in rows if r['role'] == 'test']
    if not calibration or not test:
        raise ValueError('Se requieren filas separadas con role=calibration y role=test')
    if any(r['pH_reference'] is None for r in calibration):
        raise ValueError('Cada fila calibration requiere pH_reference medido')
    cal_ids = {r['sample_id'] for r in calibration}
    test_ids = {r['sample_id'] for r in test}
    if cal_ids.intersection(test_ids):
        raise ValueError('Fuga de datos: sample_id simultáneamente en calibration y test')
    batch_ids = {str(r.get('batch_id') or '').strip() for r in rows}
    if len(batch_ids) > 1:
        raise ValueError('Distintos batch_id: analice cada electrodo/matriz por separado')
    k_est, u_k, cal_info = estimate_calibration(calibration)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in test:
        grouped[row['sample_id']].append(row)
    records = []
    for ident, obs in grouped.items():
        labels = [o['pH_reference'] for o in obs if o['pH_reference'] is not None]
        if labels and (len(labels) != len(obs) or max(labels) - min(labels) > 0.03):
            raise ValueError(f'{ident}: etiqueta de pH incoherente dentro de un mismo sample_id')
        point = assess_readings(obs, k_est, u_k, cfg)
        point.update(sample_id=ident, role='test',
                     pH_reference=float(np.mean(labels)) if labels else None,
                     temperature_C=float(np.mean([o['temperature_C'] for o in obs])))
        records.append(point)
    cal_info.update(K_cal_estimated_V=k_est, u_K_standard_V=u_k,
                    warning='uK y ruido de referencia no validados metrológicamente')
    return records, cal_info, dict(mode='real', n_test_groups=len(records),
                                   n_test_readings=len(test), n_calibration_rows=len(calibration),
                                   batch_id=next(iter(batch_ids)))


def metrics_for(records: list[dict]) -> dict:
    labelled = [r for r in records if r.get('pH_reference') is not None]
    if not labelled:
        return {'n_total': len(records), 'n_labelled': 0,
                'note': 'Sin pH de referencia en test: no calculable RMSE/R2/cobertura'}
    true = np.array([r['pH_reference'] for r in labelled], dtype=float)
    pred = np.array([r['pH_est'] for r in labelled], dtype=float)
    u = np.array([r['U95_approx_pH'] for r in labelled], dtype=float)
    errors = pred - true
    rmse = float(np.sqrt(np.mean(errors**2)))
    var = float(np.sum((true - true.mean())**2))
    out = {'n_total': len(records), 'n_labelled': len(labelled),
           'RMSE_pH': rmse, 'MAE_pH': float(np.mean(np.abs(errors))),
           'bias_pH': float(np.mean(errors)),
           'median_U95_approx_pH': float(np.median(u)),
           'empirical_coverage_pct': float(100 * np.mean(np.abs(errors) <= u)),
           'excess_kurtosis': float(kurtosis(errors, fisher=True, bias=False))
           if len(errors) >= 4 and np.std(errors) > 0 else None,
           'uncertainty_note': '±2u aproximado, NO IC 95% garantizado bajo colas pesadas'}
    out['R2'] = float(1 - np.sum(errors**2)/var) if var > 0 else None
    out['NSE'] = out['R2']  # identical formula, not an independent validation
    return out


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    columns = list(records[0])
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader(); w.writerows(records)


def plot_results(records: list[dict], metrics: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labelled = [r for r in records if r.get('pH_reference') is not None]
    if not labelled:
        return
    target = np.array([r['pH_reference'] for r in labelled])
    pred = np.array([r['pH_est'] for r in labelled])
    err = pred - target
    u = np.array([r['U95_approx_pH'] for r in labelled])
    levels = sorted(set(target))
    xs = np.array(levels)
    avg = np.array([np.mean(pred[target == p]) for p in xs])
    spread = np.array([np.std(pred[target == p], ddof=1) if sum(target == p)>1 else 0
                       for p in xs])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([min(target), max(target)], [min(target), max(target)], 'k--', label='Ideal')
    ax.errorbar(xs, avg, yerr=spread, fmt='o', capsize=2, label='Media ± desviación')
    ax.set(xlabel='pH referencia', ylabel='pH estimado', title='Evaluación fuera de calibración')
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out_dir/'fig1_pH_referencia_vs_estimado.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(err, bins=min(40, max(8, int(np.sqrt(len(err))))), edgecolor='black', alpha=.8)
    ax.axvline(0, linestyle='--', color='black')
    ax.set(xlabel='Error (pH)', ylabel='Frecuencia', title='Distribución de error')
    fig.tight_layout(); fig.savefig(out_dir/'fig2_errores.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(target, abs(err), s=14, label='|error|')
    ax.scatter(target, u, s=10, alpha=.4, label='U ≈ 2u')
    ax.set(xlabel='pH referencia', ylabel='Unidades pH', title='Errores absolutos y amplitud del intervalo')
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir/'fig3_error_e_intervalos.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.axis('off')
    fmt = lambda key: f'{metrics[key]:.4f}' if metrics.get(key) is not None else 'N/D'
    msg = '\n'.join(['HYDRO-LOGICS — RESUMEN DE VALIDACIÓN',
                     f'N con etiqueta: {len(labelled)}', f'RMSE: {fmt("RMSE_pH")}',
                     f'MAE: {fmt("MAE_pH")}', f'Sesgo: {fmt("bias_pH")}',
                     f'R²: {fmt("R2")}',
                     f'Cobertura empírica: {fmt("empirical_coverage_pct")}%'])
    ax.text(.05, .95, msg, ha='left', va='top', family='monospace', fontsize=12)
    fig.tight_layout(); fig.savefig(out_dir/'fig4_resumen.png', dpi=180); plt.close(fig)


def extract_onsets(path: Path, output_path: Path, baseline_points: int,
                   threshold_sigma: float, consecutive: int,
                   current_sign: str) -> list[dict]:
    """Operational threshold, NOT HER-specific chemical identification."""
    from scipy.signal import savgol_filter
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'sweep_id', 'voltage_V', 'current_A', 'role', 'temperature_C'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f'CSV LSV requiere columnas {sorted(required)}')
        rows = list(reader)
    group: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        group[r['sweep_id']].append(r)
    results = []
    for ident, data in group.items():
        if len(data) < max(11, baseline_points + consecutive):
            raise ValueError(f'Curva {ident}: insuficientes puntos LSV')
        v = np.array([float(d['voltage_V']) for d in data])
        i = np.array([float(d['current_A']) for d in data])
        if not np.all(np.isfinite(v)) or not np.all(np.isfinite(i)):
            raise ValueError(f'Curva {ident}: valores no finitos')
        if not np.all(np.diff(v) < 0):
            raise ValueError(f'Curva {ident}: voltajes deben descender estrictamente')
        meta = data[0]
        for key in ('role', 'sample_id', 'temperature_C', 'pH_reference', 'batch_id'):
            if len({str(d.get(key, '')) for d in data}) != 1:
                raise ValueError(f'Curva {ident}: {key} cambia durante el barrido')
        if len(i) >= 7:
            ismooth = savgol_filter(i, window_length=5, polyorder=2, mode='interp')
        else:
            ismooth = i.copy()
        baseline = ismooth[:baseline_points]
        mid = float(np.median(baseline))
        sigma = float(1.4826 * np.median(abs(baseline - mid)))
        # Floor based on raw baseline std; actual instrument detection limit is unknown.
        sigma = max(sigma, float(np.std(baseline, ddof=1)), 1e-12)
        direction = -1 if current_sign == 'negative' else 1
        exceed = direction * (ismooth - mid) > threshold_sigma * sigma
        onset = None
        for j in range(baseline_points, len(i) - consecutive + 1):
            if bool(np.all(exceed[j:j + consecutive])):
                onset = float(v[j]); break
        if onset is None:
            # Do NOT silently invent onset or include as usable measurement.
            print(f'AVISO: {ident}: sin cruce de umbral; curva descartada')
            continue
        results.append({'sample_id': meta.get('sample_id') or ident,
                        'role': meta['role'], 'pH_reference': meta.get('pH_reference', ''),
                        'Vonset_V': onset, 'temperature_C': meta['temperature_C'],
                        'batch_id': meta.get('batch_id', ''), 'sweep_id': ident,
                        'baseline_sigma_A': sigma})
    if not results:
        raise ValueError('Ninguna curva cruzó el umbral')
    write_csv(output_path, results)
    return results


def self_test() -> None:
    c = Config(sigma_ref_V=0, sigma_ohmic_residual_V=0,
               sigma_measurement_V=0, interference_probability=0)
    c.check()
    for ph in (2., 7., 12.):
        f = PHKalman(-.32, c)
        v = -.32 - slope(25.) * ph
        estimated, cov = f.update(v, 25.)
        assert abs(estimated - ph) < 1e-6, (ph, estimated)
        assert cov >= 0
    f = PHKalman(-.32, c)
    assert f.update(-.32 - slope(25.) * 4, 25)[0] < 7
    # A smaller onset voltage must imply larger pH for same intercept.
    assert slope(35) > slope(25)
    rng = np.random.default_rng(42)
    records, cal, _ = run_synthetic(Config(), rng, trials=3, replicates=3)
    assert len(records) == 21 * 3
    assert abs(cal['K_cal_estimated_V'] - (-0.32)) < .05
    assert metrics_for(records)['n_labelled'] == 63
    print('SELF-TEST OK: signo, termodinámica, calibración y Monte Carlo')


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Hydro-Logics revisado: simulación, CSV real o extracción LSV.')
    p.add_argument('--mode', choices=['synthetic', 'real', 'extract'], default='synthetic')
    p.add_argument('--input', type=Path, help='CSV para modo real/extract')
    p.add_argument('--output-dir', type=Path, default=Path('hydrologics_output'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--trials', type=int, default=50, help='Trayectorias independientes POR pH')
    p.add_argument('--replicates', type=int, default=5, help='Lecturas por trayectoria')
    p.add_argument('--temperature-C', type=float, default=25.)
    p.add_argument('--interference-probability', type=float, default=.05)
    p.add_argument('--sigma-ref-mV', type=float, default=15., help='Ruido pseudo-referencia por lectura, mV')
    p.add_argument('--sigma-ohmic-mV', type=float, default=15., help='Residual óhmico por lectura, mV')
    p.add_argument('--sigma-measurement-mV', type=float, default=2., help='Ruido de medida de potencial, mV')
    p.add_argument('--process-variance-ph2', type=float, default=0., help='Q de Kalman en pH² por actualización')
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--baseline-points', type=int, default=15)
    p.add_argument('--threshold-sigma', type=float, default=3.)
    p.add_argument('--consecutive', type=int, default=3)
    p.add_argument('--current-sign', choices=['negative', 'positive'], default='negative')
    return p.parse_args()


def main() -> None:
    args = cli()
    if args.self_test:
        self_test(); return
    cfg = Config(temperature_C=args.temperature_C,
                 interference_probability=args.interference_probability,
                 sigma_ref_V=args.sigma_ref_mV / 1000.,
                 sigma_ohmic_residual_V=args.sigma_ohmic_mV / 1000.,
                 sigma_measurement_V=args.sigma_measurement_mV / 1000.,
                 q_ph2=args.process_variance_ph2)
    cfg.check()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == 'extract':
        if args.input is None:
            raise SystemExit('--input obligatorio en extract')
        if args.baseline_points < 3 or args.threshold_sigma <= 0 or args.consecutive < 1:
            raise SystemExit('Parámetros de detección no válidos')
        f = args.output_dir/'onsets_extraidos.csv'
        rows = extract_onsets(args.input, f, args.baseline_points,
                              args.threshold_sigma, args.consecutive,
                              args.current_sign)
        print(f'{len(rows)} curvas procesadas; onsets: {f}')
        print('IMPORTANTE: el umbral no garantiza que la corriente sea HER.')
        return
    if args.mode == 'synthetic':
        records, calibration, details = run_synthetic(
            cfg, np.random.default_rng(args.seed), args.trials, args.replicates)
    else:
        if args.input is None:
            raise SystemExit('--input obligatorio en real')
        records, calibration, details = run_real(cfg, load_onset_csv(args.input))
    metrics = metrics_for(records)
    write_csv(args.output_dir/'predicciones.csv', records)
    report = {'software': 'HydroLogics corrected v1.0',
              'validation_type': 'SYNTHETIC (NOT experimental)' if args.mode == 'synthetic'
              else 'EXPERIMENTAL CSV (labels only where supplied)',
              'config': asdict(cfg), 'calibration': calibration,
              'design': details, 'metrics': metrics,
              'caveats': [
                  'Vonset es empírico y no es el potencial reversible',
                  'Pendiente Nernst fija: requiere contrastar cinética y matriz',
                  'Ru e intensidad NO se simulan físicamente; el ruido iR es un residuo',
                  'Una referencia con deriva temporal correlacionada requiere otro modelo',
                  'Cobertura ±2u no equivale a garantía 95% con colas pesadas',
                  'Un ajuste/calibración no se debe evaluar con las mismas muestras']}
    with (args.output_dir/'resultados.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
    if not args.no_plots:
        plot_results(records, metrics, args.output_dir)
    print(f'Modo: {args.mode} | sesiones: {len(records)} | pendiente 25°C: {slope(25)*1000:.3f} mV/pH')
    print(f'K_cal: {calibration["K_cal_estimated_V"]:.5f} V; uK (estándar): {calibration["u_K_standard_V"]:.5f} V')
    for name in ('RMSE_pH', 'MAE_pH', 'bias_pH', 'R2', 'empirical_coverage_pct'):
        if name in metrics:
            print(f'{name}: {metrics[name]}')
    print(f'Archivos en: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
