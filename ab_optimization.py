from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


@dataclass
class OptimizationResult:
    correlation: float
    test_stores: List[str]
    control_stores: List[str]


def mondays_in_year(year: int) -> List[date]:
    d = date(year, 1, 1)
    while d.weekday() != 0:
        d += timedelta(days=1)

    mondays = []
    while d.year <= year:
        if d.year == year:
            mondays.append(d)
        d += timedelta(days=7)
    return mondays


def generate_weekly_revenue_data(
    n_stores: int = 20,
    year: int = 2025,
    seed: int = 42,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    stores = [f"Loja_{i:02d}" for i in range(1, n_stores + 1)]
    weeks = mondays_in_year(year)

    rows: List[Dict[str, object]] = []
    for store in stores:
        store_bias = rng.gauss(1.0, 0.12)
        trend = rng.gauss(0.0015, 0.0005)
        season_amp = rng.uniform(0.05, 0.18)

        for i, week_start in enumerate(weeks):
            season = 1 + season_amp * math.sin(2 * math.pi * i / len(weeks))
            growth = 1 + trend * i
            base = rng.gauss(120_000, 12_000)
            noise = rng.gauss(0, 8_500)
            revenue = max(10_000, base * store_bias * season * growth + noise)

            rows.append(
                {
                    "loja": store,
                    "ano": year,
                    "semana": week_start.isocalendar()[1],
                    "semana_inicio": week_start.isoformat(),
                    "receita": round(revenue, 2),
                }
            )
    return rows


def pearson(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("-inf")
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in x))
    den_y = math.sqrt(sum((b - my) ** 2 for b in y))
    den = den_x * den_y
    if den == 0:
        return float("-inf")
    return num / den


def weekly_totals(rows: List[Dict[str, object]], selected: set[str]) -> Dict[int, float]:
    totals: Dict[int, float] = {}
    for r in rows:
        if r["loja"] in selected:
            wk = int(r["semana"])
            totals[wk] = totals.get(wk, 0.0) + float(r["receita"])
    return totals


def evaluate_split(rows: List[Dict[str, object]], test_stores: List[str]) -> float:
    all_stores = sorted({str(r["loja"]) for r in rows})
    test_set = set(test_stores)
    control_set = set(all_stores) - test_set

    t = weekly_totals(rows, test_set)
    c = weekly_totals(rows, control_set)
    common_weeks = sorted(set(t) & set(c))
    test_values = [t[w] for w in common_weeks]
    control_values = [c[w] for w in common_weeks]
    return pearson(test_values, control_values)


def optimize_store_assignment(
    rows: List[Dict[str, object]],
    n_trials: int = 30_000,
    seed: int = 123,
) -> OptimizationResult:
    rng = random.Random(seed)
    stores = sorted({str(r["loja"]) for r in rows})
    if len(stores) % 2 != 0:
        raise ValueError("O número de lojas deve ser par para divisão metade/metade.")

    half = len(stores) // 2
    best_corr = float("-inf")
    best_test: List[str] = []

    for _ in range(n_trials):
        test_stores = sorted(rng.sample(stores, half))
        corr = evaluate_split(rows, test_stores)
        if corr > best_corr:
            best_corr = corr
            best_test = test_stores

    control_stores = sorted(set(stores) - set(best_test))
    return OptimizationResult(best_corr, best_test, control_stores)


def build_kpis(rows: List[Dict[str, object]], result: OptimizationResult) -> Dict[str, float]:
    test_set = set(result.test_stores)
    control_set = set(result.control_stores)
    t = weekly_totals(rows, test_set)
    c = weekly_totals(rows, control_set)
    weeks = sorted(set(t) & set(c))

    abs_diff = [abs(t[w] - c[w]) for w in weeks]
    ratio = [t[w] / c[w] for w in weeks if c[w] != 0]

    return {
        "correlacao_semanal_teste_vs_controle": round(result.correlation, 6),
        "media_abs_diferenca_semanal_receita": round(mean(abs_diff), 2),
        "media_razao_teste_sobre_controle": round(mean(ratio), 6),
        "n_lojas_teste": len(result.test_stores),
        "n_lojas_controle": len(result.control_stores),
        "n_semanas": len(weeks),
    }


def save_outputs(rows: List[Dict[str, object]], result: OptimizationResult, output_dir: Path) -> Tuple[Path, Path, Path]:
    data_dir = output_dir / "dados"
    result_dir = output_dir / "resultados"
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_dir / "receita_loja_semana_2025.csv"
    assignment_path = result_dir / "classificacao_lojas.csv"
    kpis_path = result_dir / "kpis_otimizacao.json"

    with dataset_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["loja", "ano", "semana", "semana_inicio", "receita"])
        writer.writeheader()
        writer.writerows(rows)

    assignments = [
        {"loja": loja, "grupo": "teste"} for loja in result.test_stores
    ] + [
        {"loja": loja, "grupo": "controle"} for loja in result.control_stores
    ]
    assignments.sort(key=lambda x: x["loja"])

    with assignment_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["loja", "grupo"])
        writer.writeheader()
        writer.writerows(assignments)

    kpis = build_kpis(rows, result)
    with kpis_path.open("w", encoding="utf-8") as f:
        json.dump(kpis, f, ensure_ascii=False, indent=2)

    return dataset_path, assignment_path, kpis_path


def main() -> None:
    output_dir = Path("output_ab")
    rows = generate_weekly_revenue_data(n_stores=20, year=2025, seed=42)
    result = optimize_store_assignment(rows, n_trials=30_000, seed=123)
    dataset_path, assignment_path, kpis_path = save_outputs(rows, result, output_dir)

    print("Arquivos gerados com sucesso:")
    print(f"- Base semanal: {dataset_path}")
    print(f"- Classificação T/C: {assignment_path}")
    print(f"- KPIs: {kpis_path}")


if __name__ == "__main__":
    main()
