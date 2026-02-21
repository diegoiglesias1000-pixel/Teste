# Modelo de otimização para grupos de Teste A/B

Este projeto gera uma base ilustrativa de receita semanal por loja para 2025 e otimiza a alocação de lojas em grupos de **teste** e **controle** (50/50), maximizando a correlação semanal de vendas entre os grupos.

## Como usar

```bash
python3 ab_optimization.py
```

## Outputs

Ao executar, serão gerados arquivos em `output_ab/`:

1. `dados/receita_loja_semana_2025.csv`  
   Base semanal com colunas: `loja`, `ano`, `semana`, `semana_inicio`, `receita`.

2. `resultados/classificacao_lojas.csv`  
   Classificação final de cada loja no grupo `teste` ou `controle`.

3. `resultados/kpis_otimizacao.json`  
   KPIs do cenário ótimo (incluindo correlação semanal entre grupos).

## Lógica de otimização

- Sorteia múltiplas combinações aleatórias (metade das lojas em cada grupo).
- Calcula a correlação semanal entre receita agregada de teste e controle.
- Mantém a melhor divisão observada (maior correlação).
