# PPGCC Coauthorship Network Analysis

Análise de relações de coautoria nos programas de pós-graduação em Ciência da Computação no Brasil (2014-2023).

Pipeline baseado no processo Knowledge Discovery in Databases (KDD), cobrindo seleção, pré-processamento, transformação, visualização e detecção de comunidades em redes de coautoria ponderadas.

## Requisitos

- Python 3.10+

## Instalação

```bash
git clone <url-do-repositorio>
cd ppgcc-coautorias
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuração

Os parâmetros do pipeline estão em `config.py`:

| Parâmetro | Descrição | Padrão |
|---|---|---|
| `START_YEAR` | Ano inicial do filtro temporal | `2014` |
| `END_YEAR` | Ano final do filtro temporal | `2023` |
| `MIN_INSTITUTION_COAUTHORSHIPS` | Mínimo de coautorias para a rede institucional | `50` |
| `MIN_CHORD_COAUTHORSHIPS` | Mínimo de coautorias para o diagrama chord | `150` |
| `COMMUNITY_METHOD` | Método de detecção de comunidades | `"leiden"` |
| `COMMUNITY_RESOLUTION` | Parâmetro de resolução dos algoritmos de comunidade | `0.6` |
| `COMMUNITY_SEED` | Semente para reprodutibilidade | `42` |
| `COMMUNITY_COMPARISON_METHODS` | Métodos comparados na etapa de comunidades | `["greedy_modularity", "louvain", "leiden"]` |

## Execução

```bash
source .venv/bin/activate
python coauthorships.py
```

O pipeline executa 5 etapas sequenciais:

1. **Seleção** -- Download e extração dos currículos Lattes (XMLs) via Google Drive
2. **Pré-processamento** -- Parsing dos XMLs, construção do DataFrame de produções, normalização de autores e filtro temporal
3. **Transformação** -- Construção da matriz de adjacência ponderada (coautorias)
4. **Visualização** -- Distribuição de grau, caminhos mínimos, densidade, centralidade de intermediação e grafo da rede
5. **Detecção de comunidades** -- Comparação de algoritmos, perfil estrutural e institucional das comunidades, papéis dos autores e métricas institucionais

Cada etapa verifica se os artefatos já existem e pula o processamento caso positivo. Para re-executar uma etapa, remova os arquivos correspondentes em `data/` ou `results/`.

## Saídas

### Dados intermediários (`data/`)

- `01-selection/` -- Currículos Lattes em XML
- `02-preprocessing/productions.parquet` -- Produções normalizadas
- `03-transformation/adjacency.parquet` -- Matriz de adjacência

### Resultados (`results/`)

- `network/` -- Figuras da análise de topologia (distribuição de grau, caminhos mínimos, centralidade)
- `communities/` -- Artefatos da detecção de comunidades:
  - `community_comparison.csv` -- Comparação entre algoritmos
  - `community_assignments.csv` -- Atribuição autor-comunidade
  - `community_report.csv` -- Métricas globais (modularidade, cobertura, performance)
  - `community_structural_profile.csv` -- Perfil estrutural por comunidade
  - `community_institutional_profile.csv` -- Perfil institucional por comunidade
  - `institution_network_metrics.csv` -- Métricas institucionais
  - `node_roles.csv` -- Papéis dos autores (hub, connector, peripheral)
  - `fig_roles_scatter.png` -- Gráfico de dispersão P vs z_intra
  - `coauthorship_network_*.png` -- Rede de coautoria colorida por comunidade

## Estrutura do projeto

```
ppgcc-coautorias/
  config.py              # Parâmetros do pipeline
  coauthorships.py       # Pipeline principal (ponto de entrada)
  requirements.txt       # Dependências Python
  coauths/
    selection.py         # Etapa 1: download e extração
    preprocessing.py     # Etapa 2: parsing e normalização
    transformation.py    # Etapa 3: matriz de adjacência
    visualization.py     # Etapa 4-5: gráficos e análise de comunidades
    community_detection.py # Algoritmos e métricas de comunidades
    utils.py             # Funções auxiliares
```