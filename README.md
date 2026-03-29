# PPGCC Coauthorship Network Analysis

Analise de relacoes de coautoria nos programas de pos-graduacao em Ciencia da Computacao no Brasil (2014-2023).

Pipeline baseado no processo Knowledge Discovery in Databases (KDD), cobrindo selecao, pre-processamento, transformacao, visualizacao e deteccao de comunidades em redes de coautoria ponderadas.

## Requisitos

- Python 3.10+

## Instalacao

```bash
git clone <url-do-repositorio>
cd ppgcc-coautorias
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuracao

Os parametros do pipeline estao em `config.py`:

| Parametro | Descricao | Padrao |
|---|---|---|
| `START_YEAR` | Ano inicial do filtro temporal | `2014` |
| `END_YEAR` | Ano final do filtro temporal | `2023` |
| `MIN_INSTITUTION_COAUTHORSHIPS` | Minimo de coautorias para a rede institucional | `50` |
| `MIN_CHORD_COAUTHORSHIPS` | Minimo de coautorias para o diagrama chord | `150` |
| `COMMUNITY_METHOD` | Metodo de deteccao de comunidades | `"louvain"` |
| `COMMUNITY_RESOLUTION` | Parametro de resolucao dos algoritmos de comunidade | `0.6` |
| `COMMUNITY_SEED` | Semente para reprodutibilidade | `42` |
| `COMMUNITY_COMPARISON_METHODS` | Metodos comparados na etapa de comunidades | `["greedy_modularity", "louvain", "leiden"]` |

## Execucao

```bash
source .venv/bin/activate
python coauthorships.py
```

O pipeline executa 5 etapas sequenciais:

1. **Selecao** -- Download e extracao dos curriculos Lattes (XMLs) via Google Drive
2. **Pre-processamento** -- Parsing dos XMLs, construcao do DataFrame de producoes, normalizacao de autores e filtro temporal
3. **Transformacao** -- Construcao da matriz de adjacencia ponderada (coautorias)
4. **Visualizacao** -- Distribuicao de grau, caminhos minimos, densidade, centralidade de intermediacao e grafo da rede
5. **Deteccao de comunidades** -- Comparacao de algoritmos, perfil estrutural e institucional das comunidades, papeis dos autores e metricas institucionais

Cada etapa verifica se os artefatos ja existem e pula o processamento caso positivo. Para re-executar uma etapa, remova os arquivos correspondentes em `data/` ou `results/`.

## Saidas

### Dados intermediarios (`data/`)

- `01-selection/` -- Curriculos Lattes em XML
- `02-preprocessing/productions.parquet` -- Producoes normalizadas
- `03-transformation/adjacency.parquet` -- Matriz de adjacencia

### Resultados (`results/`)

- `network/` -- Figuras da analise de topologia (distribuicao de grau, caminhos minimos, centralidade)
- `communities/` -- Artefatos da deteccao de comunidades:
  - `community_comparison.csv` -- Comparacao entre algoritmos
  - `community_assignments.csv` -- Atribuicao autor-comunidade
  - `community_report.csv` -- Metricas globais (modularidade, cobertura, performance)
  - `community_structural_profile.csv` -- Perfil estrutural por comunidade
  - `community_institutional_profile.csv` -- Perfil institucional por comunidade
  - `institution_network_metrics.csv` -- Metricas institucionais
  - `node_roles.csv` -- Papeis dos autores (hub, connector, peripheral)
  - `fig_roles_scatter.png` -- Grafico de dispersao P vs z_intra
  - `coauthorship_network_*.png` -- Rede de coautoria colorida por comunidade

## Estrutura do projeto

```
ppgcc-coautorias/
  config.py              # Parametros do pipeline
  coauthorships.py       # Pipeline principal (ponto de entrada)
  requirements.txt       # Dependencias Python
  coauths/
    selection.py         # Etapa 1: download e extracao
    preprocessing.py     # Etapa 2: parsing e normalizacao
    transformation.py    # Etapa 3: matriz de adjacencia
    visualization.py     # Etapa 4-5: graficos e analise de comunidades
    community_detection.py # Algoritmos e metricas de comunidades
    utils.py             # Funcoes auxiliares
```
