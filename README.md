# PPGCC Coautorias

Análise e visualização de redes de coautoria entre 1.810 professores de pós-graduação em Ciência da Computação no Brasil, utilizando dados dos currículos Lattes (2014-2023) e a metodologia KDD.

---

## 📘 Visão Geral

Este projeto visa estudar as redes de colaboração acadêmica entre docentes de programas de pós-graduação em Ciência da Computação no Brasil. O foco abrange o período de **2014 a 2023** e considera **1.810 professores** ativos em programas de pós-graduação em Ciência da Computação no Brasil.

---

## 📁 Estrutura do Repositório

| Diretório / Arquivo | Descrição |
|---------------------|-----------|
| `notebooks/`        | Notebooks Jupyter com análise exploratória, construção da rede e visualizações |
| `src/`               | Código-fonte (scripts, módulos) para processar dados, calcular métricas, construir grafos etc. |
| `requirements.txt`   | Dependências Python necessárias para reproduzir a análise |
| `.python-version`    | Versão Python recomendada |
| `LICENSE`            | Licença de uso (MIT) |
| `README.md`          | Este documento |

---

## 🛠️ Instalação & Execução

1. Clone este repositório:

```bash
git clone https://github.com/jeojildo/ppgcc-coautorias.git
cd ppgcc-coautorias
```

2. Crie um ambiente virtual (recomendado):

```
python3 -m venv venv
source venv/bin/activate     # Linux/macOS
# ou
venv\Scripts\activate        # Windows
```

3. Instale as dependências:

```
pip install -r requirements.txt
```

4. Execute os notebooks em `notebooks/` ou scripts em `src/`, conforme o fluxo de análise desejado.

---

## 🔍 Principais funcionalidades

* **Extração** de dados do Currículo Lattes.
* **Limpeza e pré-processamento** dos dados (normalização de nomes, identidades, obras, etc.).
* **Construção de grafo de coautoria**, onde nós são professores e arestas representam colaborações.
* **Detecção de comunidades/clusters**.
* **Visualizações interativas** da rede, possibilitando filtros por ano, região, instituição, etc.
* **Exportação de resultados** para relatórios e tabelas.

---

## 📄 Licença

Este projeto está licenciado sob a **MIT License**. Consulte o arquivo `LICENSE` para mais detalhes.
