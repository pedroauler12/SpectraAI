<!-- ===================================================== -->
<!-- HEADER INSTITUCIONAL – TEMPLATE INTELI -->
<!-- ===================================================== -->
<p align="center">
  <a href="https://www.inteli.edu.br/">
    <img src="assets/logo_inteli.png" height="70" alt="Inteli">
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.fronteraminerals.com/">
    <img src="assets/logo_frontera.png" height="78" alt="Frontera Minerals">
  </a>
</p>


<h1 align="center">
SpectraAI — Sistema de Deep Learning para mapeamento de prospectividade de Terras Raras usando imagens de satélite
</h1>

<p align="center">
<b>Curso:</b> Ciência da Computação — Inteli<br>
<b>Módulo 09:</b> Deep Learning & Visão Computacional
</p>

<p align="center"><i>
Projeto acadêmico desenvolvido em parceria com a Frontera Minerals
</i></p>

---

## 👥 Grupo
**g01 — SpectraAI**

---

## 👨‍🎓 Integrantes:
- Drielly Farias
- Eduardo Rizk
- Giovanna Vieira
- Larissa Souza
- Mateus Pereira
- Pedro Auler

## 👩‍🏫 Professores:
### Orientador(a)
- Ana Cristina dos Santos
### Instrutores
- Cesar Cavini Almiñana
- Filipe Gonçalves de Souza Nogueira da Silva
- Pedro Marins Freire Teberga
- Maria Cristina Nogueira Gramani
- Jose Erinaldo da Fonseca
- Rodolfo Riyoei Goya

---

## 📜 Descrição do Projeto

O projeto SpectraAI busca identificar áreas com potencial de ocorrência de Terras Raras a partir de dados multiespectrais de satélite (ASTER), reduzindo incerteza operacional na priorização de alvos de prospecção para a Frontera Minerals.

A abordagem técnica combina pipeline de dados geoespaciais, construção de dataset para modelagem supervisionada, baselines clássicos de Machine Learning e evolução para modelos de Deep Learning, sempre com protocolo experimental reprodutível e comparação objetiva por métricas.

---

## 🎯 Objetivos do Módulo

Ao final deste módulo, o grupo deverá ser capaz de:

- aplicar técnicas de Visão Computacional e Deep Learning em um problema real
- construir e comparar modelos baseline e modelos avançados
- conduzir avaliação experimental com métricas adequadas
- garantir reprodutibilidade do pipeline (do dado ao resultado)
- documentar decisões técnicas de forma clara e estruturada
- produzir um artigo científico no formato SBC

---
## 🎯 Objetivos do Projeto

Defina metas técnicas claras e mensuráveis para o seu sistema.

- construir e validar um baseline clássico robusto para classificação (A02)
- comparar de forma justa modelos clássicos e modelos de Deep Learning nas sprints seguintes
- manter pipeline reprodutível com avaliação em treino/validação/teste e análise crítica de erros


## 📁 Estrutura do Repositório

```bash
├── artefatos/             # entregas formais de cada sprint
├── artigo/                # artigo científico (markdown + PDF)
├── src/                   # código modular reutilizável (.py)
│   └── apps/              # aplicações interativas (Streamlit)
├── notebooks/             # exploração, experimentos e narrativa
├── data/                  # dados de entrada
├── models/                # modelos treinados/checkpoints
├── outputs/               # métricas, gráficos e resultados gerados
├── slides/                # apresentações do projeto
├── assets/                # imagens, logos e recursos estáticos
├── testes/                # testes automatizados
├── Makefile
├── run_pipeline.py        # utilitário de execução iterativa do pipeline
├── requirements-dev.txt
└── README.md
```

---

## Entrega A11

A entrega final reproduzivel do pipeline end-to-end esta em
`artefatos/a11_pipeline_e2e/`.

> **Nota:** `run_pipeline.py` na raiz e um utilitario de desenvolvimento
> para experimentos iterativos. O entrypoint oficial da entrega A11 e
> `artefatos/a11_pipeline_e2e/main.py`, executado via modulo conforme
> documentado abaixo.

- README oficial do artefato: `artefatos/a11_pipeline_e2e/README.md`
- comando oficial:

```bash
python3 -m pip install -r artefatos/a11_pipeline_e2e/requirements.txt

python3 -m artefatos.a11_pipeline_e2e \
  --config artefatos/a11_pipeline_e2e/config.yaml
```

---

## 📌 Regras do Projeto

### 📘 notebooks/
- narrativa experimental
- exploração de dados (EDA)
- testes rápidos e análises
- **não concentrar lógica principal aqui**

### 🐍 src/
- funções reutilizáveis
- pipelines de processamento
- treino e inferência
- código organizado e modular

### 🧠 models/
- pesos e checkpoints leves
- versões finais dos modelos
- **evitar arquivos muito grandes (>100MB)**

### 📊 outputs/
- métricas
- gráficos
- tabelas
- resultados de experimentos
- **podem ser regenerados → evitar versionar arquivos temporários**

### 📄 artigo/
- artigo.md (fonte)
- artigo_sbc.pdf (exportado)

### ⚠️ Não versionar
- datasets grandes
- dados brutos sensíveis
- checkpoints gigantes
- outputs temporários
- arquivos intermediários
---

## ▶️ Como Executar

### 1️⃣ Clonar repositório
```bash
git clone <url-do-repo>
cd <repo>
```

### 2️⃣ Criar ambiente
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-dev.txt
```

### 3️⃣ Executar testes
```bash
make test
```

### 4️⃣ Executar notebooks
```bash
jupyter notebook
```

### 5️⃣ Executar pipeline iterativo de desenvolvimento
```bash
python run_pipeline.py
```

> Para a entrega oficial reprodutível (A11), use o comando abaixo (ver [Entrega A11](#entrega-a11)):
> ```bash
> python3 -m artefatos.a11_pipeline_e2e --config artefatos/a11_pipeline_e2e/config.yaml
> ```

### 6️⃣ Executar baseline clássico (A02)
```bash
jupyter notebook artefatos/a02_baseline_classico/a02_baseline_classico.ipynb
```
---

## 📊 Artefatos por Sprint

### Sprint 1 — Entendimento do problema e baseline clássico
- Análise exploratória dos dados (EDA)
- Pipeline inicial de pré-processamento
- Implementação e avaliação de baseline clássico (ML tradicional)

### Sprint 2 — Baseline Deep Learning
- Implementação de rede densa (MLP)
- Avaliação experimental comparativa (baseline clássico × MLP)
- Artigo científico — Versão 1 (fundamentação e metodologia inicial)

#### Entrega 2B — Funções de ativação da MLP
- `ReLU` nas camadas ocultas: acelera convergência e reduz saturação de gradiente.
- `Sigmoid` na saída para classificação binária: gera probabilidade no intervalo `[0, 1]`.
- `Softmax` na saída para classificação multiclasse: normaliza probabilidades para soma 1 entre classes.
- Implementação em: `src/models/mlp_activations.py`.

### Sprint 3 — CNN e análise de desempenho
- Implementação de CNN simples
- Avaliação experimental ampliada
- Técnicas de interpretabilidade dos modelos
- Artigo científico — Versão 2 (metodologia refinada)

### Sprint 4 — Transfer Learning e consolidação
- Modelo avançado com Transfer Learning e data augmentation
- Integração parcial do pipeline end-to-end
- Consolidação dos resultados experimentais
- Artigo científico — Versão 3 (resultados parciais)

### Sprint 5 — Entrega final
- Pipeline completo reprodutível (end-to-end)
- Artigo científico final no template SBC
- Apresentação técnica do projeto ao parceiro

---

## 🤝 Boas Práticas de Trabalho em Equipe

Este projeto adota práticas profissionais de engenharia e pesquisa.
Espera-se que o time:

### 📌 Organização técnica
- manter commits frequentes e descritivos
- versionar todas as entregas
- documentar decisões no notebook ou README
- manter código modular e reutilizável

### 📌 Colaboração
- dividir tarefas de forma equilibrada
- participar das cerimônias (Planning, Review, Retro)
- registrar decisões e experimentos
- garantir que todos compreendam o pipeline completo

### 📌 Qualidade
- resultados reprodutíveis
- notebooks narrativos (storytelling técnico)
- métricas claras e comparáveis
- estrutura de pastas padronizada

> 💡 Essas práticas impactam diretamente o desempenho individual e coletivo no módulo.

---

## 📚 Tecnologias
- Python
- Scikit-Learn
- Pandas / NumPy
- Matplotlib / Seaborn
- Rasterio / PyProj
- Jupyter Notebook

---

## 🧪 Reprodutibilidade

Todos os experimentos devem:
- ✅ rodar do zero em nova máquina
- ✅ usar seeds fixas
- ✅ salvar métricas e gráficos em `/outputs`
- ✅ documentar hipóteses e resultados no notebook
- ✅ permitir replicação por outro grupo

---

## 📄 Licença

Este repositório destina-se exclusivamente a fins acadêmicos e educacionais no contexto do curso de Ciência da Computação do Inteli.

A reutilização total ou parcial do código, dados ou materiais deve respeitar as políticas institucionais do Inteli e eventuais restrições acordadas com o parceiro do projeto.

---

✨ **Inteli — Instituto de Tecnologia e Liderança**
