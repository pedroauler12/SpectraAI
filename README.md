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
**Nome do grupo aqui**

---

## 👨‍🎓 Integrantes: 
- <a href="https://www.linkedin.com/in/victorbarq/">Nome do integrante 1</a>
- <a href="https://www.linkedin.com/in/victorbarq/">Nome do integrante 2</a>
- <a href="https://www.linkedin.com/in/victorbarq/">Nome do integrante 3</a> 
- <a href="https://www.linkedin.com/in/victorbarq/">Nome do integrante 4</a> 
- <a href="https://www.linkedin.com/in/victorbarq/">Nome do integrante 5</a>
- <a href="https://www.linkedin.com/in/victorbarq/">Nome do integrante 6</a> 
- <a href="https://www.linkedin.com/in/victorbarq/">Nome do integrante 7</a>

## 👩‍🏫 Professores:
### Orientador(a) 
- <a href="https://www.linkedin.com/in/victorbarq/">Nome do orientador</a>
### Instrutores
- <a href="https://www.linkedin.com/in/victorbarq/">Nome do instrutor 1</a>
- <a href="https://www.linkedin.com/in/victorbarq/">Nome do instrutor 2</a> 
- <a href="https://www.linkedin.com/in/victorbarq/">Nome do instrutor 3</a> 
- <a href="https://www.linkedin.com/in/victorbarq/">Nome do instrutor 4</a>
- <a href="https://www.linkedin.com/in/victorbarq/">Nome do instrutor 5</a> 


---

## 📜 Descrição do Projeto

Apresente um **resumo executivo** do projeto, descrevendo:

- o problema a ser resolvido  
- o contexto do parceiro/mercado  
- a abordagem técnica proposta  
- o impacto esperado da solução  

**Limite:** 1–2 parágrafos curtos (até ~6 linhas cada).  
Seja objetivo e evite detalhes técnicos extensos.

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

- objetivo 1
- objetivo 2
- objetivo 3


## 📁 Estrutura do Repositório

```bash
├── artefatos/ → entregas formais de cada Sprint
├── artigo/    → artigo científico (markdown + PDF)
├── src/       → código modular reutilizável (.py)
├── notebooks/ → exploração, experimentos e narrativa
├── data/      → dados de entrada (não versionar arquivos grandes)
├── models/    → modelos treinados e checkpoints leves
├── outputs/   → métricas, gráficos e resultados gerados
├── slides/    → apresentações do projeto
├── assets/    → imagens, logos e recursos estáticos
├── requirements.txt
└── README.md
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
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3️⃣ Executar notebooks
```bash
jupyter notebook
```

### 4️⃣ Executar pipeline
```bash
python src/train.py
```
---

## 📊 Artefatos por Sprint

### Sprint 1 — Entendimento do problema e baseline clássico
- Análise exploratória dos dados (EDA)
- Pipeline inicial de pré-processamento
- Implementação de modelo baseline clássico (ML tradicional)

### Sprint 2 — Baseline Deep Learning
- Implementação de rede densa (MLP)
- Avaliação experimental comparativa (baseline clássico × MLP)
- Artigo científico — Versão 1 (fundamentação e metodologia inicial)

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
- PyTorch / TensorFlow
- Scikit-Learn
- Pandas / NumPy
- Matplotlib / Seaborn
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

