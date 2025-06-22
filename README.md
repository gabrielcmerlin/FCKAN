# FCKAN: Evaluating KAN for Time Series Classification and Extrinsic Regression

## Description

Time Series Classification (TSC) and Time Series Extrinsic Regression (TSER) are critical tasks across diverse fields. While Fully Convolutional Networks (FCNs) effectively capture temporal dependencies, Kolmogorov–Arnold Networks (KANs) offer greater flexibility and interpretability. However, the integration of KANs with temporal encoders and their application to regression tasks remain largely unexplored. In this paper, we introduce FCKAN and Hybrid FCN-KAN, two novel architectures that combine FCNs and KANs for TSC and TSER. The first is an end-to-end model, while the second is a hybrid approach that leverages a pre-trained FCN as a feature extractor followed by a KAN. We conduct experiments on 147 benchmark datasets. For TSC, both architectures outperform non-temporal baselines and achieve performance competitive with FCNs. In TSER, although all models are statistically equivalent, temporal models consistently outperform non-temporal baselines.

## Structure

```bash
FCKAN/
├── experiments/                  # Scripts para execução dos experimentos
│   ├── classification/           # Lógica para experimentos de classificação
│   └── regression/               # Lógica para experimentos de regressão
├── outputs/                      # Saídas geradas pelos experimentos
│   ├── conf_matrix/              # Matrizes de confusão geradas pelas classificações
│   └── losses/                   # Curvas de perda dos modelos treinados
│   └── results/                  # Outros resultados tabulados/sumarizados
│   └── scatter/                  # Gráficos de dispersão
│   └── weights/                  # Pesos dos modelos treinados
├── src/                          # Código-fonte modular do projeto
│   ├── models/                   # Definições das arquiteturas dos modelos
│   └── utils/                    # Funções utilitárias (ex: carregamento de dados)
├── README.md                     # Este arquivo
└── environment.yaml              # Definição do ambiente Conda para reprodução
```

## How to start

### Downloading code:
```bash
git clone https://github.com/gabrielcmerlin/FCKAN.git
cd FCKAN
```

### Installing requisites:
```bash
conda env create -f environment.yaml
conda activate eniac
```

## Run

### Experiments
```bash
export PYTHONPATH=$(pwd)
python3 experiments/regression/exp_MLP.py # change 'regression' and 'MLP' for the exp you want to run
```

### Result Analysis
Run the Python Notebook named 'analysis.ipynb' located in 'outputs/results/'
