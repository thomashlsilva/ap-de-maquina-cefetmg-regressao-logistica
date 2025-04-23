# Atividade: Implementação de Regressão Logística

## Descrição

Esta atividade implementa o modelo de **regressão logística**, um dos algoritmos fundamentais de Machine Learning para classificação binária. O foco principal da atividade é:

- Implementação do cálculo da função de ativação sigmoid e sua derivada.
- Implementação do **forward propagation** e **backward propagation** para cálculo dos gradientes.
- Cálculo da **função de perda** utilizando entropia cruzada (cross-entropy).
- Atualização dos pesos e do viés utilizando o **gradiente descendente**.
- Testes automatizados para validação das implementações.

---

## Conceitos Implementados

### 1. `FuncaoAtivacao()`
Representa a função de ativação e sua derivada. No caso, utilizamos a função sigmoid:

- Função sigmoid: \( \sigma(z) = \frac{1}{1 + e^{-z}} \)
- Derivada da sigmoid: \( \sigma'(z) = \sigma(z)(1 - \sigma(z)) \)

### 2. `RegressaoLogistica()`
Classe principal que implementa a regressão logística. Contém métodos para:

- **Forward propagation**: Calcula as ativações (probabilidades) baseadas nos pesos e viés.
- **Backward propagation**: Calcula os gradientes dos pesos e viés.
- **Função de perda**: Calcula a entropia cruzada entre as previsões e os valores reais.
- **Atualização dos pesos**: Atualiza os pesos utilizando o gradiente descendente.
- **Treinamento**: Ajusta os parâmetros do modelo com base nas iterações e taxa de aprendizado.

### 3. `plot_decision_boundary()`
Função para plotar a superfície de decisão do modelo de regressão logística, visualizando como o modelo separa as classes.

---

## Testes

O arquivo `regressao_logistica_test.py` (fornecido) valida os seguintes componentes:

- A função de ativação (sigmoid) e sua derivada.
- O cálculo de **forward propagation** e **backward propagation**.
- A implementação da **função de perda** (cross-entropy).
- A atualização dos pesos e viés após a aplicação do gradiente descendente.
- A precisão do modelo no processo de treinamento com o método `fit()`.

Todos os testes passam com sucesso, validando a implementação da regressão logística.

---

## Instruções de Execução

### Instalação de Dependências

Certifique-se de ter o Python e os pacotes necessários instalados:

```bash
apt-get install python3 jupyter python3-pip
pip install -r requirements.txt
```

Atenção: Evite usar sudo com o pip.

### Executando o Jupyter Notebook

```bash
jupyter notebook
```
Abra o notebook fornecido e execute as células conforme as instruções no arquivo.

---

## Créditos

Atividade desenvolvida para a disciplina de Machine Learning do CEFET-MG, Campus Nova Gameleira.

Material baseado nas aulas do professor Daniel Hasan Dalip.