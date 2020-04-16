
# Aceleradev DS Week 2 - Challenge

Vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.

Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.


## _Set up_ da análise


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
black_friday = pd.read_csv(r'data/black_friday.csv')
```


```python
black_friday.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User_ID</th>
      <th>Product_ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Occupation</th>
      <th>City_Category</th>
      <th>Stay_In_Current_City_Years</th>
      <th>Marital_Status</th>
      <th>Product_Category_1</th>
      <th>Product_Category_2</th>
      <th>Product_Category_3</th>
      <th>Purchase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000001</td>
      <td>P00069042</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8370</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000001</td>
      <td>P00248942</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>6.0</td>
      <td>14.0</td>
      <td>15200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000001</td>
      <td>P00087842</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>12</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1422</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000001</td>
      <td>P00085442</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>12</td>
      <td>14.0</td>
      <td>NaN</td>
      <td>1057</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000002</td>
      <td>P00285442</td>
      <td>M</td>
      <td>55+</td>
      <td>16</td>
      <td>C</td>
      <td>4+</td>
      <td>0</td>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7969</td>
    </tr>
  </tbody>
</table>
</div>



É possível observar que as colunas `Product_Category_2` e `Product_Category_3` tem alguns objetos nulos, vamos ver qual a porcentagem desses dados faltantes.


```python
missing_data_summary = pd.DataFrame({'columns': black_friday.columns, 'types': black_friday.dtypes,
                                     'missing (%)': black_friday.isnull().mean().round(4) * 100})
missing_data_summary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>columns</th>
      <th>types</th>
      <th>missing (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>User_ID</th>
      <td>User_ID</td>
      <td>int64</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Product_ID</th>
      <td>Product_ID</td>
      <td>object</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Gender</th>
      <td>Gender</td>
      <td>object</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>Age</td>
      <td>object</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Occupation</th>
      <td>Occupation</td>
      <td>int64</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>City_Category</th>
      <td>City_Category</td>
      <td>object</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Stay_In_Current_City_Years</th>
      <td>Stay_In_Current_City_Years</td>
      <td>object</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Marital_Status</th>
      <td>Marital_Status</td>
      <td>int64</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Product_Category_1</th>
      <td>Product_Category_1</td>
      <td>int64</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Product_Category_2</th>
      <td>Product_Category_2</td>
      <td>float64</td>
      <td>31.06</td>
    </tr>
    <tr>
      <th>Product_Category_3</th>
      <td>Product_Category_3</td>
      <td>float64</td>
      <td>69.44</td>
    </tr>
    <tr>
      <th>Purchase</th>
      <td>Purchase</td>
      <td>int64</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



De imediato podemos observar que a porcentagem de dados faltantes de produtos na categoria 3 é de 69%. Ainda não sabemos o impacto que esses dados faltantes teriam em um possível modelo..

## Questão 1

Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.


```python
def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape
```

## Questão 2

Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.


```python
def q2():
    # Retorne aqui o resultado da questão 2.
    return len(black_friday[(black_friday['Gender']=='F') & (black_friday['Age']=='26-35')])
```

## Questão 3

Quantos usuários únicos há no dataset? Responda como um único escalar.


```python
def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday['User_ID'].nunique()
```

## Questão 4

Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.


```python
def q4():
    # Retorne aqui o resultado da questão 4.
    return black_friday.dtypes.unique().size
```

## Questão 5

Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.


```python
def q5():
    # Retorne aqui o resultado da questão 5.
    return (len(black_friday) -len(black_friday.dropna())) / len(black_friday)
```

## Questão 6

Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.


```python
def q6():
    # Retorne aqui o resultado da questão 6.
    return int(black_friday.isnull().sum().max())
```

## Questão 7

Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.


```python
def q7():
    # Retorne aqui o resultado da questão 7.
    return black_friday['Product_Category_3'].dropna().mode()[0]
```

## Questão 8

Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

Antes de inciarmos vamos olhar como é a distribuição da nossa variável `Purchase` antes da normalização.


```python
black_friday.hist(column='Purchase');
```


![png](images/output_25_0.png)


Podemos ver que há uma concentração de valores logo após 5000 (moda) e depois disso temos uns picos entre 15000 e 1600 e posteriormente em 2000.


```python
def q8():
    # Retorne aqui o resultado da questão 8.

    _min = black_friday['Purchase'].min()
    _max = black_friday['Purchase'].max()
    _norm = (black_friday['Purchase'] - _min)/(_max-_min)


    return float(_norm.mean())
```

A normalização redimensiona os valores em um intervalo de `[0,1]`. Isso pode ser útil em alguns casos em que todos os parâmetros precisam ter a mesma escala positiva. No entanto, os outliers do conjunto de dados são perdidos. Assim, 1. A normalização torna o treinamento menos sensível à escala de recursos, para que possamos resolver melhor os coeficientes e é dada pela seguinte fórmula:


<p align="center">
  <img width="500" height="100" src="images/normalization.png">
</p>

 Vamos visualizar a distribuição da nossa variável `Purchase_norm` após a normalização.


```python
df = pd.DataFrame({'Purchase_norm' : (black_friday['Purchase'] -
                                      black_friday['Purchase'].min())/(black_friday['Purchase'].max() -
                                                                       black_friday['Purchase'].min())})
df.hist(column='Purchase_norm');
```


![png](images/output_31_0.png)


Todos os valores agora estão entre `0` e `1`, e caso houvessem outliers ele teriam desaparecido Nossos recursos agora são mais consistentes entre si, o que nos permitirá avaliar melhor a produção de nossos futuros modelos. Além disso, se usássemos algoritmos nesse conjunto de dados antes de normalizarmos, seria difícil (potencialmente não possível) convergir os vetores por causa dos problemas de dimensionamento. A normalização torna os dados mais condicionados à convergência.

## Questão 9

Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.


```python
def q9():
    # Retorne aqui o resultado da questão 9.

    _mean = black_friday["Purchase"].mean()
    _std = black_friday["Purchase"].std()
    _stand = (black_friday['Purchase'] - _mean)/(_std)

    return int(_stand[_stand.between(-1,1)].count())
```

A padronização redimensiona os dados para ter uma média ($\mu$) de 0 e desvio padrão ($\sigma$) de 1 (variação unitária). A padronização  por sua vez, tende a tornar o processo de treinamento mais uniforme, porque a condição numérica dos problemas de otimização é aprimorada, e é dada pela seguinte fórmula:

<p align="center">
  <img width="300" height="100" src="images/standartization.png">
</p>

```python
df['Purchase_stand'] = (black_friday["Purchase"] - black_friday["Purchase"].mean())/(black_friday["Purchase"].std())
df.hist(column='Purchase_stand');
```


![png](images/output_37_0.png)


A ideia da padronização é fazer com que ao executarmos modelos (regressão logística, SVMs, perceptrons, redes neurais etc.), os pesos estimados serão atualizados de maneira semelhante e não utilizará taxas diferentes durante o processo de criação. Isso fornecerá resultados mais precisos quando os dados forem padronizados pela primeira vez.

## Questão 10

Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

Inicialmente `não`. Não existe uma relação direta entre ambas as colunas que indique que quando há ocorrencia de um valor nulo em `Product_Category_2` o mesmo ocorra em `Product_Category_3`. Em resumo para cada ocorrência de null em `Product_Category_2` existe ao menos uma ocorrência não nula em `Product_Category_3`.


```python
black_friday[['Product_Category_2', 'Product_Category_3']].sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product_Category_2</th>
      <th>Product_Category_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>507517</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>254668</th>
      <td>8.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>28778</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>484895</th>
      <td>14.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>126950</th>
      <td>6.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Um outro ponto, conforme vimos em nosso dataframe que sumariza os dados nulos, a porcentagem de dados nulos em `Product_Category_3` é cerca de duas vezes mais do que em `Product_Category_2`


```python
missing_data_summary.T[['Product_Category_2', 'Product_Category_3']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product_Category_2</th>
      <th>Product_Category_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>columns</th>
      <td>Product_Category_2</td>
      <td>Product_Category_3</td>
    </tr>
    <tr>
      <th>types</th>
      <td>float64</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>missing (%)</th>
      <td>31.06</td>
      <td>69.44</td>
    </tr>
  </tbody>
</table>
</div>




```python
def q10():
    # Retorne aqui o resultado da questão 10.
    category_2 = black_friday[black_friday['Product_Category_2'].isna()]
    category_3 = category_2[category_2['Product_Category_3'].isna()]
    return category_2.shape == category_3.shape
```
