################################################
## Econometria 2 - Doutorado                  ##
## Simulação de MONTE CARLO utilizando Python ##
## Aluno: Leandro Marques                     ##
################################################

###############################
## Importing useful packages ##
###############################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

data = pd.read_csv("C:/Users/leand/OneDrive/Doutorado/2023-1/Econometria II/Trabalhos e Atividades/Atvd2_SMC/Densidade_Banda_Larga_Fixa.csv", sep=';')

data["Densidade"] = data["Densidade"].str.replace(',', '.').astype(float)
x = data["Densidade"]

n = len(x)

#Questao 01
num_simulations = 1000
Ey_star_list = []
Ey_list = []
b_est_list = []
sigma_est_list = []

for i in range(num_simulations):
    # Criar o erro aleatório e a variável $y^*$:
    b = 1.5  # Parâmetro escolhido
    sigma = 0.5  # Desvio padrão escolhido
    error = np.random.normal(0, sigma, n)
    y_star = x * b + error

    # Trunca a distribuição de $y^*$ para obter $y$:
    a = 50  # Valor escolhido para truncar
    y = y_star[y_star < a]

    # Calcula $E(y^*)$ e $E(y)$:
    Ey_star = np.mean(y_star)
    Ey = np.mean(y)
    Ey_star_list.append(Ey_star)
    Ey_list.append(Ey)

    # Estimar o modelo MQO com a amostra truncada via MLE:
    def neg_log_likelihood(params, y_data):
        b_est = params[0]
        sigma_est = params[1]
        y_star_est = x[:len(y_data)] * b_est
        residuals = y_data - y_star_est
        likelihood = np.sum(np.log(1 / (sigma_est * np.sqrt(2 * np.pi))) - (residuals ** 2) / (2 * sigma_est ** 2))
        return -likelihood

    result_truncated = minimize(neg_log_likelihood, [1, 1], args=(y,), method="L-BFGS-B", bounds=((None, None), (1e-6, None)))
    b_est, sigma_est = result_truncated.x
    b_est_list.append(b_est)
    sigma_est_list.append(sigma_est)

# Médias das estimativas
mean_Ey_star = np.mean(Ey_star_list)
mean_Ey = np.mean(Ey_list)
mean_b_est = np.mean(b_est_list)
mean_sigma_est = np.mean(sigma_est_list)

print("Média E(y^*):", mean_Ey_star)
print("Média E(y):", mean_Ey)
print("Média estimativa de b:", mean_b_est)
print("Média estimativa de sigma:", mean_sigma_est)

# Gráfico
# Estilização do gráfico
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Gráfico de dispersão
scatter = sns.scatterplot(x=x[:len(y)], y=y, alpha=0.5, label='Dados Truncados', color='blue')
scatter.set(xlabel='x', ylabel='y')

# Linha de regressão estimada
plt.plot(x[:len(y)], x[:len(y)] * mean_b_est, color='red', label=f'Reta de regressão estimada (b={mean_b_est:.2f})')

# Legendas e título
plt.legend()
plt.title('Simulação de Monte Carlo: Dados Truncados e Linha de Regressão Estimada', fontsize=12)

# Exibir gráfico
plt.show()


#--------
# Questão 2

mean_b_est_zero = []
mean_sigma_est_zero = []

for i in range(num_simulations):
    # Criar o erro aleatório e a variável y*:
    b = 1.5  # Parâmetro escolhido
    sigma = 0.5  # Desvio padrão escolhido
    error = np.random.normal(0, sigma, n)
    y_star = x * b + error

    # Atribuir valor zero às observações truncadas:
    a = 50  # Valor escolhido para truncar
    y_zero = np.where(y_star < a, y_star, 0)

    # Calcular E(y*) e E(y):
    Ey_star = np.mean(y_star)
    Ey_zero = np.mean(y_zero)
    print("E(y^*):", Ey_star)
    print("E(y):", Ey_zero)

    # Estimar o modelo MQO com a amostra truncada via MLE:
    result_zero = minimize(neg_log_likelihood, [1, 1], args=(y_zero,), method="L-BFGS-B", bounds=((None, None), (1e-6, None)))
    b_est_zero, sigma_est_zero = result_zero.x
    mean_b_est_zero.append(b_est_zero)
    mean_sigma_est_zero.append(sigma_est_zero)
    print("Estimativa de b:", b_est_zero)
    print("Estimativa de sigma:", sigma_est_zero)

mean_b_est_zero = np.mean(mean_b_est_zero)
mean_sigma_est_zero = np.mean(mean_sigma_est_zero)

# Gráfico
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

scatter = sns.scatterplot(x=x, y=y_zero, alpha=0.5, label='Zero Data', color='blue')
scatter.set(xlabel='x', ylabel='y')

plt.plot(x, x * mean_b_est_zero, color='red', label=f'Linha de regressão estimada (b={mean_b_est_zero:.2f})')

plt.legend()
plt.title('Simulação de Monte Carlo: Dados com Valor Zero e Linha de Regressão Estimada', fontsize=12)
plt.show()


## censurando ##

mean_b_est_censored_x = []
mean_sigma_est_censored_x = []

for i in range(num_simulations):
    # Criar o erro aleatório e a variável y*:
    b = 1.5  # Parâmetro escolhido
    sigma = 0.5  # Desvio padrão escolhido
    error = np.random.normal(0, sigma, n)
    y_star = x * b + error

    # Censurar uma parte da distribuição de x:
    a_x = 50  # Valor escolhido para censurar x
    x_censored = x[x < a_x]
    y_censored = y_star[x < a_x]

    # Calcular E(y*) e E(y):
    Ey_star = np.mean(y_star)
    Ey_censored = np.mean(y_censored)
    print("E(y^*):", Ey_star)
    print("E(y):", Ey_censored)

    # Estimar o modelo MQO com a amostra censurada via MLE:
    result_censored_x = minimize(neg_log_likelihood, [1, 1], args=(y_censored,), method="L-BFGS-B", bounds=((None, None), (1e-6, None)))
    b_est_censored_x, sigma_est_censored_x = result_censored_x.x
    mean_b_est_censored_x.append(b_est_censored_x)
    mean_sigma_est_censored_x.append(sigma_est_censored_x)
    print("Estimativa de b:", b_est_censored_x)
    print("Estimativa de sigma:", sigma_est_censored_x)

mean_b_est_censored_x = np.mean(mean_b_est_censored_x)
mean_sigma_est_censored_x = np.mean(mean_sigma_est_censored_x)

# Gráfico
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

scatter = sns.scatterplot(x=x_censored, y=y_censored, alpha=0.5, label='Censored Data', color='blue')
scatter.set(xlabel='x', ylabel='y')

plt.plot(x_censored, x_censored * mean_b_est_censored_x, color='red', label=f'Estimated Regression Line (b={mean_b_est_censored_x:.2f})')

plt.legend()
plt.title('Simulação de Monte Carlo: Dados Censurados em x e Linha de Regressão Estimada', fontsize=12)
plt.show()





import matplotlib as mpl

# Aumentar o limite de células do backend Agg
mpl.rcParams['agg.path.chunksize'] = 10000

# Função para desenhar a linha de regressão
def plot_regression_line(x, y, b_est, label, color):
    plt.plot(x, y * b_est, color=color, label=f'{label} (b={b_est:.2f})')

# Estilização do gráfico
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Gráfico de dispersão
scatter = sns.scatterplot(x=x[:len(y)], y=y, alpha=0.5, label='Dados Truncados', color='blue')
scatter.set(xlabel='x', ylabel='y')

# Linhas de regressão estimadas
plot_regression_line(x[:len(y)], x[:len(y)], mean_b_est, "Linha de regressão Q1", 'red')
plot_regression_line(x, x, mean_b_est_zero, "Linha de regressão Q2", 'green')

# Legendas e título
plt.legend()
plt.title('Simulação de Monte Carlo: Dados Truncados e Linha de Regressão Estimada (Q1 e Q2)', fontsize=12)

# Exibir gráfico
plt.show()



##Questão 3 ##


def generate_data(n, rho, xz_corr):
    np.random.seed(42)

    # Parâmetros do modelo
    beta_x = 1.5
    beta_z = -1
    gamma = 0.5

    # Gerar variáveis independentes
    xz = np.random.multivariate_normal([0, 0], [[1, xz_corr], [xz_corr, 1]], size=n)
    x = xz[:, 0]
    z = xz[:, 1]

    # Gerar erros correlacionados
    errors = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    u = errors[:, 0]
    v = errors[:, 1]

    # Modelo de seleção (1ª equação)
    y1 = z * beta_z + v
    d = (y1 > 0).astype(int)

    # Modelo de resultado (2ª equação)
    y2 = x * beta_x + u

    # Observações truncadas
    y2_trunc = y2 * d

    return pd.DataFrame({'x': x, 'z': z, 'd': d, 'y2_trunc': y2_trunc, 'y1': y1, 'y2': y2, 'u': u, 'v': v})


#MQO e Heckman

import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from statsmodels.discrete.discrete_model import Probit


def compare_models(data):
    data = data.assign(const=1)  # Adiciona a coluna 'const' ao DataFrame

    # Ordinary Least Squares (OLS) model
    ols = sm.OLS(data['y2_trunc'][data['d'] == 1], data[['const', 'x']][data['d'] == 1]).fit()

    # Heckman selection model
    sample_selection = IV2SLS(data['y2_trunc'], data[['const', 'x']], data['d'], data['z'])
    heckman = sample_selection.fit()

    return ols, heckman


import matplotlib.pyplot as plt
import seaborn as sns

def plot_estimated_coefficients(ols, heckman, ax, title):
    ax.bar(['MQO', 'Heckman'], [ols.params['x'], heckman.params['x']], color=['orange', 'blue'])
    ax.set_ylim(1.2, 1.8)
    ax.set_title(title)
    ax.axhline(1.5, linestyle='--', color='black', label='Valor Verdadeiro')
    ax.legend()

def plot_scatter(data, ax, title):
    sns.scatterplot(x=data['x'], y=data['y2_trunc'], hue=data['d'], palette=['red', 'green'], ax=ax)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y2_trunc')

## colinearidade entre as variáveis exógenas e instrumentais.


n = 1000
rho_values = [0, 0.5, 0.9]
xz_corr_values = [0, 0.5, 0.9]

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for i, rho in enumerate(rho_values):
    for j, xz_corr in enumerate(xz_corr_values):
        data = generate_data(n, rho, xz_corr)
        ols, heckman = compare_models(data)
        plot_estimated_coefficients(ols, heckman, axes[i, j], f'rho = {rho}, xz_corr = {xz_corr}')
        if i == 0:
            plot_scatter(data, axes[-1, j], f'rho = {rho}, xz_corr = {xz_corr}')

plt.tight_layout()
plt.show()



####








