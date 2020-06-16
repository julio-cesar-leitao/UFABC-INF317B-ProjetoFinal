#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 12:34:16 2020

@author: juliocesar
"""

"""
Este código foi feito em python 3
Parte do código foi retirada de:
https://www.ritchievink.com/blog/2018/05/18/algorithm-breakdown-affinity-propagation/
O dataset olivettifaces.gif pode ser baixado em <https://cs.nyu.edu/~roweis/data.html>
O arquivo olivettifaces.gif deve estar no mesmo diretório do código
O objetivo é aplicar o algoritmo Affinity Propagation ao dataset acima.
"""

# Todos os imports necessarios para este trabalho
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg

# importando o arquivo
# essa função retorna um dataframe com 400 linhas, cada linha "i" é uma imagem 
# i transformada em 1 dimensão, presente no dataset original.
def preparar_arquivo_base(arquivo = "olivettifaces.gif"):
    import pandas as pd
    if (arquivo=="olivettifaces.gif"):
        import matplotlib.image as mpimg
        img = mpimg.imread(arquivo)
        base=pd.DataFrame(img[0:,0:,0])
        largura = 47
        altura = 57
        df3=pd.DataFrame()
        for i in range(1,21):
            for j in range(1,21):
                imagem_linha = i
                imagem_coluna = j
                base3=base.iloc[(imagem_linha-1)*altura:(imagem_linha-1)*altura+56,
                                (imagem_coluna-1)*largura+(imagem_coluna-1)//10:(imagem_coluna-1)*largura+(imagem_coluna-1)//10+46]
                plt.imshow(base3)
                df2=base3.stack().reset_index().rename(columns={'level_0':'Linha','level_1':'Coluna', 0:'Valores'})        
                df3=df3.append(df2['Valores'])            
        #reset dos índices
        df3 = df3.reset_index(drop=True)
        df4=df3.T.reset_index(drop=True)
        df4=df4.append(df4.iloc[0,:])
        for i in range(400):
            df4.iloc[2576,i]=i//10+1
        #df4 = df4.reset_index(drop=True)
        base=df4.T
    return base

# base recebe um arquivo com 400 linhas e 2577 colunas, sendo a ultima a classe do vetor de atributos
base=preparar_arquivo_base()
# para uma clusterizacao nao se precisa da classe, enstao tiramos ela
base2=base.iloc[:,:2576]
# faça a base um numpy
x=np.array(base2)

# retorna a similaridade, S(i,k) = -(Xi-Xk)^2.
def similarity(xi, xj):
    return -((xi - xj)**2).sum()

# retorna uma matriz com as similaridades (S) e 2 matrizes 
# nulas, Responsabilidade (R) e Disponibilidade (A), com dimensão N.

def create_matrices():
    S = np.zeros((x.shape[0], x.shape[0]))
    R = np.array(S)
    A = np.array(S)
    # laço 'for'que calcula a similaridade entre todas as combinações 2 a 2.
    for i in range(x.shape[0]):
        for k in range(x.shape[0]):
            S[i, k] = similarity(x[i], x[k])
    return A, R, S

# Cada valor r da matriz R será atualizado com
# r = r * damping + (1 - damping) * r_
# r=(1−λ)*r+λ*r_
# Onde r_(i,k)=s(i,k)−max{a(i,k')+s(i,k')}, com k'≠k
def update_r(damping=0.9):
    global R
    # For every column k, except for the column with the maximum value the max is the same.
    # So we can subtract the maximum for every row, 
    # and only need to do something different for k == argmax

    v = S + A
    rows = np.arange(x.shape[0])
    # We only compare the current point to all other points, 
    # so the diagonal can be filled with -infinity
    np.fill_diagonal(v, -np.inf)

    # max values
    idx_max = np.argmax(v, axis=1)
    first_max = v[rows, idx_max]

    # Second max values. For every column where k is the max value.
    v[rows, idx_max] = -np.inf
    second_max = v[rows, np.argmax(v, axis=1)]

    # Broadcast the maximum value per row over all the columns per row.
    max_matrix = np.zeros_like(R) + first_max[:, None]
    max_matrix[rows, idx_max] = second_max

    new_val = S - max_matrix

    R = R * damping + (1 - damping) * new_val

# a_(i,k) = min{0,r(k,k)+ ∑ max{0,r(i',k)}, com i'≠{i,k},sei≠k
# a_(i,k) = ∑ max{0,r(i',k)}, i'≠{k},sei=k
# a(i,k) = (1−λ)*a(i,k) + λ*a_(i,k)
def update_a(damping=0.9):
    global A

    k_k_idx = np.arange(x.shape[0])
    # set a(i, k)
    a = np.array(R)
    a[a < 0] = 0
    np.fill_diagonal(a, 0)
    a = a.sum(axis=0) # columnwise sum
    a = a + R[k_k_idx, k_k_idx]

    # broadcasting of columns 'r(k, k) + sum(max(0, r(i', k))) to rows.
    a = np.ones(A.shape) * a

    # For every column k, subtract the positive value of k. 
    # This value is included in the sum and shouldn't be
    a -= np.clip(R, 0, np.inf)
    a[a > 0] = 0

    # set(a(k, k))
    w = np.array(R)
    np.fill_diagonal(w, 0)

    w[w < 0] = 0

    a[k_k_idx, k_k_idx] = w.sum(axis=0) # column wise sum
    A = A * damping + (1 - damping) * a

################################
######## TESTE DO CÓDIGO #######
################################
    
# Criando as matrizes para a rede
A, R, S = create_matrices()
# preferencia é um atributo que diz quanto um nó pode ser seu representante 
# comparado com os outros, nesse exemplo escolhemos a preferencia com a mediana dos 
# valores de S
preference = np.median(S)
np.fill_diagonal(S, preference)

# Coeficiente de amortecimento (λ)
damping = 0.6

# para a condição de parada do laço for abaixo.
last_sol = np.ones(A.shape)
# laço de iterações de updates das matrizes
# ele termina quando tiver N iterações, onde N é o mesmo número de entradas, ou
# quando os exemplares se mantiverem iguais por 2 iterações.
for i in range(len(x)):
    update_r(damping)
    update_a(damping)
    sol = A + R
    exemplars = np.unique(np.argmax(sol, axis=1))
    if np.allclose(last_sol, sol):
        print(exemplars, i)
        break
    last_sol = sol

# guardando a relação entre os rostos equivalentes entre si
# esse vetor indica quais são as imagens mais semelhantes "exemplares" de cada imagem i.
vetor_rostos_equivalentes = []
sol2 = pd.DataFrame(sol)
for i in range(len(sol2)):
    #print(sol2.iloc[i,:].idxmax())
    vetor_rostos_equivalentes.append(sol2.iloc[i,:].idxmax())


# função que retorna do dataframe de uma foto dentre as 400 disponíveis no dataset.
# A ordem é da esqueda para a direita, de cima para baixo.
def retorne_rosto(numero=0): 
    arquivo="olivettifaces.gif"
    img = mpimg.imread(arquivo)
    base=pd.DataFrame(img[0:,0:,0])
    largura = 47
    altura = 57
    i=numero//20+1
    j=numero%20+1
    imagem_linha = i
    imagem_coluna = j
    base3=base.iloc[(imagem_linha-1)*altura:(imagem_linha-1)*altura+56,
                    (imagem_coluna-1)*largura+(imagem_coluna-1)//10:(imagem_coluna-1)*largura+(imagem_coluna-1)//10+46]
    #plt.imshow(base3)
    return base3.reset_index(drop=True)

#para visualizar o procedimento realizado com o algoritmo
clusters = []
len(clusters)
n_clusters=0
contagem = [0]
for i in range(400):
    #i=395
    indice_pessoa = i
    melhor_representante = vetor_rostos_equivalentes[i]
    
    if (melhor_representante not in clusters):
        clusters.append(melhor_representante)
        n_clusters = n_clusters+1
    bigdata = pd.concat([retorne_rosto(indice_pessoa) ,retorne_rosto(melhor_representante)], axis=1)
    plt.imshow(bigdata)
    plt.title('N Clusters:'+str(n_clusters)+' | Foto:'+str(indice_pessoa+1)+'/400 | Representante:'+str(melhor_representante+1)+"| Acertos:"+str(pd.DataFrame(contagem).sum()[0]))
    plt.show()
    if (indice_pessoa//10 == melhor_representante//10):
        contagem.append(1)
    else:
        contagem.append(0)

acertos = pd.DataFrame(contagem).sum()
print(len(exemplars),"Exemplares e acertos =",acertos[0])

# Resultado:66 Exemplares (clusters), de 40 desejados, e acertos = 357/400



################################
########  Melhor Damping #######
################################
# esse trecho facilita a escolha do melhor damping
def mse(xi, xj):
    xi = xi.T.reset_index(drop=True).T
    xj = xj.T.reset_index(drop=True).T
    return ((xi-xj)**2).sum().sum()


divisoes = 20
amortecimentos = pd.DataFrame(columns={'Damping','N clusters','Acertos','MSE'})
for damping in range(divisoes):
    #damping = 1
    print(damping,"/",divisoes)
    A, R, S = create_matrices()
    preference = np.median(S)
    np.fill_diagonal(S, preference)
    damping = damping/divisoes
    last_sol = np.ones(A.shape)
    last_exemplars = np.array([])
    clusters = []
    for i in range(len(x)):
        update_r(damping)
        update_a(damping)
        sol = A + R
        exemplars = np.unique(np.argmax(sol, axis=1))
        if np.allclose(last_sol, sol):
            #print(exemplars, i)
            break
        last_sol = sol
    acertos = 0
    somaMSE=0
    for i in range(400):
        #i=0
        #pd.DataFrame(sol).iloc[i,:].idxmax()
        #mse(retorne_rosto(i),retorne_rosto(pd.DataFrame(sol).iloc[i,:].idxmax()))
        #mse(retorne_rosto(0),retorne_rosto(4))
        somaMSE = somaMSE + mse(retorne_rosto(i),retorne_rosto(pd.DataFrame(sol).iloc[i,:].idxmax()))
        if(pd.DataFrame(sol).iloc[i,:].idxmax()//10==i//10):
           acertos=acertos+1
    #print("Damping =",damping,"n exemplares =",len(exemplars))
    amortecimentos = amortecimentos.append({'Damping': damping,'N clusters': len(exemplars),'Acertos': acertos,'MSE': somaMSE}, ignore_index=True)

t = amortecimentos['Damping']
data1 = amortecimentos['N clusters']
data2 = amortecimentos['Acertos']

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Amortecimento (λ)')
ax1.set_ylabel('N clusters', color=color)
ax1.plot(t, data1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

'''
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Acertos', color=color)  # we already handled the x-label with ax1
ax2.plot(t, data2, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
'''


data3 = amortecimentos['MSE']
ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'green'
ax3.set_ylabel('MSE', color=color)  # we already handled the x-label with ax1
ax3.plot(t, data3, color=color)
ax3.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
