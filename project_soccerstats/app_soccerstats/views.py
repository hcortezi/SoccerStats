from django.shortcuts import render, redirect
from .models import jogador_collection
from django.http import HttpResponse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


fw_features1 = ["Goals", "Shots", "SoT", "G/Sh", "G/SoT", "ShoDist", "GCA", "SCA", "Off", "PKwon", "ScaDrib", "Assists",
                    "ScaPassLive", "Car3rd", "ScaFld", "ToAtt", "ToSuc", "Carries", "CarTotDist", "CarPrgDist", 'CPA', "CarMis", "CarDis","PasTotCmp"]
mf_features1 = ["Goals","PasTotCmp", "PasTotCmp%", "PasTotDist", "PasTotPrgDist", "Assists", "PasAss", "Pas3rd", "Crs", "PasCmp",
                       "PasOff", "PasBlocks", "SCA", "ScaPassLive", "ScaPassDead", "ScaDrib", "ScaSh", "ScaFld", "GCA", "GcaPassLive",
                       "GcaPassDead", "GcaDrib", "GcaSh", "GcaFld", "Tkl", "TklWon", "TklDef3rd", "TklMid3rd", "TklAtt3rd", "TklDri",
                       "TklDriAtt", "TklDri%", "TklDriPast", "Blocks", "BlkSh", "Int", "Recov", "Carries", "CarTotDist", "CarPrgDist" , "Fld"]
df_features1 = ["PasTotCmp", "PasTotDist", "PasTotPrgDist", "Tkl", "TklWon", "TklDef3rd", "TklMid3rd", "TklAtt3rd", "TklDri", "TklDriAtt", "TklDriPast", "Blocks",
                     "BlkSh", "Int", "Tkl+Int", "Recov", "AerWon", "AerLost", "CarTotDist", "CarPrgDist", "CrdY", "CrdR","Fls", "Clr","Carries"
                     ,"TouDefPen","TouDef3rd","TouMid3rd","TouAtt3rd","TouAttPen","Assists"]
dfmf_features1 = ["PasTotCmp", "PasTotDist", "PasTotPrgDist", "Tkl", "TklWon", "TklDef3rd", "TklMid3rd", "TklAtt3rd", "TklDri", "TklDriAtt", "TklDriPast", "Blocks", "BlkSh", "Int", "Tkl+Int", "Recov", "AerWon", "AerLost", "Carries", "CarTotDist", "CarPrgDist", "CrdY", "CrdR", "Fls", "Clr", "TouDefPen", "TouDef3rd", "TouMid3rd", "TouAtt3rd", "TouAttPen", "GCA", "GcaPassLive", "GcaPassDead", "GcaDrib"]
mfdf_features1 = ["Goals", "Shots", "SoT", "G/Sh", "G/SoT", "ShoDist", "GCA", "SCA", "Off", "PKwon", "ScaDrib", "Assists", "PasAss", "Pas3rd", "Crs", "PasCmp", "PasOff", "PasBlocks", "ScaPassLive", "ScaPassDead", "ScaSh", "ScaFld", "GcaPassLive", "GcaPassDead", "GcaDrib", "GcaSh", "GcaFld", "Tkl"]
dffw_features1 = ["PasTotCmp", "PasTotDist", "PasTotPrgDist", "Tkl", "TklWon", "TklDef3rd", "TklMid3rd", "TklAtt3rd", "TklDri", "TklDriAtt", "TklDriPast", "Blocks", "BlkSh", "Int", "Tkl+Int", "Recov", "AerWon", "AerLost", "Carries", "CarTotDist", "CarPrgDist", "CrdY", "CrdR", "Fls", "Clr", "TouDefPen", "TouDef3rd", "TouMid3rd", "TouAtt3rd", "TouAttPen", "Assists", "Goals", "Shots", "SoT", "G/Sh", "G/SoT", "ShoDist", "GCA", "SCA", "Off", "PKwon", "ScaDrib", "PasAss", "Pas3rd"]
fwmf_features1 = ["Goals", "Shots", "SoT", "G/Sh", "G/SoT", "ShoDist", "Off", "PKwon", "Assists",
                  "Car3rd", "ToAtt", "ToSuc", "Carries", "CarTotDist", "CarPrgDist", 'CPA', "CarMis", "CarDis", "PasTotCmp","PasAss", "Pas3rd", "Crs", "PasCmp",
                       "PasOff", "PasBlocks", "SCA", "ScaPassLive", "ScaPassDead", "ScaDrib", "ScaSh", "ScaFld", "GCA", "GcaPassLive",
                       "GcaPassDead", "GcaDrib", "GcaSh", "GcaFld", "Tkl", "TklWon", "TklDef3rd", "TklMid3rd", "TklAtt3rd", "TklDri",
                       "TklDriAtt", "TklDri%"]
fwdf_features1=["Goals", "Shots", "SoT", "G/Sh", "G/SoT", "ShoDist", "GCA", "SCA", "Off", "PKwon", "Assists", "ScaPassLive", "Car3rd",
                "ScaFld", "ToAtt", "ToSuc", "Carries", "CarTotDist", "CarPrgDist", 'CPA', "CarMis", "CarDis", "PasTotCmp", "PasAss", "Pas3rd",
                "Crs", "PasCmp", "PasOff", "PasBlocks", "ScaPassDead", "ScaDrib", "ScaSh", "GcaPassDead", "GcaDrib", "GcaSh", "GcaFld", "Tkl",
                "TklWon", "TklDef3rd", "TklMid3rd", "TklAtt3rd", "TklDri", "TklDriAtt", "TklDri%"]
gk_features1 = ["PasTotCmp", "PasTotCmp%", "PasTotDist", "PasTotPrgDist", "Assists", "PasAss", "Pas3rd", "Crs", "PasCmp",
                       "PasOff", "PasBlocks", "SCA", "ScaPassLive", "ScaPassDead", "ScaDrib", "ScaSh", "ScaFld", "GCA", "GcaPassLive",
                       "GcaPassDead", "GcaDrib", "GcaSh", "GcaFld", "Tkl", "TklWon", "TklDef3rd", "TklMid3rd", "TklAtt3rd", "TklDri",
                       "TklDriAtt", "TklDri%", "TklDriPast", "Blocks", "BlkSh", "Int"]
mffw_features1= ["Goals", "Shots", "SoT", "G/Sh", "G/SoT", "ShoDist", "SCA", "Off", "PKwon", "ScaDrib", "Assists",
                 "Car3rd", "ScaFld", "ToAtt", "ToSuc", "Carries", "CarTotDist", "CarPrgDist", 'CPA', "CarMis", "CarDis","PasTotCmp","PasAss", "Pas3rd", "Crs", "PasCmp",
                       "PasOff", "PasBlocks", "ScaPassLive", "ScaPassDead", "ScaSh", "GCA", "GcaPassLive",
                       "GcaPassDead", "GcaDrib", "GcaSh", "GcaFld", "Tkl", "TklWon", "TklDef3rd", "TklMid3rd", "TklAtt3rd", "TklDri",
                       "TklDriAtt", "TklDri%"]


def features_por_posicao(posicao):
    if posicao == 'FW':
        return fw_features1
    elif posicao == 'MF':
        return mf_features1
    elif posicao == 'DF':
        return df_features1
    elif posicao == 'DFMF':
        return dfmf_features1
    elif posicao == 'MFDF':
        return mfdf_features1
    elif posicao == 'DFFW':
        return dffw_features1
    elif posicao == 'FWMF':
        return fwmf_features1
    elif posicao == 'FWDF':
        return fwdf_features1
    elif posicao == 'GK':
        return gk_features1
    elif posicao== 'MFFW':
      return mffw_features1
    else:
        return None

#Retorna todo o um cluster específico de uma posição a partir de um jogador específico
def filtragem_pos_clus(nome_jogador, df):
    dados_jogador = df[df['Player'] == nome_jogador]
                       
    posicao_jogador = dados_jogador['Pos'].iloc[0]
    cluster_jogador = dados_jogador['Cluster'].iloc[0]

    jogadores_filtrados = df[(df['Pos'] == posicao_jogador) & (df['Cluster'] == cluster_jogador)]
    features = features_por_posicao(posicao_jogador)

    return jogadores_filtrados, features,dados_jogador


#Cálculo dos jogadores mais próximos de um ponto, de acordo com posição de cluster, usando Nearest Neighbors
def calculo_jogadores_recomendados(nome_jogador, df, metric):
    jogadores_filtrados, features, dados_jogador = filtragem_pos_clus(nome_jogador, df)

    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(jogadores_filtrados[features])

    nbrs = NearestNeighbors(n_neighbors=10, algorithm="auto", metric=metric)
    nbrs.fit(dados_padronizados)

    dados_jogador_padronizados = scaler.transform(dados_jogador[features])

    distancias, indices = nbrs.kneighbors(dados_jogador_padronizados)

    indices_jogadores_recomendados = indices[0]
    jogadores_recomendados = jogadores_filtrados.iloc[indices_jogadores_recomendados]

    nomes_jogadores_recomendados = jogadores_recomendados['Player'].tolist()

    return nomes_jogadores_recomendados


def home(request):
    query = request.POST.get('query')
    listaJogadores = []

    if query:
        listaJogadores = jogador_collection.find({"Player": query})
    else:
        listaJogadores = jogador_collection.find({})

    context = {"jogadores": listaJogadores}
    return render(request, 'home.html', context)

def details(request, id):
    jogador = jogador_collection.find_one({"Rk": int(id)})
    df = pd.read_csv("C:\\Users\\hcort\\SoccerStats\\dfSoccerStats2.csv", sep=';', encoding="utf-8")

    if jogador:
        jogadores_recomendados_nomes = calculo_jogadores_recomendados(jogador['Player'], df, "manhattan")
        jogadores_recomendados_dados = []
        
        # Comece o loop a partir da segunda posição para pular a primeira
        for nome_jogador in jogadores_recomendados_nomes[1:]:
            jogador_recomendado = jogador_collection.find_one({"Player": nome_jogador})
            if jogador_recomendado:
                jogadores_recomendados_dados.append(jogador_recomendado)
        
        context = {"jogador": jogador, "jogadores": jogadores_recomendados_dados}
        return render(request, 'details.html', context)
    else:
        return HttpResponse("Player not found", status=404)