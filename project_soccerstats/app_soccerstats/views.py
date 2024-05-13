from django.shortcuts import render, redirect
from .models import jogador_collection
from django.http import HttpResponse

# Create your views here.
def home(request):
    query = request.POST.get('query')
    listaJogadores = []

    if query == '':
        for jogador in jogador_collection.find({}):
            listaJogadores.append(jogador)
    else:
        for jogador in jogador_collection.find({"nome": query}):
            listaJogadores.append(jogador)

    context = {"jogadores": listaJogadores}
    return render(request, 'home.html', context)

def details(request, id):
    jogador = jogador_collection.find({"id": id})

    listaJogadores = []
    for jogadorR in jogador_collection.find({}):
        listaJogadores.append(jogadorR)

    context = {"jogador": jogador[0], "jogadores": listaJogadores}
    return render(request, 'details.html', context)