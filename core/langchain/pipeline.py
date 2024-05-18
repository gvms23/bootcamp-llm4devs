from loader import load
from transformer import transform
from embedder import embed_doc, embed_docs
from storer import store, retrieve

dados = [
    "https://pt.wikipedia.org/wiki/Pindamonhangaba"
]

print('carregando...')
dados_carregados = []
for dado in dados:
    dados_carregados.append(load(dados))
print('ok...')

print('transformando...')
dados_transformados = transform(dados_carregados[0][0].page_content)
print('ok...')
# print(dados_transformados)
# print(len(dados_transformados))

print('embedding...')
dados_embeddados = embed_docs(dados_transformados)
print('ok...')
        
# print(len(dados_embeddados))

print('storing...')
stored = store(dados_transformados, dados_embeddados)
print('ok')

print('querying...')
query = "quais os principais destinos em Pindamonhangaba"
chunks = retrieve(embed_doc(query))
print("--------------")
print("chunks")
print(chunks)