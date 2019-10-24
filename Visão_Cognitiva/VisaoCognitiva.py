# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 13:35:01 2019

@author: amanoels
"""
# Import da Biblioteca que chama a API Cognitive Face MicroSoft
import cognitive_face as CF

# Sua chave de subcrição e sua 
SUBSCRIPTION_KEY = 'YOUR_SUBSCRIPTION_KEY'

# Enderço da API 
BASE_URL = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0'

# Criaçõa de grupo conhecido de pessoas
PERSON_GROUP_ID = 'known-persons'

# Armazenamento das variaveis
CF.BaseUrl.set(BASE_URL)
CF.Key.set(SUBSCRIPTION_KEY)

# Criação dos ID's para as pessoas conhecidas no grupo
CF.person_group.create(PERSON_GROUP_ID, 'Known Persons')

# Passagem de dados da pessoa
name = "Joao Ninguem"
user_data = 'Usuário de Testes'

# Armazenamento dos dados da pessoa com o ID
response = CF.person.create(PERSON_GROUP_ID, name, user_data)

# Obtenção do ID gerado pela API
person_id = response['personId']

# Passando a face a ser usada como modelo
CF.person.add_face('joao.jpg', PERSON_GROUP_ID, person_id)

# Lista de ID's treinados
CF.person.lists(PERSON_GROUP_ID)

# Passagem de novos rostos
# Treino o grupo fazendo a API reconhecer os pontos referentes a imagem obtida
CF.person_group.train(PERSON_GROUP_ID)

# Gero pela API o Status do treinamento
response = CF.person_group.get_status(PERSON_GROUP_ID)

# Armazeno o Status
status = response['status']

# Apresento o Status
status

# Gero dados de confronto com a base treinada ('Test.jpg') é a imagem a ser usada
# como parametro para analisar a foto original
response = CF.face.detect('test.jpg')

# Laço para identificar o faceId na imagem de teste para cada rosto encontrado na imagem
face_ids = [d['faceId'] for d in response]

# Apresento os faceId's gerados (deve ser 1 por rosto)
face_ids

# Utilizo a API para identificar o rosto vs o rosto usado para o treino
identified_faces = CF.face.identify(face_ids, PERSON_GROUP_ID)

# Apresento o resultado em Json com a confiança. 
identified_faces








