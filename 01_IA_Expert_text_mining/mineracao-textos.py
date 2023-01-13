import nltk

nltk.download('stopwords')
nltk.download('rslp')

base = [('eu sou admirada por muitos','alegria'),
        ('me sinto completamente amado','alegria'),
        ('amar e maravilhoso','alegria'),
        ('estou me sentindo muito animado novamente','alegria'),
        ('eu estou muito bem hoje','alegria'),
        ('que belo dia para dirigir um carro novo','alegria'),
        ('o dia est� muito bonito','alegria'),
        ('estou contente com o resultado do teste que fiz no dia de ontem','alegria'),
        ('o amor e lindo','alegria'),
        ('nossa amizade e amor vai durar para sempre', 'alegria'),
        ('estou amedrontado', 'medo'),
        ('ele esta me ameacando a dias', 'medo'),
        ('isso me deixa apavorada', 'medo'),
        ('este lugar e apavorante', 'medo'),
        ('se perdermos outro jogo seremos eliminados e isso me deixa com pavor', 'medo'),
        ('tome cuidado com o lobisomem', 'medo'),
        ('se eles descobrirem estamos encrencados', 'medo'),
        ('estou tremendo de medo', 'medo'),
        ('eu tenho muito medo dele', 'medo'),
        ('estou com medo do resultado dos meus testes', 'medo')]

stopwords = ['a', 'agora', 'algum', 'alguma', 'aquele', 'aqueles', 'de', 'deu', 'do', 'e', 'estou', 'esta', 'esta',
             'ir', 'meu', 'muito', 'mesmo', 'no', 'nossa', 'o', 'outro', 'para', 'que', 'sem', 'talvez', 'tem', 'tendo',
             'tenha', 'teve', 'tive', 'todo', 'um', 'uma', 'umas', 'uns', 'vou']



def remove_stop_words(text):
    sentences = []
    for (words, emotions) in text:
        not_stopword = [p for p in words.split() if p not in stop_words_nltk]

        sentences.append((not_stopword, emotions))

    return sentences

#print(remove_stop_words(base))

# stopwords NLTK
stop_words_nltk = nltk.corpus.stopwords.words('portuguese')
#print(stop_words_nltk)

# remocao de radicais
def aplicastemmer(texto):
    stemmer = nltk.stem.RSLPStemmer()
    frasesstemming = []
    for (palavras, emocao) in texto:
        comstemming = [str(stemmer.stem(p)) for p in palavras.split() if p not in stop_words_nltk]
        frasesstemming.append((comstemming, emocao))
    return frasesstemming

frasescomstemming = aplicastemmer(base)
#print(frasescomstemming)

#buscando as palavras
def buscapalavras(frases):
    todaspalavras = []
    for (palavras, emocao) in frases:
        todaspalavras.extend(palavras)
    return todaspalavras

palavras = buscapalavras(frasescomstemming)

#Quantidade de palavras
def buscafrequencia(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras

frequencia = buscafrequencia(palavras)
#print(frequencia.most_common(50))

# com base na frequencia, extrair a lista final sem repeticao

def buscapalavrasunicas(frequencia):
    freq = frequencia.keys()
    return freq

palavrasunicas = buscapalavrasunicas(frequencia)
#print(palavrasunicas)

def extratorpalavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavra in palavrasunicas:
        caracteristicas['%s' % palavra] = (palavra in doc)
    return caracteristicas

caracteristicasfrase = extratorpalavras(['tim', 'gole', 'nov'])
#print(caracteristicasfrase)

basecompleta = nltk.classify.apply_features(extratorpalavras, frasescomstemming)

# constroi a tabela de probabilidade
classificador = nltk.NaiveBayesClassifier.train(basecompleta)
#print(classificador.labels())
#print(classificador.show_most_informative_features(5))

# realizando testes
teste = 'estou com amor'
teste_stemming = []
stemmer = nltk.stem.RSLPStemmer()

for (palavras) in teste.split():
    comstem = [p for p in palavras.split()]
    teste_stemming.append(str(stemmer.stem(comstem[0])))
#print(teste_stemming)

novo = extratorpalavras(teste_stemming)
#print(novo)

#print(classificador.classify(novo))

#retornando a probabilidade
distribuicao = classificador.prob_classify(novo)
for classe in distribuicao.samples():
    print("%s : %f" % (classe, distribuicao.prob(classe)))