<H1>Mineração de Emoção em Textos com Python e NLTK</H1>


---

<h2>Abordagens de mineração de textos</h2>

- Estatística

- Processamento de Linguagem Natural

---

<h2>Termos e expressões</h2>

- Valência
Frase positiva ou negativa

---

<h2>Step by Step </h2>

1. Realizar pré-processamento da base - substituir caracteres especiais, espaços no início e fim, símbolos, etc.

2. Remoção das stop words
    2.1. NLTK stopwords

    ```python
    nltk.corpus.stopwords.words('portuguese')
    ```

3. Remoção de radicais

4. Criação de lista com as palavras

5. Criação de lista com as palavras - distinct
    5.1 Busca da frequência das palavras

    ```python
    def buscafrequencia(palavras):
        palavras = nltk.FreqDist(palavras)

        return palavras

    frequencia = buscafrequencia(palavras)
    
    print(frequencia.most_common(50))
    ```

6. Criando a tabela com cada palavra e True ou False se a palavra (coluna) contém na frase em questão

7. Para a base inteira, é possível utilizar função do NLTK ``` nltk.classify.apply_features(parametro1, parametro2) ```

---
Procurar por:

- Correlação entre as palavras
