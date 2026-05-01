# TnyThinker — Aplicação Prática dos Conceitos de LLM (Transformer, Attention e GPT) 

POR BRENO CAMPOS RIBEIRO

## Introdução {#introdução}

Meu objetivo com este projeto é desenvolver um modelo de linguagem simples, leve e rápido, utilizando conjuntos de dados mais enxutos e eficientes. Há alguns meses, publiquei em [meu blog um artigo](https://brenocamposribeiro.com.br/blog/7) apresentando os conceitos de Transformers, Attention e GPT, acompanhado de exemplos simples em Python para demonstrar a teoria por trás dessa tecnologia. Agora, apresento aqui uma aplicação real e escalável de uma LLM (Large Language Model). 

## Transformers e Attention {#transformers-e-attention}

Antes do surgimento dos Transformers, em 2017, a arquitetura dominante para tarefas de geração de texto era baseada em Redes Neurais Recorrentes (RNNs). Essas redes processavam informações de maneira sequencial, palavra por palavra, mantendo um estado interno que representava o contexto da sequência.

As variações mais conhecidas e eficientes dessa abordagem eram:

* **LSTM (Long Short-Term Memory):** projetada para reduzir o problema de esquecimento de informações em sequências longas, permitindo que o modelo preserve o contexto por mais tempo.  
* **GRU (Gated Recurrent Unit):** uma versão mais simples e rápida das LSTMs, com menos parâmetros e desempenho semelhante em muitos cenários.  
* **Seq2Seq (Sequence-to-Sequence):** arquitetura composta por um *Encoder*, responsável por interpretar o texto de entrada, e um *Decoder*, responsável por gerar a saída. Foi amplamente utilizada em sistemas de tradução automática, como versões antigas do Google Tradutor.

O principal problema dessas arquiteturas era o processamento estritamente sequencial. Como cada palavra dependia do processamento da anterior, o treinamento e a inferência se tornavam mais lentos. Além disso, modelos recorrentes tinham dificuldade em capturar relações entre palavras muito distantes dentro de um texto longo, limitando a qualidade do contexto compreendido pelo modelo.

Como já publiquei um conteúdo completo abordando esses conceitos, não vou me aprofundar tanto aqui. Ainda assim, vale contextualizar brevemente: Transformers é uma arquitetura de redes neurais profundas que utiliza o mecanismo de Attention para compreender relações entre palavras, mesmo quando estão distantes dentro de um texto.

Na prática, imagine a frase: “o gato está no…”. Se o modelo foi treinado com exemplos como “o gato está no telhado” e “o gato está no chão”, a probabilidade de prever palavras como “telhado” ou “chão” será muito maior do que prever algo sem relação contextual, como “banana”.

Esse mecanismo permite que o modelo analise palavras em paralelo, mantenha o contexto ao longo da sequência e produza respostas mais coerentes e estáveis, especialmente conforme aumenta a quantidade de tokens processados.

![][image1]  
https://www.comet.com/site/blog/explainable-ai-for-transformers/

## **Core** {#core}

Basicamente, a estrutura do projeto é organizada de forma bastante direta. Na pasta **core**, temos o “coração” de todo o sistema: os arquivos **tokenizer.py** e **model.py**. Esses dois componentes são os pilares centrais de um modelo de linguagem.

O **tokenizer.py** é responsável por transformar texto em *tokens*, convertendo palavras e caracteres em representações numéricas que podem ser processadas pela rede neural. Já o **model.py** contém a arquitetura do modelo em si, definindo como os dados serão processados, como o mecanismo de *Attention* será aplicado e como as próximas palavras serão previstas durante a geração de texto.

### 

## **Tokenizer (tokenizer.py)** {#tokenizer-(tokenizer.py)}

O tokenizer.py funciona como o “tradutor” entre a linguagem humana e os valores numéricos que o modelo consegue processar. Em outras palavras, ele é responsável por converter texto em tokens numéricos e, posteriormente, reconstruir esses tokens em texto legível.

Esse processo é fundamental em qualquer modelo de linguagem, já que redes neurais não entendem palavras diretamente — apenas números.

### **1. Mapeamento (stoi e itos)** {#1.-mapeamento-(stoi-e-itos)}

O tokenizer mantém dois dicionários principais:

* **stoi (String to Index):** converte palavras em índices numéricos.  
   Exemplo: "casa" → 42  
* **itos (Index to String):** realiza o processo inverso, transformando índices novamente em palavras.  
   Exemplo: 42 → "casa"

Esses mapeamentos formam o vocabulário do modelo.

### **2. Tokens Especiais** {#2.-tokens-especiais}

O código define alguns tokens especiais que funcionam como estruturas auxiliares durante o treinamento e geração de texto:

* **<pad>**: utilizado para preencher sequências menores e manter todas as entradas com o mesmo tamanho.  
* **<unk>**: representa palavras desconhecidas que não existem no vocabulário.  
* **<nl>**: representa quebras de linha, permitindo que o modelo aprenda estruturação e formatação de texto.

### **3. O método _tokenize (quebra de texto)** {#3.-o-método-_tokenize-(quebra-de-texto)}

O método _tokenize utiliza expressões regulares (re.compile) para fragmentar o texto em partes menores.

Ele separa:

* palavras;  
* números;  
* pontuações;  
* quebras de linha.

Além disso, o texto é convertido para minúsculas com text.lower(), garantindo que palavras como "Casa" e "casa" sejam tratadas da mesma forma.

### **4. encode e decode** {#4.-encode-e-decode}

Esses são os métodos responsáveis pela conversão entre texto e tokens:

* **encode**: recebe uma frase e retorna uma lista de IDs numéricos que o modelo consegue processar.  
* **decode**: recebe os tokens gerados pelo modelo e reconstrói o texto legível.

### **Exemplo de encode e decode** {#exemplo-de-encode-e-decode}

Imagine que temos a seguinte frase:
```
texto = "Olá, tudo bem?" 
```
Quando usamos o método encode, o tokenizer converte cada token em um identificador numérico: 
```
tokens = tokenizer.encode(texto)
```
print(tokens)

Saída hipotética: 
```
[15, 8, 42, 91, 7] 
```
Nesse caso:

* 15 = "olá"  
* 8 = ","  
* 42 = "tudo"  
* 91 = "bem"  
* 7 = "?"

Depois que o modelo processa esses números, podemos reconstruir o texto utilizando o método decode: 
```
texto_reconstruido = tokenizer.decode(tokens)

print(texto_reconstruido)  
```
Resultado: Olá, tudo bem? 

Esse processo é essencial porque o modelo trabalha apenas com números internamente. O tokenizer é responsável por fazer a ponte entre o texto humano e as representações numéricas utilizadas pela rede neural. 

### **5. O método _detokenize (reconstrução textual)** {#5.-o-método-_detokenize-(reconstrução-textual)}

O _detokenize é responsável pela etapa final de reconstrução do texto.

Ele remove espaçamentos desnecessários antes de pontuações e ajusta a formatação para tornar a saída mais natural.

**Exemplo:**
```
["olá", ",", "tudo", "bem", "?"]
``` 

é convertido para:

"olá, tudo bem?"

# **Model (model.py)** {#model-(model.py)}

O model.py define a arquitetura principal do modelo de linguagem baseado em Transformers, seguindo uma estrutura semelhante à utilizada em modelos da família GPT.

É nesse arquivo que o “cérebro” da IA é construído.

## **1. GPTConfig** {#1.-gptconfig}

A classe GPTConfig centraliza os hiperparâmetros do modelo em um único local, facilitando a configuração e manutenção.

Os principais parâmetros são:

* **vocab_size**: tamanho do vocabulário conhecido pela IA.  
* **d_model**: dimensão dos vetores internos que representam cada token.  
* **n_layers**: quantidade de camadas Transformer empilhadas.  
* **max_seq_len**: tamanho máximo da sequência de tokens processada de uma vez.

## **2. GPTModel (Arquitetura Principal)** {#2.-gptmodel-(arquitetura-principal)}

A classe GPTModel implementa a arquitetura do modelo utilizando PyTorch.

### **token_emb** {#token_emb}

Transforma os IDs dos tokens em vetores densos (*embeddings*), permitindo que palavras semanticamente parecidas tenham representações próximas.

### **pos_emb** {#pos_emb}

Como Transformers processam tokens em paralelo, eles não possuem noção natural de ordem. A camada de *positional embedding* adiciona informações sobre a posição de cada token na sequência.

### **transformer** {#transformer}

É o núcleo do modelo. Utilizando nn.TransformerEncoder, essa camada aplica o mecanismo de Attention para analisar relações contextuais entre os tokens.

### **head** {#head}

A camada final converte os vetores internos do modelo em probabilidades sobre o vocabulário, permitindo prever qual será o próximo token da sequência.

## **3. O método forward** {#3.-o-método-forward}

O método forward define o fluxo completo de processamento dos dados dentro do modelo.

### **Soma dos embeddings** {#soma-dos-embeddings}

Primeiramente, o modelo combina:

* o embedding do token;  
* o embedding posicional.

Assim, cada token carrega tanto significado semântico quanto informação de posição.

### **Criação da máscara causal (*Causal Mask*)** {#criação-da-máscara-causal-(causal-mask)}

A função torch.triu(...) cria uma máscara triangular que impede o modelo de acessar tokens futuros durante o treinamento.

Esse mecanismo é essencial em modelos generativos como GPT, pois força a rede a prever a próxima palavra utilizando apenas o contexto anterior.

### **Processamento pelo Transformer** {#processamento-pelo-transformer}

Após a aplicação da máscara, os dados percorrem todas as camadas Transformer, onde o mecanismo de Attention aprende relações contextuais entre os tokens.

### **Saída do modelo** {#saída-do-modelo}

Por fim, o modelo retorna os *logits*, que representam as probabilidades associadas a cada palavra do vocabulário.

Esse arquivo é o mesmo que está sendo importado no app.py através de core.model para carregar os arquivos .pt.

Isso é importante porque o PyTorch precisa reconstruir exatamente a mesma arquitetura utilizada durante o treinamento para conseguir carregar corretamente os pesos salvos do modelo.

# **Config (config.py)** {#config-(config.py)}

## O config.py funciona como o centro de controle técnico do projeto. É nele que ficam definidas tanto as configurações de execução quanto os parâmetros responsáveis pelo comportamento do modelo durante a geração de texto. {#o-config.py-funciona-como-o-centro-de-controle-técnico-do-projeto.-é-nele-que-ficam-definidas-tanto-as-configurações-de-execução-quanto-os-parâmetros-responsáveis-pelo-comportamento-do-modelo-durante-a-geração-de-texto.}

## 1. Localização do Modelo e Hardware  {#1.-localização-do-modelo-e-hardware}

### **MODEL_PATH** {#model_path}

Define o caminho padrão onde o arquivo de pesos do modelo (.pt) está armazenado.

Exemplo:
```
MODEL_PATH = "models/model.pt" 
```
### **DEVICE** {#device}

Detecta automaticamente qual hardware será utilizado para executar o modelo.

Se houver uma GPU compatível com CUDA disponível, o PyTorch utilizará a placa de vídeo para acelerar o processamento. Caso contrário, o sistema utilizará a CPU.

Exemplo:
```
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
```
# **Parâmetros de Geração (GEN_PARAMS)** {#parâmetros-de-geração-(gen_params)}

Os parâmetros de geração funcionam como “controles de comportamento” da IA, influenciando criatividade, coerência e diversidade das respostas.

---

### **max_new_tokens** {#max_new_tokens}

Define o tamanho máximo da resposta gerada pelo modelo.
```
max_new_tokens = 200   
```
Nesse caso, o modelo poderá gerar até 200 tokens por resposta. 

### **temperature** {#temperature}

Controla o nível de criatividade da geração.
```
temperature = 1.0 
```
* Valores baixos (0.2, 0.5) tornam as respostas mais previsíveis e objetivas.  
* Valores altos (1.2, 1.5) aumentam a aleatoriedade e criatividade.

Quanto maior a temperatura, maior a diversidade das respostas — mas também maior a chance de incoerências.

### **top_k** {#top_k}

Limita a escolha do próximo token às palavras mais prováveis.
```
top_k = 100 
```
Nesse exemplo, o modelo considera apenas as 100 palavras com maior probabilidade antes de selecionar a próxima.

Isso ajuda a evitar escolhas totalmente aleatórias ou sem sentido.

### **top_p** {#top_p}

Também conhecido como *Nucleus Sampling*.
```
top_p = 0.9 
```
O modelo seleciona o menor conjunto de palavras cuja soma das probabilidades atinge 90%.

Essa técnica mantém a geração mais natural e fluida, reduzindo respostas excessivamente aleatórias.

### **repetition_penalty** {#repetition_penalty}

Penaliza palavras que já apareceram anteriormente na geração.
```
repetition_penalty = 1.0
```
* 1.0 → sem penalidade adicional.  
* Valores maiores (1.1, 1.2) reduzem repetições excessivas.

Esse parâmetro é útil para evitar loops e frases repetitivas.

### **frequency_penalty** {#frequency_penalty}

Desencoraja o uso frequente das mesmas palavras ao longo da resposta.
```
frequency_penalty = 0.05  
```
Mesmo sendo uma penalidade leve, ela ajuda o modelo a utilizar um vocabulário mais variado e menos repetitivo durante a conversa. 

# **Treinando o Modelo** {#treinando-o-modelo}

### Para realizar o treinamento e o *fine tuning* do modelo, foram desenvolvidos dois arquivos Jupyter Notebook que podem ser executados tanto no Google Colab quanto no Kaggle. {#stop_loss-=-1.0}

### Na pasta /train, temos: {#stop_loss-=-1.0}

* ### train.ipynb {#stop_loss-=-1.0}

* ### fine_tuning.ipynb {#stop_loss-=-1.0}

### O arquivo train.ipynb mantém praticamente a mesma estrutura principal do projeto — utilizando model.py e tokenizer.py — porém, o foco aqui está no loop de treinamento, que é responsável pelo processo de aprendizado da IA. {#stop_loss-=-1.0}

### Além do treinamento em si, o código inclui mecanismos de controle para evitar problemas comuns em modelos pequenos, principalmente o *overfitting*, quando a IA deixa de aprender padrões gerais e passa apenas a memorizar os textos do dataset. {#stop_loss-=-1.0}

# **1. Parada de Emergência (STOP_LOSS)** {#1.-parada-de-emergência-(stop_loss)}

### Durante o treinamento, o código monitora continuamente o valor de erro (*loss*) do modelo. {#stop_loss-=-1.0}

### Caso o *loss* fique abaixo de 1.0, o treinamento é interrompido automaticamente. {#caso-o-loss-fique-abaixo-de-1.0,-o-treinamento-é-interrompido-automaticamente.}

### STOP_LOSS = 1.0 {#stop_loss-=-1.0}

Em modelos menores, um *loss* excessivamente baixo geralmente indica que a rede começou a memorizar frases específicas do dataset em vez de aprender padrões linguísticos gerais.

Isso costuma resultar em respostas repetitivas, robóticas e pouco naturais durante o uso do modelo.

# **2. Checkpoints Automáticos (SAVE_AT_LOSS)** {#2.-checkpoints-automáticos-(save_at_loss)}

O treinamento também possui um sistema automático de checkpoints.

O modelo é salvo sempre que atinge determinados níveis de qualidade previamente definidos.

Exemplo:
```
SAVE_AT_LOSS = [2.5, 2.0, 1.8, 1.5]
```
Isso garante que versões intermediárias do modelo sejam preservadas ao longo do treinamento.

A principal vantagem é permitir recuperar versões que estavam apresentando melhor desempenho antes de um possível overfitting causado por treino excessivo.

Os pesos são salvos em arquivos .pt.

# **3. O Ciclo de Aprendizado (*Forward & Backward*)** {#3.-o-ciclo-de-aprendizado-(forward-&-backward)}

Dentro do loop principal de treinamento acontece o fluxo central de aprendizado do PyTorch.

## **Forward** {#forward}

O modelo recebe uma sequência de tokens e tenta prever qual será o próximo token.

## **Loss** {#loss}

O sistema compara a previsão do modelo com o texto real do dataset e calcula o erro da previsão.

Quanto menor o *loss*, melhor o desempenho do modelo.

## **Backward** {#backward}

Na etapa de *backpropagation*, o otimizador ajusta os pesos neurais para reduzir o erro nas próximas previsões.

Esse processo é o que efetivamente faz o modelo “aprender”.

## **Gradient Clipping** {#gradient-clipping}

O código também utiliza clip_grad_norm_, técnica que limita o tamanho dos gradientes durante o treinamento.

Isso evita problemas conhecidos como *gradient explosion*, onde valores extremamente altos podem desestabilizar completamente o treinamento.

Exemplo:
```
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```
# **4. Monitoramento Visual do Treinamento** {#4.-monitoramento-visual-do-treinamento}

Durante o treinamento, o sistema exibe alertas baseados no valor atual do *loss*, permitindo acompanhar a qualidade do aprendizado em tempo real.

### **LOSS > 3.5** {#loss->-3.5}

APRENDENDO  
O modelo ainda está confuso e aprendendo padrões básicos da linguagem. 

### **LOSS > 1.5 e 2.5** {#loss->-1.5-e-2.5}

IDEAL

Essa faixa costuma representar o melhor equilíbrio entre coerência e criatividade.

O modelo compreende contexto sem ficar excessivamente preso ao dataset.

### **LOSS < 1.0** {#loss-<-1.0}

OVERFIT 

Indica que o modelo provavelmente começou a memorizar os dados do treinamento, reduzindo sua capacidade de generalização.

# **5. Geração de Amostras em Tempo Real** {#5.-geração-de-amostras-em-tempo-real}

A cada determinado número de etapas (LOG_EVERY), o código gera pequenas amostras de texto automaticamente.

Exemplo:
```
LOG_EVERY = 100 
```
Durante o treino, o sistema pode gerar prompts simples como: 
```
"o que é java?" 
```
O objetivo dessas amostras é acompanhar visualmente a evolução do modelo em tempo real, verificando se:

* o português está ficando mais coerente;  
* as respostas fazem sentido;  
* ou se o modelo começou a degradar a qualidade do texto.

# **Pré-Treinamento com Dados Brutos** {#pré-treinamento-com-dados-brutos}

Para iniciar o treinamento do modelo, basta definir o caminho do dataset na seção **“Carregar Dados”** utilizando a variável DATA_PATH.

Exemplo:
```
DATA_PATH = "/kaggle/input/my-model-2/pytorch/default/1/texto_exemplo.txt"
```
No meu caso, o dataset de pré-treinamento foi construído a partir do scraping de aproximadamente 250 artigos da Wikipédia sobre temas variados, principalmente filosofia e história. Todo esse conteúdo foi consolidado em um único arquivo .txt.

Durante a execução do treinamento, os logs exibem informações sobre:

* progresso das épocas (*epochs*);  
* valor atual do *loss*;  
* pequenos testes de geração de texto ao longo do processo.

No início do treinamento, é normal que:

* o *loss* seja muito alto;  
* o modelo gere textos completamente aleatórios;  
* apareçam sequências sem sentido ou apenas combinações desconexas de letras.

Conforme o treinamento avança e o *loss* diminui, o modelo começa gradualmente a:

* estruturar frases;  
* aprender padrões sintáticos;  
* gerar textos mais organizados;  
* entender parcialmente o contexto das palavras.

Mesmo que o texto ainda não faça muito sentido semanticamente, já é possível perceber evolução na construção da linguagem.

Ao final do treinamento, será gerado um arquivo .pt, que representa os pesos do modelo treinado.

Esse arquivo deve ser salvo na pasta models.

Esse primeiro modelo funciona como um **modelo base** (*base model*). Nesse estágio, ele já:

* consegue montar frases;  
* possui noções básicas de contexto;  
* reconhece parcialmente relações entre palavras;  
* entende padrões gerais da linguagem.

No entanto, ainda não se comporta como um assistente conversacional. É um modelo que “fala bonito”, mas ainda sem grande capacidade de raciocínio ou alinhamento de respostas.


# **Fine Tuning** {#fine-tuning}

O processo de *fine tuning* é a etapa em que o modelo passa a aprender como se comportar como um assistente virtual.

Em vez de utilizar textos brutos, agora treinamos o modelo com exemplos de conversação estruturada.

No meu caso, utilizei o seguinte padrão de sintaxe:

usuário: pergunta  
assistente: resposta 

Foram criadas aproximadamente 200 variações de conversas, incluindo:

* cumprimentos;  
* diálogos simples;  
* perguntas e respostas;  
* reforço de conteúdos aprendidos durante o pré-treinamento.

Exemplo do dataset utilizado:
```
usuário: olá  
assistente: olá! como posso te ajudar hoje?

usuário: oi  
assistente: oi! tudo bem? me diga sua dúvida.

usuário: olá, tudo bem?  
assistente: olá, tudo bem sim! sou seu assistente, como posso ajudar?

usuário: boa tarde  
assistente: boa tarde! em que posso te ajudar?

usuário: boa noite  
assistente: boa noite! o que você gostaria de saber?
```
## **Executando o Fine Tuning** {#executando-o-fine-tuning}

Ao abrir o fine_tuning.ipynb no Google Colab ou Kaggle, é necessário configurar:

### **MODEL_PATH** {#model_path-1}

O caminho do modelo base gerado no pré-treinamento.

Exemplo:
```
MODEL_PATH = "/kaggle/input/model-base/model.pt"
```
### **FINETUNE_TXT** {#finetune_txt}

O caminho do arquivo .txt contendo os exemplos de conversação.

Exemplo:
```
FINETUNE_TXT = "/kaggle/input/dataset/conversas.txt"
```
Ao final do processo, um novo arquivo .pt será gerado.

Esse novo modelo deve ser salvo também na pasta models.

Agora, diferente do modelo base, esse modelo já foi ajustado para responder como um assistente virtual, apresentando:

* respostas mais naturais;  
* comportamento conversacional;  
* melhor contextualização;  
* maior alinhamento com perguntas e respostas humanas.

# **Inferência** {#inferência}

Agora que temos o modelo treinado e ajustado com *fine tuning*, já podemos realizar a etapa de inferência — ou seja, testar o modelo gerando respostas em tempo real.

Para isso, basta pegar o arquivo .pt gerado durante o treinamento e colocá-lo dentro da pasta models.

O sistema identifica automaticamente os modelos disponíveis nessa pasta assim que a aplicação é iniciada.

# **Executando a Aplicação** {#executando-a-aplicação}

O projeto possui duas formas principais de execução:

* via terminal, utilizando o main.py;  
* via interface web, utilizando o app.py.

## **1. Execução via Terminal (main.py)** {#1.-execução-via-terminal-(main.py)}

A forma mais simples de testar o modelo é diretamente pelo terminal.

Exemplo:
```
python main.py
```
rodando via terminal:  
![][image2]

## **2. Execução via Interface Web (app.py)** {#2.-execução-via-interface-web-(app.py)}

Também é possível executar o modelo utilizando a interface web integrada do projeto.

Exemplo:
```
python app.py 
```
Ao iniciar, a aplicação sobe um servidor local e disponibiliza uma interface acessível pelo navegador.

A interface permite:

* enviar mensagens;  
* selecionar modelos;  
* visualizar respostas em tempo real;  
* testar diferentes parâmetros de geração.

Após executar a aplicação, o funcionamento é bastante direto. O sistema carrega automaticamente o modelo selecionado e já permite iniciar conversas imediatamente.

rodando via front-end web: 

## REFERÊNCIAS: {#referências:}

VASWANI, Ashish et al. *Attention Is All You Need*. arXiv preprint arXiv:1706.03762, 2017. Disponível em: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762). Acesso em: 1 maio 2026. 

OpenAI. *Por que os modelos de linguagem alucinam?* 2025. Disponível em: [https://openai.com/pt-BR/index/why-language-models-hallucinate/](https://openai.com/pt-BR/index/why-language-models-hallucinate/). Acesso em: 1 maio 2026. 

RIBEIRO, Breno Campos. *Introdução aos conceitos de Transformer, Attention e GPT*. Disponível em: [https://brenocamposribeiro.com.br/blog/7](https://brenocamposribeiro.com.br/blog/7). Acesso em: 1 maio 2026.