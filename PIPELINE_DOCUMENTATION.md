# Documentação técnica do pipeline experimental

## 1. Objetivo e escopo

Este documento descreve o fluxo implementado para detecção binária de
patologia vocal, desde a descoberta dos arquivos nas bases HUPA e SVD
até a avaliação dos modelos. A descrição corresponde ao código atual e
ao protocolo confirmatório `gpu_confirmatory_v2`.

O objetivo clínico-computacional é classificar cada amostra como:

- `healthy` (`target = 0`);
- `pathological` (`target = 1`).

O desenho experimental foi construído para responder duas perguntas:

1. O modelo generaliza para locutores não vistos da mesma base?
2. O modelo treinado em uma base é transportável para outra base, com
   idioma, equipamento, população e protocolo de aquisição diferentes?

O fluxo completo é:

```text
linha de comando
    -> configuração centralizada e resolvida do experimento
    -> indexação dos arquivos e metadados
    -> manifesto bruto
    -> aplicação das regras de coorte e deduplicação
    -> manifesto de treinamento auditável
    -> leitura e pré-processamento de cada sinal
    -> extração de atributos acústicos
    -> separação agrupada de treino e teste
    -> seleção em validação cruzada apenas no treino
    -> retreinamento da configuração selecionada
    -> avaliação única no holdout ou na base externa
    -> métricas, predições, modelos e relatórios
```

As principais implementações estão distribuídas entre:

| Responsabilidade | Implementação principal |
|---|---|
| Entrada da aplicação | `main.py` |
| Configuração canônica | `classes/experiment/application/experiment_config_factory.py` |
| Adaptação da HUPA | `classes/dataset/adapters/hupa_adapter.py` |
| Adaptação da SVD | `classes/dataset/adapters/svd_adapter.py` |
| Definição das coortes | `classes/dataset/preparation/*_training_manifest_builder.py` |
| Leitura do áudio | `classes/audio_sample/audio_loader/audio_file_reader.py` |
| Pré-processamento | `classes/audio_sample/audio_loader/preprocessing/audio_preprocessor.py` |
| Extração dos atributos | `classes/vpd/vpd_feature_extractor.py` |
| Cenários e modelos | `classes/experiment/training/training_plan.py` |
| Divisões, seleção e avaliação | `classes/experiment/runners/model_training_runner.py` |
| Protocolo reproduzível | `classes/experiment/training/experimental_protocol_writer.py` |

## 2. Execução e estágios

O argumento `--dataset` escolhe o protocolo:

```bash
.venv/bin/python main.py --dataset hupa --stage train --compute-backend cuda
.venv/bin/python main.py --dataset svd --stage train --compute-backend cuda
.venv/bin/python main.py --dataset cross --stage train --compute-backend cuda
.venv/bin/python main.py --dataset pooled --stage train --compute-backend cuda
```

O argumento `--stage` representa o último estágio executado:

| Estágio | Operações executadas |
|---|---|
| `prepare` | indexação, manifesto bruto, coorte, deduplicação, perfil e figuras |
| `features` | tudo de `prepare` mais pré-processamento e extração de atributos |
| `train` | pipeline completo, incluindo seleção, treino e avaliação |

O treinamento confirmatório deve usar explicitamente
`--compute-backend cuda`. O valor padrão da interface ainda é `cpu` por
segurança e compatibilidade de desenvolvimento, mas resultados de CPU
recebem o protocolo `cpu_development_fallback_v1` e não são elegíveis
para o resultado final.

Ao iniciar um treino CUDA, `cuml.accel` é ativado antes da importação do
scikit-learn. A MLP confirmatória é implementada em PyTorch/Skorch e
executada em `device="cuda"`. O código também verifica a disponibilidade
real de CUDA e interrompe a execução se o backend solicitado não estiver
ativo. Como o protocolo CUDA usa uma única GPU, `n_jobs=1`.

Os caminhos padrão das bases são definidos em
`ExperimentSettings.default()`:

```text
HUPA: /mnt/d/masters_degree/datasets/hupa/BDAtos HUPA Segmentada
SVD:  /mnt/d/masters_degree/datasets/svd
```

Cada execução recebe uma pasta própria:

```text
data/<BASE>/experiments/<timestamp>_<base>_<nome_do_experimento>/
```

Isso evita a sobrescrita silenciosa de artefatos de execuções anteriores.

## 3. Manifestos: separação entre fatos e decisões experimentais

O pipeline mantém dois níveis de manifesto.

### 3.1. Manifesto bruto

O manifesto bruto representa o que foi encontrado fisicamente na base:
caminho, formato, duração, taxa de amostragem, canais, metadados,
identidade disponível, classe inferida e eventuais erros de leitura.

Nenhuma linha já indexada é removida durante essa etapa. Um arquivo
ilegível, por exemplo, permanece registrado com
`audio_read_status="error"` e a mensagem em `audio_read_error`. Na SVD,
um arquivo NSP cujo nome não corresponde a nenhum dos padrões
reconhecidos não chega a formar uma linha; seu caminho é emitido como
aviso durante a indexação.

Essa escolha preserva a rastreabilidade: é possível diferenciar “o
arquivo não existe na base” de “o arquivo existe, mas foi excluído por
uma decisão experimental”.

### 3.2. Manifesto de treinamento

O `TrainingManifestBuilder` transforma o manifesto bruto em uma coorte
adequada ao experimento. Ele gera simultaneamente:

- o manifesto de treinamento;
- a relação de amostras excluídas, com motivo primário e detalhe;
- a auditoria dos grupos duplicados;
- um resumo JSON com contagens, distribuição das classes e configuração.

Cada linha pode receber somente um motivo primário de exclusão. As
regras são aplicadas em ordem, o que torna as contagens reproduzíveis.
Os artefatos são salvos em CSV e, quando aplicável, Parquet.

## 4. Carregamento e preparação da HUPA

### 4.1. Indexação do áudio

O adaptador procura recursivamente todos os arquivos `*.wav`. Para cada
arquivo:

- calcula o caminho relativo;
- cria um `sample_id` determinístico a partir desse caminho;
- lê taxa de amostragem, duração e número de canais;
- fixa a tarefa vocal como vogal sustentada `/a/`;
- infere a classe a partir dos marcadores da pasta.

Marcadores como `normal` e `healthy` produzem `healthy`. Marcadores como
`pathol`, `patologico` e suas variantes produzem `pathological`. A
execução falha se um arquivo não puder ter a classe inferida, evitando
que uma amostra ambígua seja aceita automaticamente.

### 4.2. Metadados

O arquivo `HUPA segmentada.xls` é lido nas planilhas `Normales` e
`Patológicos`, usando a segunda linha como cabeçalho (`header=1`). Os
nomes são convertidos para um esquema comum, incluindo idade, sexo,
patologia, código de patologia, GRBAS, formantes e controles de
qualidade.

Campos numéricos são convertidos com valores inválidos transformados em
ausentes. O sexo é harmonizado da codificação original para `male` e
`female`.

O vínculo entre áudio e metadado usa a chave composta
`[nome_normalizado_do_arquivo, classe]`. Tanto o índice acústico quanto
os metadados devem ser únicos nessa chave e a junção é validada como
`one_to_one`.

### 4.3. Identidade do locutor na HUPA

A versão segmentada disponível não fornece um identificador independente
de sujeito. Inicialmente o `speaker_id` é derivado do `sample_id`.
Depois da deduplicação física, o identificador da amostra canônica passa
a ser também o grupo de locutor, com a fonte
`assumed_unique_acoustic_sample`.

Essa é uma limitação importante: o procedimento garante que cópias
idênticas do mesmo arquivo não cruzem partições, mas não consegue provar
que dois arquivos acusticamente diferentes pertencem a pessoas
diferentes. Caso um identificador clínico real se torne disponível, ele
deve substituir essa suposição.

### 4.4. Regras da coorte HUPA

Na configuração atual, permanecem apenas amostras que:

- podem ser lidas sem erro;
- possuem rótulo binário válido;
- possuem `speaker_id`;
- possuem idade válida e idade maior ou igual a 18 anos;
- têm duração mínima de 0,5 segundo.

O uso exclusivo de adultos harmoniza o critério etário com a SVD e
reduz a possibilidade de o classificador aprender diferenças acústicas
de maturação vocal em vez de patologia.

## 5. Carregamento e preparação da SVD

### 5.1. Indexação do áudio

O adaptador procura diretórios que contenham `overview.csv`. Cada nome
de diretório-pai representa um grupo de patologia. Dentro das pastas de
gravação, são indexados os arquivos `vowels/*.nsp`.

Os nomes de vogais sustentadas seguem o padrão:

```text
<recording_id>-<vogal>_<condição>.nsp
```

São reconhecidas as vogais `a`, `i` e `u` e as condições `h`, `l`,
`lhl` e `n`. A sequência `iau` também pode ser indexada no manifesto
bruto, mas não entra na coorte atual.

O ID presente no nome do arquivo é comparado ao ID da pasta. Diferenças
são relatadas como diagnóstico. O rótulo `healthy` é atribuído somente
aos nomes conhecidos de grupos saudáveis; os demais grupos são
`pathological`.

### 5.2. Metadados

Cada `overview.csv` fornece, entre outros campos:

- ID da gravação;
- ID do locutor;
- data de gravação;
- data de nascimento;
- sexo;
- diagnóstico e patologia.

A idade é calculada como a diferença entre as datas em dias dividida por
365,2425 e arredondada para uma casa decimal. O sexo `m` é convertido em
`male` e `w` em `female`.

Duplicatas de metadados com a mesma chave
`[grupo_de_patologia, recording_id]` são consolidadas apenas se o
conteúdo clínico for idêntico. Se houver qualquer conflito relevante, a
execução falha. A junção áudio-metadado é validada como `many_to_one`.

### 5.3. Regras da coorte SVD

Na configuração atual, permanecem apenas:

- a vogal sustentada `/a/`;
- a condição normal `n`;
- arquivos legíveis;
- rótulos binários válidos;
- registros com `speaker_id`;
- adultos com idade válida e idade maior ou igual a 18 anos;
- sinais com pelo menos 0,5 segundo.

A restrição `/a/` em condição normal aproxima a tarefa vocal da HUPA.
Sem essa harmonização, o modelo poderia explorar diferenças entre
vogais, pitch induzido ou modo de fonação, produzindo uma estimativa
inflada de desempenho que não representa patologia.

## 6. Deduplicação e verificações de integridade

Depois dos filtros de coorte, cada arquivo recebe um SHA-256 calculado
em blocos de 1 MiB. O hash identifica igualdade física de conteúdo sem
depender de nome ou pasta.

### 6.1. HUPA

Arquivos com o mesmo hash são consolidados em uma amostra canônica. O
ID canônico tem o formato `hupa_<16 primeiros caracteres do hash>`.
Patologias e códigos clínicos observados são agregados; outros campos
conflitantes são anulados e registrados em
`metadata_conflict_columns`.

Há duas regras de falha segura:

- o mesmo áudio não pode possuir rótulos binários diferentes;
- na coorte adulta, um grupo de áudio duplicado com idades adultas
  conflitantes é excluído integralmente.

### 6.2. SVD

A chave acústica é `[recording_id, vowel, condition]`. Cópias dessa
mesma gravação só podem ser consolidadas se tiverem o mesmo SHA-256.
Também se exige que cada hash esteja associado a uma única chave
acústica, um único locutor e um único rótulo.

Essas verificações evitam duas formas graves de vazamento:

1. a mesma onda aparecer no treino e no teste com nomes diferentes;
2. cópias ou conflitos de metadados serem contados como evidências
   independentes.

## 7. Leitura do sinal

O `AudioFileReader` aceita:

- WAV, lido por `soundfile`;
- NSP, lido por `nspfile`.

O sinal é convertido para `float32`. Inteiros com sinal são divididos
pelo maior valor absoluto representável; inteiros sem sinal são
centralizados em seu ponto médio antes da normalização. Apenas matrizes
mono ou multicanal bidimensionais são aceitas.

Na extração de atributos, cada linha do manifesto de treinamento é
carregada novamente. Os metadados da linha são anexados ao
`AudioSample`, mantendo o vínculo entre a representação numérica e sua
origem clínica.

## 8. Pré-processamento do áudio

O pré-processamento é determinístico e aplicado independentemente a
cada amostra. Nenhum parâmetro é estimado usando o conjunto completo,
portanto essa etapa não transmite estatísticas entre treino e teste.

### 8.1. Conversão para mono

Sinais com mais de um canal são convertidos pela média aritmética dos
canais, calculada em `float64` e retornada em `float32`.

Motivo: todas as amostras passam a ter a mesma representação e nenhum
modelo recebe vantagem por uma disposição específica de canais.

### 8.2. Remoção de componente DC

Subtrai-se a média temporal do sinal:

```text
y_dc[n] = y[n] - média(y)
```

Motivo: offsets de aquisição não representam a vibração vocal e podem
afetar energia, entropia e cruzamentos por zero.

### 8.3. Reamostragem

Todos os sinais são convertidos para 16 kHz com
`scipy.signal.resample_poly`. Os fatores de interpolação e decimação
são reduzidos pelo máximo divisor comum das taxas original e desejada.

Motivo: uma taxa comum torna comparáveis FFT, MFCC, frequência e janelas
temporais entre bases. A reamostragem polifásica também aplica a
filtragem necessária para limitar aliasing.

### 8.4. Normalização RMS

O RMS é:

```text
rms = sqrt(média(y²))
```

O ganho leva esse valor a -20 dBFS:

```text
rms_alvo = 10^(-20/20) = 0,1
ganho = rms_alvo / rms
```

Se o pico resultante ultrapassar 0,99, todo o sinal é reduzido pelo
mesmo fator para respeitar esse limite. Sinais praticamente silenciosos
não são amplificados.

Motivo: reduz-se a dependência do classificador em ganho de microfone,
distância e escala do arquivo. O limite de pico evita clipping numérico.
Essa transformação não elimina variações temporais e espectrais de
interesse.

### 8.5. Duração

`center_crop=False`: não há recorte central, preenchimento ou
truncamento. O sinal completo é usado depois do filtro mínimo de 0,5
segundo.

Motivo: os atributos atuais agregam estatísticas ao longo do tempo e
aceitam durações variáveis. Recortar poderia descartar início, região
estável ou final da fonação; preencher poderia introduzir silêncio
artificial.

### 8.6. Validação e perfil

Antes e depois das transformações, o pipeline rejeita sinais vazios,
taxas inválidas, formatos inesperados e valores NaN ou infinitos. Um
perfil separado registra duração, RMS e pico processados para permitir
auditoria do efeito real do pré-processamento.

## 9. Extração de atributos

Os atributos são calculados sobre o sinal completo pré-processado. A
configuração espectral comum usa:

```text
taxa de amostragem = 16.000 Hz
n_fft              = 1.024 amostras (64 ms)
hop_length         = 128 amostras (8 ms)
```

Além dos atributos numéricos, a tabela mantém metadados como base,
locutor, classe, idade, sexo, patologia, tarefa, hash e duração. Esses
campos são explicitamente excluídos da matriz de entrada do modelo.

### 9.1. Harmônicos espectrais: 64 atributos

Primeiro, o HPSS do Librosa separa a porção harmônica. Calcula-se a STFT
e a magnitude média temporal de cada bin. São escolhidos os 30 bins de
maior magnitude acima de 50 Hz e depois ordenados por frequência para
estabilidade das colunas.

Para cada bin selecionado são salvas frequência e amplitude
(`30 × 2 = 60`). Também são salvas média e desvio-padrão das frequências
e amplitudes (`4`).

Importante: esses valores são picos espectrais candidatos após HPSS, e
não harmônicos glotais explicitamente estimados como múltiplos de F0.

### 9.2. Distribuição temporal da energia: 9 atributos

Calcula-se a energia acumulada `cumsum(y²)`. Para cada limiar de 10% a
90%, registra-se a posição temporal normalizada em que a energia
acumulada atinge o limiar.

Esses atributos descrevem como a energia se distribui ao longo da
fonação sem exigir sinais de duração fixa.

### 9.3. Entropia segmentada: 58 atributos

O sinal é dividido separadamente em 2, 3, 5, 7, 11, 13 e 17 partes. Em
cada parte, estima-se a entropia de Shannon a partir de um histograma de
64 bins:

```text
H = - soma(p_i * log2(p_i))
```

As entropias de cada particionamento são normalizadas pelo maior valor
observado naquele particionamento. A soma do número de segmentos é 58.

Essa família captura variações de complexidade e irregularidade em
escalas temporais diferentes.

### 9.4. Cruzamentos por zero: 6 atributos

Depois de recentralizar o sinal, registra-se a posição normalizada em
que o total acumulado de cruzamentos por zero alcança 20%, 40%, 60% e
80%. Também são salvos o total de cruzamentos e a taxa por transição
possível.

Esses atributos resumem conteúdo de alta frequência, ruído e evolução
temporal da oscilação.

### 9.5. MFCC, delta e delta-delta: 450 atributos

Antes do MFCC é aplicada pré-ênfase:

```text
y_pre[0] = y[0]
y_pre[n] = y[n] - 0,97 * y[n-1]
```

São extraídos 30 coeficientes MFCC. Também se calculam a primeira e a
segunda derivadas temporais. Para cada coeficiente de cada uma das três
matrizes são agregados:

- média;
- desvio-padrão;
- mínimo;
- máximo;
- mediana.

Assim, a configuração normal produz `30 × 3 × 5 = 450` atributos. As
derivadas fornecem informação temporal resumida, embora o modelo final
continue recebendo um vetor fixo, e não uma sequência de frames.

### 9.6. Qualidade vocal/glotal: 9 atributos

O Parselmouth/Praat estima:

- média, desvio-padrão, mínimo e máximo de F0;
- proporção de frames sonoros;
- média e desvio-padrão da relação harmônico-ruído (HNR);
- jitter local;
- shimmer local.

O intervalo de pitch é 75–600 Hz e o passo temporal é 10 ms. Se a
extração falhar para uma amostra, esses nove valores recebem NaN; a
amostra não é descartada por isso, pois a imputação ocorre dentro do
pipeline de treino.

### 9.7. Dimensão e cenários

Com todas as famílias presentes, existem até 596 atributos acústicos:

```text
64 harmônicos + 9 energia + 58 entropia + 6 ZCR
+ 450 MFCC/deltas + 9 glotais = 596
```

Eles são avaliados em seis cenários:

| Cenário | Famílias incluídas |
|---|---|
| `mfcc` | MFCC, delta e delta-delta |
| `harmonics` | harmônicos espectrais |
| `energy_entropy_zcr` | energia, entropia e ZCR |
| `all_without_glottal` | todas, exceto glotais |
| `glottal` | somente qualidade vocal/glotal |
| `all_with_glottal` | todas as famílias |

O cenário também é selecionado apenas por validação cruzada no treino.
Ele não é escolhido a partir do holdout ou da base externa.

## 10. Tratamento de falhas na extração

Cada amostra é processada em um bloco independente. Sucesso gera
`status="ok"`; uma exceção gera `status="error"` e preserva a mensagem e
os metadados disponíveis. A quantidade de linhas deve continuar igual à
do manifesto e `sample_id` duplicado interrompe a execução.

Antes do treino, somente linhas `status="ok"` e com classe válida são
usadas. Essa filtragem é explícita e auditável, em vez de desaparecer
com linhas silenciosamente.

## 11. Experimentos implementados

### 11.1. HUPA → holdout HUPA

A coorte HUPA é dividida em aproximadamente 80% para treino e 20% para
teste. A divisão é agrupada por `speaker_id`, estratificada pela classe
e escolhida entre os cinco folds candidatos pela combinação de:

- proximidade do tamanho desejado de 20%;
- proximidade da distribuição global das classes.

Somente os 80% de treino participam da seleção. O campeão global é o
único resultado primário. Como análise secundária pré-especificada, os
campeões SVM e MLP selecionados pela CV de treino também são avaliados
no mesmo holdout; o desempenho no holdout não participa dessa escolha.

### 11.2. SVD → holdout SVD

O procedimento é o mesmo, mas o agrupamento por locutor tem identidade
real proveniente dos metadados da SVD. Todas as sessões de um locutor
permanecem na mesma partição.

### 11.3. HUPA → SVD

Toda a HUPA harmonizada é a base de origem. Cenário, família de modelo,
pré-processamento e hiperparâmetros são escolhidos por validação cruzada
agrupada somente na HUPA. O pipeline selecionado é retreinado na HUPA
completa e avaliado uma única vez na SVD completa.

### 11.4. SVD → HUPA

Repete-se o procedimento invertendo origem e destino. A HUPA não
participa de nenhuma decisão tomada durante a seleção realizada na SVD.

### 11.5. HUPA + SVD → holdout misto

As tabelas são alinhadas para usar apenas atributos acústicos numéricos
presentes nas duas bases. Para impedir colisões de IDs, o grupo é:

```text
<base>::<speaker_id>
```

A divisão é agrupada por esse identificador e estratificada pelas quatro
combinações disponíveis de base e classe. Cada fold deve conter todas as
combinações tanto no treino quanto no teste.

Um único pipeline é escolhido como resultado primário. A comparação
secundária SVM–MLP segue a mesma regra pré-especificada. Para cada
modelo avaliado, os resultados pooled são emitidos:

- globalmente;
- somente para HUPA;
- somente para SVD;
- como macro média entre as duas bases.

A macro média dá o mesmo peso a cada base, independentemente do número
de amostras.

## 12. Construção da matriz de modelagem

O rótulo é convertido para inteiro, mas permanece fora de `X`. Também
ficam fora de `X` todos os campos de identidade, demografia, patologia,
base, duração, taxa de amostragem, caminho, hash, status e auditoria.

Somente colunas numéricas que pertencem ao cenário acústico selecionado
entram no modelo. Para cross-database e pooled, usa-se a interseção dos
atributos disponíveis nas duas bases, e o esquema utilizado é salvo em
CSV.

Isso impede que o modelo aprenda diretamente:

- o nome da base;
- idade ou sexo;
- o ID do locutor;
- o diagnóstico textual;
- o caminho do arquivo;
- a duração ou taxa de amostragem armazenada como metadado.

## 13. Pipeline aprendido dentro da validação cruzada

Para cada candidato, o scikit-learn constrói:

```text
SimpleImputer(median)
    -> StandardScaler
    -> conversão explícita para NumPy no limite GPU/CPU
    -> SelectPercentile(f_classif, 10%, 25% ou 50%)
    -> classificador
```

O ponto mais importante é que imputer, scaler e seletor fazem parte do
`Pipeline` entregue ao `GridSearchCV`. Em cada fold:

1. a mediana é calculada somente no subconjunto de ajuste;
2. média e desvio-padrão do `StandardScaler` são calculados somente
   nesse subconjunto;
3. o teste F ANOVA e o ranking de atributos usam somente esse
   subconjunto;
4. o classificador é ajustado somente depois dessas transformações;
5. o fold de validação é apenas transformado com os parâmetros já
   aprendidos.

Se essas operações fossem feitas antes da validação cruzada, o fold de
validação influenciaria a representação usada no treino e produziria
vazamento de informação.

O mesmo espaço de pré-processamento é obrigatório para SVM linear, SVM
RBF e MLP no protocolo confirmatório. Isso mantém justa a comparação
entre famílias: diferenças de resultado devem vir da capacidade do
modelo, e não de um tratamento privilegiado dos atributos.

## 14. Divisões e validação cruzada

### 14.1. Holdout

O holdout usa `test_size=0.20` e semente 42. Com grupos disponíveis,
usa-se `StratifiedGroupKFold` com cinco folds embaralhados e escolhe-se o
fold mais próximo do tamanho e da distribuição desejados. O código
verifica explicitamente:

- presença das duas classes no treino e no teste;
- presença de todos os estratos configurados;
- interseção vazia entre grupos de treino e teste.

### 14.2. Validação cruzada interna

A seleção usa cinco folds com `StratifiedGroupKFold`, embaralhamento e
semente fixa. Cada fold deve conter as duas classes e não pode
compartilhar locutores entre ajuste e validação.

No pooled, o alvo de estratificação combina:

```text
base::target=<classe>
```

Logo, os folds preservam simultaneamente base e classe.

### 14.3. Validação cruzada aninhada repetida

Como diagnóstico de generalização e estabilidade de seleção, roda-se
uma validação aninhada exclusivamente sobre a partição de treino da
origem:

```text
3 folds externos × 2 repetições
5 folds internos para seleção em cada fold externo
```

Em cada fold externo, todo o processo de escolha de cenário, família,
pré-processamento e hiperparâmetros é repetido. São registrados
desempenho externo, gap treino-validação, configuração selecionada e
frequência de seleção.

Esse procedimento fornece uma estimativa menos otimista do processo de
seleção completo. Ele não consulta o holdout final nem a base externa,
mas aumenta substancialmente o custo de execução.

## 15. Modelos e espaços de busca

Todos os modelos usam a mesma métrica primária:
`balanced_accuracy`.

### 15.1. SVM linear

```text
kernel       = linear
class_weight = balanced
C            = 0,01; 0,1; 1; 10
```

O SVM linear é a referência de menor capacidade. Valores baixos de `C`
aceitam mais erros de treino em troca de uma margem mais larga e,
portanto, maior regularização.

### 15.2. SVM RBF

```text
kernel       = rbf
class_weight = balanced
C            = 0,01; 0,1; 1; 10
gamma        = scale; 0,0001; 0,001; 0,01
```

O RBF permite fronteiras não lineares. `C` controla a penalização de
erros e `gamma` controla o alcance de cada amostra. Valores altos podem
criar regiões locais muito específicas; por isso a grade inclui valores
baixos que favorecem fronteiras suaves.

### 15.3. MLP PyTorch/CUDA

A rede tem:

```text
entrada
    -> camada densa de 8 ou 16 neurônios
    -> ReLU
    -> dropout
    -> opcionalmente camada de 8 neurônios no perfil forte
    -> camada de saída com 2 logits
```

O treinamento usa Adam, learning rate 0,001, mini-batches de 32 e
cross-entropy ponderada. A grade testa 5, 10, 15 ou 20 épocas e dois
perfis conjuntos:

| Perfil | Dropout | Weight decay | Label smoothing |
|---|---:|---:|---:|
| moderado | 0,20 | 0,001 | 0,05 |
| forte | 0,40 | 0,01 | 0,10 |

O perfil moderado testa `(8,)` e `(16,)`. O perfil forte também testa
`(16, 8)`, atendendo à hipótese pré-especificada de uma MLP mais profunda
sem expô-la à regularização moderada. A remoção de 30 épocas mantém o
número total de candidatos CUDA igual ao plano anterior.

Os perfis são avaliados como pacotes de regularização. Eles não devem
ser interpretados no paper como uma ablação capaz de isolar o efeito
individual de dropout, weight decay ou label smoothing.

## 16. Seleção parcimoniosa

Escolher sempre o maior valor médio de CV favorece diferenças pequenas
e instáveis. Por isso o protocolo aplica uma regra inspirada em
one-standard-error.

### 16.1. Dentro de cada modelo/cenário

Para o candidato numericamente melhor:

```text
erro_padrão = desvio_padrão_dos_folds / sqrt(n_folds)
queda_aceita = max(erro_padrão, 0,005)
limiar = melhor_média - queda_aceita
```

Todos os candidatos acima do limiar são considerados equivalentes. O
mais simples é escolhido por uma chave que considera, conforme a
arquitetura:

- menor percentual de atributos;
- menos neurônios e menor profundidade;
- menos épocas;
- menor `C`;
- menor `gamma`;
- maior regularização;
- maior dropout.

### 16.2. Entre cenários e famílias

O melhor resultado numérico de todo o treino estabelece outro limiar
com a mesma lógica. Entre candidatos elegíveis, não há preferência fixa
por SVM linear, SVM RBF ou MLP. O desempate global usa:

1. menor desvio-padrão de CV;
2. menor valor absoluto do gap entre treino e CV;
3. menos atributos selecionados;
4. maior média de CV;
5. ordem determinística apenas como último desempate.

A seleção é salva em `source_model_selection.csv`, contendo tanto o
máximo numérico quanto a configuração parcimoniosa.

## 17. Retreinamento e avaliação intocada

O `GridSearchCV` refaz o ajuste do pipeline selecionado em toda a
partição de treino. Só então esse pipeline é aplicado ao conjunto
reservado.

No cross-database, o destino nunca é passado à busca. Ele é usado apenas
depois que existe uma única configuração selecionada e retreinada na
origem. Portanto, comparar todos os candidatos na base externa e
escolher o “vencedor externo” não faz parte do protocolo atual.

Essa separação deve ser refletida no paper:

- CV interna: escolha de configuração;
- CV aninhada: diagnóstico do processo de escolha na origem;
- curva de treino/validação: diagnóstico de otimização;
- holdout ou base externa: estimativa primária do campeão global;
- comparação SVM–MLP: análise secundária pré-especificada entre os
  campeões de família escolhidos exclusivamente na CV de treino.

## 18. Métricas

São calculadas:

- acurácia;
- acurácia balanceada;
- UAR;
- precisão;
- sensibilidade;
- especificidade;
- F1 da classe positiva;
- macro-F1;
- MCC;
- ROC AUC;
- PR AUC;
- matriz de confusão (`TN`, `FP`, `FN`, `TP`).

UAR continua registrado nos CSVs para compatibilidade, mas não aparece
no gráfico comparativo final porque, neste problema binário, é
numericamente redundante com a acurácia balanceada.

A acurácia balanceada é a métrica primária porque atribui o mesmo peso à
sensibilidade de cada classe, mesmo quando as classes possuem tamanhos
diferentes.

Quando o modelo fornece `predict_proba`, a probabilidade da classe
patológica é usada para AUC. Para SVM sem probabilidade calibrada, usa-se
`decision_function`.

### 18.1. Intervalos de confiança

São executadas 1.000 réplicas bootstrap com nível de confiança de 95%.
O reamostramento é feito por `speaker_id`, com reposição, e não por linha
individual. Todas as gravações de um locutor selecionado entram juntas
na réplica.

Isso respeita a dependência entre sessões do mesmo sujeito. Réplicas que
não contêm as duas classes são ignoradas.

## 19. Curvas e diagnósticos de overfitting

### 19.1. Curvas da MLP por época

Depois da seleção, uma cópia diagnóstica da MLP é ajustada em uma
divisão adicional, agrupada por locutor, feita somente dentro da
partição de treino. O pré-processamento é aprendido apenas no
subconjunto de ajuste dessa curva.

Por época são registrados:

- `train_loss`;
- `valid_loss`;
- `train_accuracy`;
- `valid_accuracy`;
- `train_balanced_accuracy`;
- `valid_balanced_accuracy`.

Esse ajuste diagnóstico existe para visualizar quando a loss de
validação começa a subir, quando a acurácia balanceada estagna e como o
gap se desenvolve. Ele não substitui o modelo final e não escolhe uma
nova época depois de olhar o holdout.

### 19.2. Curva de aprendizado do SVM

Quando o modelo final selecionado é SVM, são usados 25%, 50%, 75% e
100% dos grupos disponíveis em cada fold interno. A validação permanece
fixa e sem locutores compartilhados.

São registrados:

- acurácia balanceada de treino;
- acurácia balanceada de validação;
- gap de generalização;
- média e desvio-padrão entre folds;
- grupos usados em cada ponto.

Treino alto e validação baixa em todas as frações indica alta variância.
Melhora consistente da validação com mais grupos sugere que mais dados
reais podem ser mais úteis que aumentar a complexidade.

### 19.3. Gap treino–CV

O `GridSearchCV` salva resultados de treino e validação. Para a
configuração escolhida:

```text
gap = média_balanced_accuracy_treino
      - média_balanced_accuracy_validação
```

O gap participa do desempate global e também é persistido nas métricas.
Ele não “prova” sozinho que há overfitting, mas permite comparar modelos
com desempenho de validação semelhante.

## 20. Técnicas usadas para mitigar overfitting

O overfitting observado não é tratado por uma única técnica. Foram
combinadas medidas em dados, representação, modelo, seleção e avaliação.

| Técnica | Problema mitigado | Mecanismo e justificativa |
|---|---|---|
| Deduplicação SHA-256 | memorização de cópias | impede que o mesmo conteúdo físico seja contado mais de uma vez ou apareça em partições distintas |
| Divisão por locutor | reconhecimento do sujeito | mantém todas as sessões do mesmo locutor juntas, quando a identidade está disponível |
| Coortes adultas harmonizadas | confundimento por idade | reduz diferenças de maturação vocal entre base, classe e partição |
| Mesma tarefa vocal | confundimento por vogal/condição | restringe SVD a `/a/` normal para aproximar o protocolo da HUPA |
| Sem SMOTE | amostras clínicas sintéticas imprecisas | evita interpolar vetores que podem não corresponder a sinais fisiologicamente plausíveis |
| Pesos de classe/amostra | desbalanceamento | altera o custo dos erros somente no treino, sem inventar pacientes |
| Imputação dentro dos folds | vazamento estatístico | a mediana do fold de validação/teste não influencia o treino |
| Escalonamento dentro dos folds | vazamento e domínio de escala | estatísticas de centralização e escala vêm apenas do ajuste |
| Seleção de 10%, 25% ou 50% dos atributos | alta dimensionalidade | reduz variáveis irrelevantes e a relação parâmetros/amostras |
| Cenários de atributos | dependência de uma representação grande | permite selecionar uma família mais simples se ela generalizar de forma equivalente |
| SVM linear | fronteira excessivamente flexível | fornece uma referência explicitamente de baixa capacidade |
| Grade de `C` e `gamma` baixos | RBF muito local | inclui margens mais largas e kernels mais suaves |
| MLP com 8 ou 16 neurônios; `(16, 8)` apenas no perfil forte | excesso de parâmetros | limita capacidade e testa profundidade apenas sob regularização forte |
| Máximo de 20 épocas | memorização tardia e custo | cobre a região útil observada e mantém o orçamento após adicionar `(16, 8)` |
| Mini-batches | ajuste determinístico excessivo | introduzem ruído de gradiente e reduzem custo computacional |
| Dropout | coadaptação de neurônios | remove ativações aleatoriamente durante o treino |
| Weight decay | pesos de grande magnitude | penaliza parâmetros e favorece soluções mais suaves |
| Label smoothing | excesso de confiança | evita alvos exatamente 0/1 na loss e reduz logits extremos |
| One-standard-error + tolerância | escolher ganho irrelevante | prefere candidatos simples quando a diferença de CV não é convincente |
| Desvio e gap no desempate | soluções instáveis | entre médias semelhantes, favorece menor variabilidade e menor separação treino–CV |
| CV aninhada repetida | otimismo da seleção | avalia o processo inteiro em folds externos nunca vistos pela busca interna |
| Holdout/base externa intocados | viés de seleção | impede escolher o vencedor a partir do conjunto declarado como teste |
| Bootstrap por locutor | incerteza subestimada | preserva a unidade experimental real ao calcular intervalos |
| Hash do protocolo | mudança oportunista | torna detectáveis alterações de configuração entre execuções |
| Seeds pré-especificados | escolha oportunista da melhor execução | mantém holdout em 42, usa 42/43 somente nas repetições externas e reporta todas as estimativas |

### 20.1. Por que não foi usado SMOTE

SMOTE interpola pontos no espaço de atributos. Neste projeto, um ponto é
uma combinação de medidas espectrais, temporais, cepstrais e glotais.
Não há garantia de que a interpolação corresponda a uma voz que poderia
ser produzida por um trato vocal real, nem de que preserve coerentemente
idade, sexo, patologia e severidade.

Além do risco fisiológico, a síntese dentro de uma amostra pequena pode
amplificar ruído de anotação e regiões específicas da base. Por isso o
protocolo usa pesos de classe ou amostra calculados somente no
subconjunto de treino. Essa decisão corrige a função de custo sem criar
novas observações clínicas.

SMOTE só deveria ser reconsiderado como experimento separado, aplicado
exclusivamente dentro de cada fold de treino e acompanhado de uma
validação clínica e acústica das amostras geradas. Seus resultados não
deveriam ser misturados ao protocolo confirmatório atual.

### 20.2. Por que não foi usado early stopping interno da MLP

O early stopping padrão costuma criar uma divisão aleatória interna.
Isso poderia colocar sessões do mesmo locutor no treino e na validação,
ou ignorar os estratos de base/classe do pooled.

Para manter o mesmo princípio de comparação usado pelos SVMs, o número
de épocas é tratado como hiperparâmetro e escolhido pelos folds
agrupados externos do `GridSearchCV`. A curva com validação agrupada é
gerada separadamente para diagnóstico.

Se futuramente for implementado early stopping agrupado dentro de cada
fold, isso constituirá uma alteração de protocolo e deverá receber nova
versão e justificativa no paper.

### 20.3. Por que os modelos têm planos específicos

Há um núcleo comum obrigatório:

- mesma coorte;
- mesmos atributos candidatos;
- mesma imputação, escala e seleção;
- mesmas divisões agrupadas;
- mesma métrica;
- mesma política de seleção;
- mesmo conjunto final de teste.

As diferenças restantes são inerentes à arquitetura. `C` e `gamma` não
existem em uma MLP; dropout, épocas e weight decay não existem em um
SVM. A comparação é justa quando parâmetros compartilháveis são
mantidos iguais e parâmetros exclusivos são apresentados como controle
de capacidade de cada família, com a grade registrada antes da avaliação
final.

## 21. Reprodutibilidade e artefatos

Uma execução preparada contém, conforme a base:

```text
config.json
manifests/
    *_raw_manifest.csv|parquet
    *_training_manifest.csv|parquet
    *_excluded_samples.csv
    *_duplicate_groups.csv
profiles/
    *_processed_audio_profile.csv|parquet
features/
    *_features_v1.csv|parquet
figures/
reports/
    *_preparation_summary.json
    summary.txt
```

O diretório de treino contém a estrutura abaixo. No cross-database, a
mesma estrutura fica diretamente dentro de cada diretório direcional
`train_hupa_test_svd/` e `train_svd_test_hupa/`, em vez de uma pasta
intermediária chamada `training`.

```text
training/
    experimental_protocol.json
    experimental_protocol.md
    models/
    metrics/
        metrics.csv|parquet
        source_model_selection.csv|parquet
        *_cv_results.csv
        repeated_nested_cv_*.csv|parquet
    predictions/
    splits/
        holdout_assignments.csv
        feature_schema.csv                    # cross, quando aplicável
        *_curve_assignments.csv
        repeated_nested_cv_assignments.csv
    figures/
        training_curves/
        learning_curves/
```

O protocolo resolvido inclui configuração, cenários, estimadores, grades,
racionais e política de seleção. Uma representação canônica recebe um
SHA-256, gravado também nas métricas. Resultados com hashes diferentes
não devem ser agregados como se pertencessem ao mesmo protocolo.

## 22. Limitações e interpretação

- A identidade de locutor da HUPA é uma suposição após deduplicação, não
  um identificador clínico confirmado.
- HUPA e SVD ainda diferem em idioma, equipamento, sala, critérios de
  inclusão e distribuição de patologias. A queda cross-database mistura
  overfitting e domain shift.
- Os atributos MFCC delta/delta-delta resumem tempo por estatísticas.
  Eles exploram dinâmica, mas não preservam a sequência completa.
- A curva por época é um ajuste diagnóstico adicional em um split
  agrupado; ela não é a estimativa final de desempenho.
- Os dois perfis de regularização da MLP são pacotes. O protocolo não
  mede o efeito causal isolado de cada componente.
- A validação aninhada completa é computacionalmente cara porque repete
  toda a seleção de cenários e famílias.
- O backend CPU usa outra implementação de MLP e serve somente ao
  desenvolvimento. O paper deve reportar apenas execuções CUDA com o
  mesmo hash confirmatório.

## 23. Possível extensão temporal profunda

O pipeline atual transforma cada gravação em um vetor fixo. Uma extensão
para explorar a sequência completa poderia usar, por exemplo, uma
CNN temporal, TCN, CNN-BiLSTM ou Transformer pequeno sobre
log-mel-spectrogramas ou frames de MFCC.

Essa extensão não deve substituir silenciosamente o protocolo atual,
pois mudaria:

- a unidade de entrada;
- padding e máscaras;
- data augmentation;
- capacidade do modelo;
- custo de treino;
- espaço de hiperparâmetros;
- interpretação dos atributos.

O desenho mais defensável seria manter os modelos tabulares atuais como
baselines confirmatórios e registrar o modelo sequencial como protocolo
novo. Para controlar overfitting, ele precisaria de arquitetura pequena,
divisões pelos mesmos locutores, regularização, seleção apenas na origem
e avaliação externa intocada.

## 24. Checklist para relatar no paper

Antes de consolidar resultados:

- confirmar `protocol_version="gpu_confirmatory_v2"`;
- confirmar `eligible_for_final_reporting=true`;
- confirmar que os hashes dos experimentos comparados são iguais;
- auditar exclusões e duplicatas;
- auditar `holdout_assignments.csv`;
- verificar ausência de locutores compartilhados;
- revisar `source_model_selection.csv`;
- revisar gaps, curvas e estabilidade da CV aninhada;
- revisar distribuição demográfica e de patologias por base;
- reportar métricas com intervalos de confiança;
- declarar explicitamente a limitação de identidade da HUPA;
- não selecionar modelos pela métrica obtida na base externa;
- não combinar resultados CPU e CUDA.
