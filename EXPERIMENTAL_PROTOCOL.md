# Protocolo experimental confirmatório

> Estado consolidado em 2 de agosto de 2026. Este documento distingue o
> protocolo efetivamente executado das extensões propostas para trabalhos
> futuros.

## 1. Escopo e versões

O objetivo experimental foi avaliar a classificação binária de vozes
`healthy` e `pathological` a partir de vogais sustentadas, considerando tanto
a generalização para locutores não observados da mesma base quanto a
transportabilidade entre HUPA e SVD.

Duas versões de protocolo produziram resultados elegíveis para o texto final:

- `gpu_confirmatory_v2`: cinco experimentos harmonizados pela vogal /a/;
- `gpu_multivowel_extension_v1`: dois experimentos complementares com a SVD
  em /a/, /i/ e /u/ na condição normal.

Somente execuções CUDA com `eligible_for_final_reporting=true` integram os
resultados. A MLP do scikit-learn em CPU pertence ao protocolo
`cpu_development_fallback_v1` e serve apenas para desenvolvimento.

O protocolo foi congelado antes da avaliação final. Alterações em formação
dos conjuntos, pré-processamento, atributos, modelos, grades, partições,
métricas ou política de seleção exigem nova versão. Resultados de protocolos
ou hashes diferentes não devem ser agregados como repetições equivalentes.

## 2. Perguntas experimentais e experimentos concluídos

O delineamento respondeu às seguintes perguntas:

1. O pipeline generaliza para locutores não observados da própria base?
2. Um pipeline escolhido em uma base mantém desempenho em outra?
3. Combinar HUPA e SVD melhora o comportamento quando ambas as origens são
   conhecidas durante o treinamento?
4. Acrescentar /i/ e /u/ ao treinamento da SVD melhora o teste interno ou a
   transferência para a vogal /a/ da HUPA?

| Origem | Avaliação final | Material | Natureza da análise |
|---|---|---|---|
| HUPA | holdout HUPA | /a/ | interna e agrupada |
| SVD | holdout SVD | /a/ | interna e agrupada |
| HUPA | SVD completa | /a/ → /a/ | externa |
| SVD | HUPA completa | /a/ → /a/ | externa |
| HUPA + SVD | holdout misto | /a/ | interna, global e por base |
| SVD | holdout SVD | /a/, /i/, /u/ | extensão interna multivogal |
| SVD | HUPA completa | /a/, /i/, /u/ → /a/ | extensão externa multivogal |

Nos experimentos externos, nenhuma amostra, estatística ou métrica da base de
destino participou da escolha de atributos, modelo ou hiperparâmetros.

## 3. Formação dos conjuntos de amostras

Foram aceitas somente gravações que:

- podiam ser decodificadas e produziam um vetor numérico válido;
- possuíam rótulo binário e identificador de agrupamento;
- pertenciam a participantes com idade conhecida e igual ou superior a 18
  anos;
- tinham duração mínima de 0,5 segundo;
- correspondiam às vogais e condições previamente definidas.

O SHA-256 foi calculado sobre os bytes de cada arquivo para consolidar cópias
binariamente idênticas. O hash não identifica gravações apenas semelhantes,
versões reamostradas ou emissões diferentes do mesmo indivíduo.

### 3.1. HUPA

A cópia recebida continha 440 arquivos WAV da vogal /a/. Após elegibilidade e
deduplicação, permaneceram 411 amostras: 234 saudáveis e 177 patológicas, com
244 registros femininos e 167 masculinos. Doze menores de idade, quatro
registros sem idade e oito cópias com idades adultas conflitantes foram
excluídos; cinco linhas duplicadas foram consolidadas.

A versão segmentada não fornece um identificador clínico independente de
participante. Depois da deduplicação, cada arquivo canônico foi tratado como
um locutor presumidamente distinto. O procedimento impede que cópias exatas
cruzem partições, mas não prova que dois áudios diferentes pertençam a pessoas
diferentes. Essa limitação acompanha todos os resultados da HUPA.

### 3.2. SVD

A SVD foi limitada à condição normal. O conjunto univogal contém 1.997
gravações /a/, provenientes de 1.637 locutores identificados. O conjunto
multivogal contém 5.997 gravações: 1.997 de /a/, 2.001 de /i/ e 1.999 de /u/,
provenientes de 1.642 locutores.

Todas as gravações e vogais de um mesmo `speaker_id` permaneceram na mesma
partição. A extensão multivogal não cria observações independentes quando
três vogais pertencem à mesma pessoa; o locutor continua sendo a unidade de
agrupamento.

### 3.3. Treinamento misto

No conjunto HUPA + SVD, o identificador de grupo recebeu o prefixo da base:

```text
HUPA::<speaker_id>
SVD::<speaker_id>
```

Isso evita colisões nominais entre identificadores de origens diferentes. A
estratificação combina base e classe, garantindo a presença das quatro
combinações disponíveis no treinamento e no teste.

## 4. Pré-processamento compartilhado

Cada gravação foi processada individualmente, sem estimar parâmetros globais
da base:

1. leitura WAV com `SoundFile` ou NSP com `nspfile`;
2. conversão para `float32`;
3. média dos canais para mono;
4. remoção do componente DC;
5. reamostragem polifásica para 16 kHz;
6. normalização para RMS alvo de −20 dBFS;
7. limitação uniforme do ganho quando o pico ultrapassaria 0,99;
8. validação final do sinal processado.

Não houve recorte central, preenchimento, truncamento ou filtragem adicional
de frequência. O sinal completo foi usado depois do limite mínimo de duração.

## 5. Representação acústica

Foram produzidos até 596 atributos por gravação:

| Família | Quantidade | Conteúdo |
|---|---:|---|
| Harmônicos espectrais | 64 | trinta picos após HPSS, frequências, amplitudes e estatísticas |
| Distribuição da energia | 9 | posições dos limiares acumulados de 10% a 90% |
| Entropia segmentada | 58 | entropia de Shannon em sete particionamentos temporais |
| Cruzamentos por zero | 6 | posições acumuladas, total e taxa |
| MFCC, delta e delta-delta | 450 | 30 coeficientes × 3 matrizes × 5 estatísticas |
| Qualidade vocal | 9 | F0, proporção sonora, HNR, jitter e shimmer estimados pelo Praat |

A configuração espectral comum usou `n_fft=1024` e `hop_length=128` em 16
kHz. Antes dos MFCCs foi aplicada pré-ênfase de 0,97.

As nove colunas prefixadas por `glottal_` são medidas acústicas relacionadas à
qualidade e periodicidade da fonte vocal. Elas não foram calculadas a partir
de fluxo glotal obtido por filtragem inversa e não devem ser apresentadas como
uma representação direta da fonte glotal.

Foram avaliados seis cenários previamente definidos:

```text
mfcc
harmonics
energy_entropy_zcr
all_without_glottal
glottal
all_with_glottal
```

## 6. Pipeline aprendido dentro dos folds

Todas as famílias de classificadores receberam a mesma sequência:

```text
SimpleImputer(strategy="median")
    → StandardScaler()
    → SelectPercentile(f_classif, 10%, 25% ou 50%)
    → classificador
```

Imputador, padronizador e seletor foram ajustados somente no subconjunto de
treinamento de cada fold. Validação, holdout e base externa receberam apenas
as transformações já estimadas. Nenhuma linha de validação ou teste foi
criada, removida ou utilizada no cálculo desses parâmetros.

Em experimentos interbases, somente atributos acústicos numéricos presentes
nas duas bases, com os mesmos nomes e na mesma ordem, entraram na matriz. Nas
execuções concluídas, as 596 características estavam disponíveis em ambas as
bases.

## 7. Modelos e espaço de busca

### 7.1. SVM linear

- `C`: 0,01; 0,1; 1; 10;
- `class_weight="balanced"`.

### 7.2. SVM RBF

- `C`: 0,01; 0,1; 1; 10;
- `gamma`: `scale`; 0,0001; 0,001; 0,01;
- `class_weight="balanced"`.

As SVMs foram implementadas com `sklearn.svm.SVC`. A aceleração compatível
foi fornecida por `cuml.accel`.

### 7.3. MLP PyTorch/Skorch

- perfil moderado: camadas `(8,)` ou `(16,)`, dropout 0,20, weight decay
  0,001 e label smoothing 0,05;
- perfil forte: camadas `(8,)`, `(16,)` ou `(16, 8)`, dropout 0,40, weight
  decay 0,01 e label smoothing 0,10;
- épocas: 5, 10, 15 ou 20;
- lote: 32;
- taxa de aprendizado: 0,001;
- otimizador Adam;
- entropia cruzada ponderada pelas classes;
- saída com dois logits.

Os perfis são pacotes coerentes de capacidade e regularização. O protocolo
não isola causalmente o efeito individual de dropout, weight decay ou label
smoothing. O número de épocas foi tratado como hiperparâmetro porque uma
divisão interna aleatória para early stopping poderia compartilhar locutores.

## 8. Partições, validação e política de seleção

O holdout interno corresponde a aproximadamente 20% dos grupos e foi formado
uma única vez com a semente mestre 42. O algoritmo procurou, entre cinco folds
agrupados, aquele mais próximo do tamanho e da distribuição de classes
desejados. O código verificou a presença das duas classes e a ausência de
grupos compartilhados.

A seleção utilizou cinco folds de `StratifiedGroupKFold`. A acurácia
balanceada foi a métrica primária. Dentro de cada combinação de cenário de
atributos e família de classificador, candidatos suficientemente próximos do
máximo foram considerados equivalentes:

```text
tolerância = max(erro-padrão do melhor candidato, 0,005)
limiar = melhor média − tolerância
```

Entre candidatos elegíveis, a política favoreceu, nessa ordem:

1. menor desvio-padrão entre folds;
2. menor diferença absoluta entre treino e validação;
3. menor quantidade de atributos;
4. maior média de validação;
5. ordem determinística apenas como último desempate.

O campeão global e os representantes secundários SVM e MLP foram definidos
exclusivamente pela validação da origem. Depois disso, foram reajustados nos
dados de treinamento disponíveis e aplicados ao teste. Um representante
secundário numericamente superior no teste não substitui retrospectivamente o
campeão, pois isso transformaria o conjunto final em instrumento de seleção.

## 9. Diagnósticos de estabilidade e sobreajuste

A validação aninhada repetida foi executada apenas na origem:

```text
3 folds externos × 2 repetições
5 folds internos em cada seleção
sementes externas 42 e 43
```

Todos os seis resultados externos foram preservados; nenhuma repetição foi
escolhida por produzir o melhor valor. O holdout continuou intocado.

Para SVMs selecionadas, foram produzidas curvas de aprendizado com 25%, 50%,
75% e 100% dos grupos de ajuste. Para MLPs, uma divisão diagnóstica adicional,
agrupada e restrita ao treinamento, registrou perdas e acurácias por época.
Essas curvas não redefiniram o modelo final.

## 10. Métricas e incerteza

A classe patológica foi considerada positiva. Foram calculadas acurácia,
acurácia balanceada, precisão, sensibilidade, especificidade, F1, macro-F1,
MCC, ROC AUC, PR AUC e matriz de confusão.

Os intervalos de confiança de 95% utilizaram 1.000 réplicas bootstrap por
locutor. Em cada réplica, todas as gravações do grupo sorteado entraram
juntas. Réplicas sem as duas classes foram descartadas.

## 11. Resultados primários consolidados

| Origem → teste | Material | Treino/teste | Campeão | Cenário | Acurácia balanceada [IC95%] |
|---|---|---:|---|---|---:|
| HUPA → HUPA | /a/ | 329/82 | SVM RBF | MFCC | 0,786 [0,696; 0,876] |
| SVD → SVD | /a/ | 1.598/399 | SVM RBF | todos sem qualidade vocal | 0,654 [0,608; 0,698] |
| HUPA → SVD | /a/ → /a/ | 411/1.997 | SVM RBF | todos sem qualidade vocal | 0,563 [0,538; 0,587] |
| SVD → HUPA | /a/ → /a/ | 1.997/411 | MLP | todos com qualidade vocal | 0,698 [0,653; 0,741] |
| HUPA + SVD → misto | /a/ | 1.927/481 | SVM RBF | todos sem qualidade vocal | 0,733 [0,692; 0,776] |
| SVD → SVD | /a/, /i/, /u/ | 4.799/1.198 | SVM RBF | todos com qualidade vocal | 0,699 [0,655; 0,740] |
| SVD → HUPA | /a/, /i/, /u/ → /a/ | 5.997/411 | MLP | todos com qualidade vocal | 0,662 [0,617; 0,706] |

### 11.1. Configurações primárias

| Experimento | Configuração selecionada na origem |
|---|---|
| HUPA interno | `C=1`, `gamma=0,01`, retenção de 50% |
| SVD interno /a/ | `C=1`, `gamma=0,001`, retenção de 50% |
| HUPA → SVD | `C=1`, `gamma=0,01`, retenção de 50% |
| SVD → HUPA | MLP `(8,)`, perfil moderado, 10 épocas, retenção de 25% |
| Misto | `C=10`, `gamma=0,0001`, retenção de 50% |
| SVD multivogal interno | `C=1`, `gamma=0,001`, retenção de 50% |
| SVD multivogal → HUPA | MLP `(8,)`, perfil moderado, 5 épocas, retenção de 50% |

No teste misto, a SVM obteve acurácia balanceada de 0,733 tanto no recorte
HUPA quanto no recorte SVD. Isso descreve equilíbrio quando as duas bases
estão representadas no treinamento; não demonstra transportabilidade para
uma terceira origem desconhecida.

Na SVD multivogal, os resultados por vogal foram 0,697 para /a/, 0,706 para
/i/ e 0,694 para /u/. Os intervalos se sobrepõem e não sustentam a
superioridade de uma vogal.

## 12. Resultados secundários e estabilidade

Os representantes secundários produziram os seguintes resultados finais de
acurácia balanceada:

| Experimento | Representante secundário | Resultado |
|---|---|---:|
| HUPA interno | MLP | 0,708 |
| SVD interno /a/ | MLP | 0,678 |
| HUPA → SVD | MLP | 0,599 |
| SVD → HUPA | SVM RBF | 0,691 |
| Misto | MLP | 0,708 |
| SVD multivogal → HUPA | SVM RBF | 0,694 |

Esses valores avaliam a sensibilidade do resultado à família de classificador
e não constituem uma nova seleção baseada no teste.

A validação aninhada repetida resultou em:

- HUPA: 0,754 ± 0,047;
- SVD /a/: 0,724 ± 0,023;
- HUPA + SVD: 0,707 ± 0,023;
- SVD multivogal: 0,682 ± 0,014.

As curvas de aprendizado terminaram com diferenças treino–validação de 0,198
na HUPA, 0,068 na SVD /a/, 0,054 no conjunto misto e 0,058 na SVD
multivogal. Nos três conjuntos maiores, a validação ainda cresceu na fração
final, sem evidência inequívoca de saturação.

## 13. Interpretação permitida

Os resultados sustentam que atributos acústicos interpretáveis de vogais
sustentadas contêm informação discriminativa para a tarefa binária. Também
mostram que desempenho interno não garante transferência entre bases.

A direção SVD → HUPA foi mais robusta do que HUPA → SVD, mas o delineamento
não isola uma causa. Tamanho, idade, sexo, idioma, diagnóstico, equipamento,
duração e protocolo de aquisição variam simultaneamente.

O treinamento multivogal melhorou descritivamente o teste interno da SVD, mas
não produziu ganho consistente na HUPA. Os holdouts univogal e multivogal não
contêm exatamente os mesmos grupos; a comparação não é pareada nem causal.

O sistema realiza classificação experimental entre rótulos agregados. Ele
não determina uma patologia específica, gravidade, tratamento ou diagnóstico
clínico.

## 14. Ambiente e reprodutibilidade

| Componente | Configuração |
|---|---|
| Processador | AMD Ryzen 9 9900X, 12 núcleos e 24 threads |
| Memória no WSL2 | 30,2 GiB |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16.303 MiB |
| Sistema | Ubuntu 24.04.4 LTS sobre WSL2; kernel Linux 6.18.33.2 |
| Python | 3.12.3 |
| PyTorch/CUDA | PyTorch 2.12.1; CUDA 13.2 |
| Skorch | 1.4.0 |
| scikit-learn | 1.8.0 |
| RAPIDS `cuml.accel` | 26.6.0 |
| Driver NVIDIA | 610.74 |

O PyTorch solicitou algoritmos determinísticos, desativou o benchmark do
cuDNN e utilizou uma única GPU. O protocolo completo foi serializado em JSON
e recebeu um SHA-256 incorporado às métricas.

O modo `--resume-experiment` valida configuração e hash antes de reutilizar
atributos, candidatos de seleção, folds da validação aninhada e pontos das
curvas de aprendizado. Um `metrics.csv` final torna a retomada idempotente.
Tempos de execução não foram usados para comparar modelos, pois processos
independentes puderam compartilhar CPU, memória e GPU.

## 15. Execuções canônicas e hashes

| Escopo | Diretório canônico | Hash do protocolo |
|---|---|---|
| HUPA interno | `data/HUPA/experiments/20260728_135925_hupa_hupa_gpu_confirmatory_v2` | `efa9e8c4aa8fed5da11eeb6f274cd7adce03fee88c78c6566f071a60d1f5df47` |
| SVD interno /a/ | `data/SVD/experiments/20260728_193701_svd_svd_gpu_confirmatory_v2_restart1` | `efa9e8c4aa8fed5da11eeb6f274cd7adce03fee88c78c6566f071a60d1f5df47` |
| Interbases /a/ | `data/CROSS_DATABASE/experiments/20260729_091545_cross_database_cross_gpu_confirmatory_v2_checkpointed` | `efa9e8c4aa8fed5da11eeb6f274cd7adce03fee88c78c6566f071a60d1f5df47` |
| Misto | `data/POOLED/experiments/20260729_154444_pooled_pooled_gpu_confirmatory_v2_checkpointed` | `ef3977cdd252aded4b27188b20337be4870cde4425c00999893fadbe1ef0b1de` |
| SVD multivogal interno | `data/SVD/experiments/20260730_225453_svd_svd_multivowel_gpu_confirmatory_v2` | `34c91f77c2051be34f2f420ab965a83fe1e31dda238edbc1548b1bd594759a1c` |
| SVD multivogal → HUPA | `data/CROSS_DATABASE/experiments/20260730_231938_cross_database_svd_multivowel_to_hupa_gpu_confirmatory_v2` | `1f91b0a523332e9f2d57eae5a5fa73ab970987744e6f043adee3e4d1619a1240` |

Hashes diferentes são esperados quando o desenho resolvido muda, como no
treinamento misto ou na extensão multivogal. A regra é comparar o hash com o
artefato canônico do mesmo desenho, não exigir um único hash para todos os
experimentos.

## 16. Fronteira com trabalhos futuros

Não fazem parte dos resultados atuais:

- filtragem glotal inversa por QCP, IAIF ou métodos relacionados;
- fluxo glotal estimado;
- matrizes temporais de atributos glotais;
- DCA-ResNet, CNN residual multirramos, Transformer ou outra rede profunda
  sobre MFCC, delta e delta-delta preservados como matrizes;
- fusão neural entre representações acústicas e glotais;
- explicabilidade de redes profundas;
- adaptação de domínio ou validação prospectiva clínica.

Uma publicação futura poderá manter este protocolo tabular como baseline e
criar um novo protocolo para representação matricial. O desenho recomendado é
um ramo cepstral para MFCC/delta/delta-delta e um ramo glotal obtido por
filtragem inversa, com fusão intermediária ou tardia, estudos de ablação,
explicabilidade e avaliação externa intocada.
