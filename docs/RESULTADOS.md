# Resultados

O resultado fornecido contém 1.031 fontes de avaliação, das quais 86 têm
escores 4 ou 5. As 100 primeiras posições contêm 43 fontes de alto interesse.

| Medida | Valor |
| --- | ---: |
| Fontes de alto interesse nas 100 primeiras | 43 de 86 |
| Recall@100 | 50,00% |
| Precisão@100 | 43,00% |
| Acurácia nas 100 primeiras | 90,30% |
| F1 nas 100 primeiras | 46,24% |

O artigo original reporta 48 das 86 fontes de alto interesse nas 100 primeiras
posições. A diferença deve ser interpretada considerando a execução única do
BYOL, o replay offline da interação e a dimensionalidade diferente do PCA
(30, em vez de 52 componentes).

## Arquivos fornecidos

| Arquivo | Conteúdo |
| --- | --- |
| `catalogue.csv` | campos do catálogo e escores de avaliação |
| `pca_features.parquet` | 30 coordenadas PCA para as 6.161 fontes |
| `pca_components.npy` | matriz de rotação do PCA |
| `active_learning_trace.csv` | fontes selecionadas em cada rodada do Protege |
| `evaluation_ranking.csv` | ranking final e escores previstos |
| `byol_loss.csv` | perda do BYOL por época |
| `protege_evaluation_metrics.json` | resumo salvo do experimento |
| `derived_metrics.json` | métricas recalculadas a partir do ranking final |
| `projection_2d.csv` | projeção bidimensional usada nas figuras do paper |

`python -m mgcls_protege check` verifica dimensões, rótulos, ordenação e
consistência das métricas. `python -m mgcls_protege replay` reconstrói de forma
independente o ranking do Protege a partir das características PCA.

## Análises adicionais

As análises complementares foram executadas sobre os mesmos resultados
congelados, sem novo treinamento BYOL. Na comparação com 100 notas fixas, a
PCA-20 e o whitening recuperaram 44 fontes de alto interesse no top-100,
enquanto a PCA-30 da reprodução principal recuperou 43. A redução para 10 e 5
componentes recuperou, respectivamente, 32 e 13. O Isolation Forest sem
rótulos recuperou quatro fontes no mesmo corte. As diferenças de um acerto
entre PCA-20, PCA-30 e whitening são pequenas; não indicam, por si só,
superioridade geral de uma configuração.

| Comparação | Hits@100 | ROC AUC | Precisão média |
| --- | ---: | ---: | ---: |
| PCA-5 + processo gaussiano | 13/86 | 0,6148 | 0,1096 |
| PCA-10 + processo gaussiano | 32/86 | 0,8214 | 0,3434 |
| PCA-20 + processo gaussiano | 44/86 | 0,8457 | 0,4197 |
| PCA-30 + processo gaussiano (principal) | 43/86 | 0,8542 | 0,4570 |
| PCA-30 com whitening + processo gaussiano | 44/86 | 0,8511 | 0,4546 |
| Isolation Forest sobre PCA-30 | 4/86 | 0,4777 | 0,0759 |

No replay ativo, as combinações PCA-20/`epsilon` 0, PCA-20/`epsilon` 3,
PCA-30/`epsilon` 0 e PCA-30/`epsilon` 3 obtiveram 36, 38, 40 e 43 acertos no
top-100. A última configuração reproduziu o resultado principal. Como a
representação e a política de consulta variam em conjunto, o replay é uma
comparação fatorial exploratória.

O SVM RBF auxiliar obteve 45 acertos com PCA-20 e 42 com PCA-30. A seleção de
hiperparâmetros usou validação cruzada de cinco partes sobre as 100 notas
fixas. Esse teste complementa a análise do processo gaussiano e não substitui
o método PROTEGE.

Rankings, métricas, tabelas de validação cruzada e checksums estão em
`results/additional/`. Os scripts que os geram e as figuras quantitativas
estão em `experiments/`. As imagens do levantamento não são redistribuídas.
