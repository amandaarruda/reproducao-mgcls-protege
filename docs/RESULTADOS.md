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
