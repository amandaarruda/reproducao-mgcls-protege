# Método

A reprodução segue o experimento principal do MGCLS descrito por Lochner e
Rudnick (2025).

## Recursos computacionais da execução completa

A etapa de treinamento do BYOL foi executada uma vez em um cluster universitário
com uma GPU NVIDIA A100 PCIe de 80 GB. O job utilizou oito núcleos de CPU e
16 GB de memória do sistema. Depois da geração das características pelo BYOL,
PCA e Protege foram executados como etapas posteriores.

## Dados e rótulos

O catálogo contém 6.161 fontes de rádio. Os autores atribuíram escores de 1 a 5
a um subconjunto de avaliação com 1.031 fontes. Os escores 4 e 5 definem o
grupo de alto interesse, com 86 exemplos.

## Representação das imagens

O BYOL foi treinado uma vez nos 6.161 recortes publicados, usando um encoder
EfficientNet-B0. O treinamento utilizou 100 épocas, lote de 32, taxa de
aprendizado base `5e-4` e semente 42. O encoder produziu 1.280 características
por fonte.

O PCA foi ajustado à tabela completa de características. Foram mantidos os
menores números de componentes capazes de explicar pelo menos 95% da
variância. Nesta execução, foram mantidos 30 componentes, que explicam 95,23%
da variância. O artigo original reporta 52 componentes para sua representação.

## Dois caminhos de execução

O caminho completo treina o BYOL, ajusta o PCA e executa a avaliação do Protege
a partir dos recortes publicados. Ele é iniciado com `python -m mgcls_protege
full` e requer as dependências CUDA opcionais listadas em
`requirements/full-cu121.txt`.

O caminho rápido começa pelas características PCA fornecidas. Ele serve para
verificar o resultado publicado em uma CPU comum e é iniciado com os comandos
`check`, `metrics` ou `replay`.

## Replay do Protege

O Protege modela o escore atribuído pelo usuário com regressão por processo
gaussiano. O lote inicial contém dez fontes distribuídas ao longo do primeiro
componente principal. Depois de cada ajuste, a melhoria esperada seleciona as
dez fontes não pontuadas mais promissoras. O processo termina após 100 rótulos,
usando o mesmo valor de `epsilon = 3` do estudo original.

A interação original é reproduzida de forma offline: sempre que o algoritmo
solicita uma fonte, seu escore publicado é fornecido. Depois do ajuste final,
as 1.031 fontes de avaliação são ordenadas pelo escore previsto.

O caminho rápido começa pelas características PCA salvas e repete somente essa
etapa em CPU. O caminho completo executa o pré-processamento das imagens, o
treinamento BYOL e o ajuste do PCA antes de chegar à etapa do Protege.

## Avaliação

A medida principal é o recall entre as 100 primeiras posições. Uma fonte é
considerada de alto interesse quando seu escore de avaliação é 4 ou 5. Também
são apresentados precisão, acurácia e F1 como medidas descritivas para o mesmo
limite de 100 posições.

Esta é uma reprodução procedural de uma execução de treinamento. São usados os
recortes publicados, e não uma reconstrução a partir dos arquivos FITS brutos
do MGCLS.
