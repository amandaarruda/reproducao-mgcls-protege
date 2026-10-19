# Reprodução do Astronomaly Protege

Este repositório reúne o código legível e os resultados compactos da reprodução
do trabalho *Astronomaly Protege: Discovery Through Human--Machine
Collaboration* aplicada aos dados do MeerKAT Galaxy Cluster Legacy Survey
(MGCLS).

O experimento reproduzido ordena 1.031 fontes avaliadas depois de 100
consultas de aprendizado ativo. As primeiras 100 posições contêm 43 das 86
fontes classificadas como de alto interesse (`recall@100 = 50%`). O artigo
original reporta 48 de 86 fontes.

## Como executar

Há dois caminhos, com finalidades diferentes.

### Verificação rápida

Este caminho não treina o BYOL e não exige GPU nem download de imagens. Ele
confere os resultados fornecidos, recalcula as métricas e repete a etapa do
Protege a partir das características PCA já salvas.

É necessário Python 3.10 ou mais recente:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

python -m mgcls_protege check
python -m mgcls_protege metrics
python -m mgcls_protege replay
```

O comando `check` verifica dimensões, rótulos, ordenação e consistência das
métricas. `metrics` recalcula os valores publicados a partir do ranking final.
`replay` repete a etapa de aprendizado ativo do Protege usando as características
PCA fornecidas e compara o resultado com o ranking congelado.

Quando o replay informa que o resultado é equivalente, isso significa que o
conjunto das 100 primeiras fontes, a sequência de consultas, a matriz de
confusão e os escores, dentro da tolerância numérica, estão de acordo com o
resultado fornecido. Pequenas diferenças de ponto flutuante são aceitáveis;
equivalência não exige que todas as casas decimais sejam idênticas.

### Reprodução completa

O experimento completo pode ser executado a partir do catálogo e dos recortes
de imagem publicados pelos autores, em uma máquina com CUDA. Essa etapa treina
o BYOL na GPU, ajusta o PCA às características de todas as fontes e executa a
avaliação do Protege. Os dados devem permanecer fora deste repositório.

Na reprodução apresentada neste trabalho, o treinamento BYOL foi executado em
uma GPU NVIDIA A100 de 80 GB. Uma GPU NVIDIA compatível com CUDA e memória
suficiente pode ser usada em uma nova execução, mas os resultados numéricos
fornecidos aqui correspondem à execução documentada no paper.

Instale as dependências opcionais da etapa com GPU e o pacote:

```bash
python -m pip install -r requirements/full-cu121.txt
python -m pip install -e .
```

Depois, execute:

```bash
python -m mgcls_protege full \
  --catalogue /caminho/para/protege_catalogue.csv \
  --image-dir /caminho/para/images \
  --output-dir /caminho/para/resultados
```

O comando verifica a disponibilidade de CUDA antes de iniciar o treinamento.
Os valores padrão correspondem à reprodução: 100 épocas de BYOL, lote de 32,
taxa de aprendizado `5e-4`, oito workers e semente 42. O caminho completo
precisa dos recortes de imagem e de uma GPU NVIDIA compatível; a verificação
rápida não precisa desses recursos.

Para executar os testes:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Estrutura do repositório

```text
mgcls_protege/   reprodução completa, replay do Protege e validações
results/final/   resultados compactos da reprodução
results/replay/  estado salvo após cada rodada de aprendizado ativo
tests/           testes unitários e de regressão
docs/            método e resultados
requirements/    dependências opcionais da etapa com GPU
```

O experimento completo treinou o BYOL em uma GPU NVIDIA A100 de 80 GB. Depois
da extração das características, PCA e Protege foram executados como etapas
subsequentes. O caminho rápido começa nessa etapa posterior e pode ser
executado em uma CPU comum. O caminho completo é a versão portátil do
treinamento e da avaliação; o conjunto de imagens deve ser fornecido
separadamente. Scripts específicos de cluster não fazem parte deste
repositório.

As configurações experimentais estão descritas em
[`docs/METODO.md`](docs/METODO.md), e os arquivos fornecidos são apresentados
em [`docs/RESULTADOS.md`](docs/RESULTADOS.md).

## Dados

O catálogo, os rótulos e os recortes de imagem foram publicados pelos autores
do estudo original. Eles não são redistribuídos aqui, além das tabelas
compactas necessárias para verificar o resultado apresentado. Os dados de
origem e seus termos de uso estão no
[repositório original](https://github.com/MichelleLochner/mgcls.protege).

## Referência

M. Lochner e L. Rudnick, “Astronomaly Protege: Discovery Through Human--Machine
Collaboration”, *The Astronomical Journal*, vol. 169, n. 3, 2025. DOI:
[10.3847/1538-3881/ada14c](https://doi.org/10.3847/1538-3881/ada14c).
