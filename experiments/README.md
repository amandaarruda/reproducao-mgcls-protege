# Análises adicionais

Os experimentos desta pasta partem dos arquivos congelados em `results/final`.
Não treinam novamente o BYOL nem ajustam outro PCA. Assim, podem ser executados
em CPU sem as imagens do levantamento.

```bash
python -m pip install -e .
python experiments/run.py
```

Para gerar as figuras, instale a dependência opcional e execute o segundo
script:

```bash
python -m pip install -e '.[experiments]'
python experiments/plot_results.py
```

O primeiro comando cria `results/additional/` com rankings, métricas,
configurações e checksums. Ele preserva uma pasta de saída que já contenha
arquivos; passe `--refresh` para atualizar os resultados nesse mesmo destino.
Para uma execução separada, indique outro destino:

```bash
python experiments/run.py --output-dir /tmp/mgcls-additional
python experiments/plot_results.py --results-dir /tmp/mgcls-additional
```

As análises são três: comparação de dimensões PCA com as mesmas 100 notas,
replays do PROTEGE para PCA-20/PCA-30 e `epsilon` 0/3, e um SVM RBF auxiliar
com seleção de hiperparâmetros por validação cruzada. Isolation Forest serve
como referência não supervisionada. Essas comparações complementam a
reprodução; não substituem o resultado principal nem constituem uma avaliação
independente.

As figuras geradas são quantitativas. A tabela `qualitative_cases_pca5.csv`
registra as fontes cuja posição mudou; as imagens de rádio não são copiadas
para este repositório.
