PYTHON ?= python3

.DEFAULT_GOAL := help
.PHONY: help check metrics replay experiments figures test all

help:  ## listar os comandos disponíveis
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

check:  ## validar os resultados fornecidos
	$(PYTHON) -m mgcls_protege check

metrics:  ## recalcular as métricas apresentadas
	$(PYTHON) -m mgcls_protege metrics

replay:  ## repetir o Protege a partir das características PCA
	$(PYTHON) -m mgcls_protege replay

experiments:  ## executar as análises adicionais com os resultados congelados
	$(PYTHON) experiments/run.py

figures:  ## gerar as figuras quantitativas das análises adicionais
	$(PYTHON) experiments/plot_results.py

test:  ## executar os testes
	$(PYTHON) -m pytest -q

all: check metrics replay test  ## executar as verificações da reprodução principal
