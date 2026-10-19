PYTHON ?= python3

.DEFAULT_GOAL := help
.PHONY: help check metrics replay test all

help:  ## listar os comandos disponíveis
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

check:  ## validar os resultados fornecidos
	$(PYTHON) -m mgcls_protege check

metrics:  ## recalcular as métricas apresentadas
	$(PYTHON) -m mgcls_protege metrics

replay:  ## repetir o Protege a partir das características PCA
	$(PYTHON) -m mgcls_protege replay

test:  ## executar os testes
	$(PYTHON) -m pytest -q

all: check metrics replay test  ## executar todas as verificações locais
