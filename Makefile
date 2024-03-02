
SHELL := /bin/bash

.DEFAULT_GOAL := help

.PHONY: doc test dev

#################################################################################
# GLOBALS                                                                       #
#################################################################################

message = @make -s printmessage RULE=${1}

printmessage: 
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/^/---/" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} | grep "\---${RULE}---" \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=0 \
		-v col_on="$$(tput setaf 4)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s ", col_on, -indent, ">>>"; \
		n = split($$3, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i] ; \
		} \
		printf "%s ", col_off; \
		printf "\n"; \
	}' 

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Build C extension and install S4 Python package
install-S4:
	$(call message,${@})
	@cd src && make -s install

## Build C extension and install S4 Python package
install:
	$(call message,${@})
	@pip install .

	
## Check the installation by running some simple tests
check:
	$(call message,${@})
	@cd src && make -s check
	@python -c "import pyS4; print(f'{pyS4.__name__} {pyS4.__version__}, {pyS4.__author__}: {pyS4.__description__}'); print('pyS4 installation OK!');"


## Cleaning
clean:
	$(call message,${@})
	@cd src && make -s clean
	@cd doc && make -s clean

## Build html documentation (live reload)  
livedoc:
	$(call message,${@})
	sphinx-autobuild -a doc doc/build/html --watch examples/ \
    --watch doc/_static/  --port=8001 --open-browser --delay 1 \
    --re-ignore 'doc/examples/*'

## Build html documentation (only updated examples)
doc:
	$(call message,${@})
	cd doc && make -s html

## Show html documentation
show-doc:
	$(call message,${@})
	xdg-open doc/build/html/index.html

# Build html documentation (except examples)
doc-noplot:
	$(call message,${@})
	cd doc && make -s html-noplot


## Install requirements for doc
doc-req:
	$(call message,${@})
	@cd doc && pip install --upgrade -r requirements.txt

## Install requirements for testing
test-req:
	$(call message,${@})
	@cd test && pip install --upgrade -r requirements.txt


## Run the test suite
test:
	$(call message,${@})
	pytest ./test

## Install Python dependencies for development
dev:
	@pip install -r dev/requirements.txt


## Reformat code
style:
	$(call message,${@})
	@isort -l 88 .
	@black -l 88 .

## Push to remote
remote:
	$(call message,${@})
	@git add -A
	@read -p "Enter commit message: " MSG; \
	git commit -a -m "$$MSG"
	@git push origin master

## Format and push to remote
save: style remote
	$(call message,${@})
	
python:
	cd src && make install && make clean

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################


# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>

help:
	@echo -e "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo -e
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
