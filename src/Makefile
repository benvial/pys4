
SHELL := /bin/bash

.DEFAULT_GOAL := help


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

#### Python extension

install:
	cd src && make -s setup
	pip install .

check:
	@python -c "import S4; print(f'{S4.__name__} {S4.__version__}, {S4.__author__}: {S4.__description__}'); print('S4 installation OK!');"

## Cleaning
clean:
	$(call message,${@})
	@cd src && make -s clean
	@rm -rf S4.egg-info setup.py build

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
