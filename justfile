
PROJECT_NAME := "pys4"

BRANCH := "$(git branch --show-current)"

PROJECT_DIR := "$(realpath $PWD)"

VERSION := """$(python3 -c "import toml; print(toml.load('pyproject.toml')['project']['version'])")"""


version:
    @echo {{VERSION}}

doc:
    cd doc && make html

show:
    firefox doc/build/html/index.html


clean:
    rm -rf builddir build .coverage *.egg-info doc/build .pytest_cache \
    htmlcov .ruff_cache wheelhouse builddir dist
    cd examples && rm -rf  *.txt *.log *.npz *.png *.pdf *.csv
    cd doc && make clean


set:
    meson setup --wipe builddir

bld:
    meson compile -C builddir

test-import:
    cd builddir/src/S4/S4 && python -c "import S4"


meson: set bld test-import

# Push to github
gl:
    @git add -A
    @read -p "Enter commit message: " MSG; \
    git commit -a -m "$MSG"
    @git push origin {{BRANCH}}


# Clean, reformat and push to github
save: gl


test:
    pytest


install:
    pip install . --no-build-isolation

install-edit:
    pip install -e . --no-build-isolation

check:
	@python -c "from pys4 import S4; print(f'{S4.__name__} {S4.__doc__}'); print('S4 installation OK!');"
	@python -c "import pys4; print(f'{pys4.__name__} {pys4.__version__}, {pys4.__author__}: {pys4.__description__}'); print('pys4 installation OK!');"


docker-build:
    docker build . -t pys4