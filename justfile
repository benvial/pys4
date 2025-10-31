
PROJECT_NAME := "pys4"

BRANCH := "$(git branch --show-current)"

PROJECT_DIR := "$(realpath $PWD)"

VERSION := """$(python3 -c "import toml; print(toml.load('pyproject.toml')['project']['version'])")"""


version:
    @echo {{VERSION}}

clean:
    rm -rf builddir build .coverage *.egg-info doc/build .pytest_cache \
    htmlcov .ruff_cache wheelhouse builddir dist
    cd examples && rm -rf  *.txt *.log *.npz *.png *.pdf *.csv

cleandoc:
    cd doc && just clean


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
    pip install .

install-edit:
    pip install -e . --no-build-isolation

install-conda:
    pip install . -Csetup-args="-Dconda=true"

install-conda-edit:
    pip install -e . --no-build-isolation -Csetup-args="-Dconda=true"

check:
	@python -c "from pys4 import _S4; print(f'{_S4.__name__} {_S4.__version__}, {_S4.__author__}: {_S4.__description__}'); print('S4 installation OK!');"
	@python -c "import pys4; print(f'{pys4.__name__} {pys4.__version__}, {pys4.__author__}: {pys4.__description__}'); print('pys4 installation OK!');"


docker-build:
    docker build . -t pys4


# Build the documentation
doc:
    cd doc && just build

doc-noplot:
    cd doc && just build-noplot

# Build the documentation
autodoc:
    cd doc && just autobuild


show:
    firefox doc/build/html/index.html

api:
    cd doc && just api