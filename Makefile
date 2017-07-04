#
# jade Makefile
#
# @author <bprinty@asuragen.com>
# ------------------------------------------------------


# config
# ------
VERSION    = `python -c 'import jade; print jade.__version__'`


# targets
# -------
.PHONY: docs clean tag

help:
	@echo "clean    - remove all build, test, coverage and Python artifacts"
	@echo "lint     - check style with flake8"
	@echo "test     - run tests quickly with the default Python"
	@echo "docs     - generate Sphinx HTML documentation, including API docs"
	@echo "release  - package and upload a release"
	@echo "build    - package module"
	@echo "install  - install the package to the active Python's site-packages"

clean:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

lint:
	flake8 jade tests

test: test-py2 test-py3

test-py2:
	@echo "Running python2 tests ... "
	if [ ! -e .py2 ]; then virtualenv .py2; fi
	. .py2/bin/activate && \
	pip install nose nose-parameterized && \
	pip install -r requirements.txt && \
	python setup.py test

test-py3:
	@echo "Running python3 tests ... "
	if [ ! -e .py3 ]; then virtualenv -p python3 .py3; fi
	. .py3/bin/activate && \
	pip install nose nose-parameterized && \
	pip install -r requirements.txt && \
	python setup.py test

tag:
	VER=$(VERSION) && if [ `git tag | grep "$$VER" | wc -l` -ne 0 ]; then git tag -d $$VER; fi
	VER=$(VERSION) && git tag $$VER -m "jade, release $$VER"

docs:
	cd docs && make html

release: tag
	VER=$(VERSION) && git push origin :$$VER || echo 'Remote tag available'
	VER=$(VERSION) && git push origin $$VER
	python setup.py sdist upload -r internal
	python setup.py bdist upload -r internal
	python setup.py bdist_wheel upload -r internal

build: clean
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean
	python setup.py install
