
.PHONY: all tests dist docs

all: sac

# Demo
reinforce:
	python examples/reinforce_cartpole.py

ac:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python examples/actor_critic_cartpole.py

grid:
	python examples/actor_critic_gridworld.py

# Atari
dist-a2c:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python -m torch.distributed.launch \
	          --nproc_per_node=6 \
		    examples/atari/dist_a2c_atari.py

a2c:
	OMP_NUM_THREADS=4 \
	MKL_NUM_THREADS=4 \
	python examples/atari/a2c_atari.py

ppoa:
	OMP_NUM_THREADS=4 \
	MKL_NUM_THREADS=4 \
	python examples/atari/ppo_atari.py

bug:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python examples/atari/debug_atari.py

dqn:
	python examples/atari/dqn_atari.py

# PyBullet
dist-ppo:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python -m torch.distributed.launch \
	          --nproc_per_node=16 \
		    examples/pybullet/dist_ppo_pybullet.py

ppo:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python examples/pybullet/ppo_pybullet.py

sac:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python examples/pybullet/sac_pybullet.py

tsac:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python examples/pybullet/delayed_tsac_pybullet.py

# Tabular
tabular-s:
	python examples/tabular/sarsa.py

tabular-q:
	python examples/tabular/q_learning.py

# bsuite

bsuite:
	python examples/bsuite/trpo_v_random.py

# Admin
dev:
	pip install --progress-bar off torch gym pycodestyle >> log_install.txt
	python setup.py develop

lint:
	pycodestyle cherry/ --max-line-length=160

lint-examples:
	pycodestyle examples/ --max-line-length=80

lint-tests:
	pycodestyle tests/ --max-line-length=180

tests:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python -W ignore -m unittest discover -s 'tests/' -p '*_tests.py' -v
	make lint

docs:
	cd docs && pydocmd build && pydocmd serve

docs-deploy:
	cd docs && pydocmd gh-deploy

# https://dev.to/neshaz/a-tutorial-for-tagging-releases-in-git-147e
release:
	echo 'Do not forget to bump the CHANGELOG.md'
	echo 'Tagging v'$(shell python -c 'print(open("cherry/_version.py").read()[15:-2])')
	sleep 3
	git tag -a v$(shell python -c 'print(open("cherry/_version.py").read()[15:-2])')
	git push origin --tags

publish:
	python setup.py sdist  # Create package
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*  # Push to PyPI
