env_project_name = MLDL_project

import_environment:
	mamba env create --name $(env_project_name) --file environment.yaml

export_environment:
	mamba env export --from-history > environment.yaml

format:
	isort --profile black -l 100 src/
	black -l 100 src/