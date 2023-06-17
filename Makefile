env_project_name = MLDL_project

import_environment:
	mamba env create --name $(env_project_name) --file environment.yaml

export_environment:
	mamba env export --from-history > environment.yaml