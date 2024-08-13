import yaml
import os

with open("conda_env.yml") as file_handle:
    environment_data = yaml.safe_load(file_handle)

with open("requirements.txt", "w") as file_handle:
    for dependency in environment_data["dependencies"]:
        if isinstance(dependency, dict):
            for lib in dependency['pip']:
                file_handle.write(f"pip install {lib}")
                file_handle.write("\n")