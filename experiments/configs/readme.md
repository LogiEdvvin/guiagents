# Configurations

This folder stores config files used by different scripts. You can use these configuration files as templates for your modifiations.

# Default loading
If the script as well as the configuration file are still in the project's original structure then you do not need to specify the location of the configuration files becuase all of the scripts load their appropriate config by looking inside of this folder. Just the config you need to and do not rename it.

# secrets.yaml

If you look inside the .gitignore you might notice a secrets.yaml file. This is a yaml file used to store secrets like passwords or API keys. If you want to use a secrets.yaml you can just rename the example_secrets.yaml file (it contains only placeholders) to secrets.yaml and you will no longer need to add API keys to your environment variables.