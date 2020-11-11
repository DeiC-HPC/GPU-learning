#!/usr/bin/env bash

# Setup nginx required folders
if [ ! -d /var/cache/nginx ]; then
    echo "Creating nginx cache directory"
    sudo mkdir -p /var/cache/nginx
    sudo chown -R $USER: /var/cache/nginx
fi
mkdir -p logs

exec nix-shell
