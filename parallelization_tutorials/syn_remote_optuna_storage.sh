#!/bin/bash 

# Variables
REMOTE_DIR="username@cluster_adress/remote_dir/"
REMOTE_FILES_PATTERN="optuna-journal*"
DESTINATION_DIR="local_dir"

rsync -av "$REMOTE_DIR/$REMOTE_FILES_PATTERN" "$DESTINATION_DIR"
echo "Synchronized optuna storages from remote cluster" 
