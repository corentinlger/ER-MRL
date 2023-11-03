#!/bin/bash 

# Variables
REMOTE_DIR="username@cluster_adress/remote_dir/logs/"
DESTINATION_DIR="local_dir/logs"

rsync -av "$REMOTE_DIR" "$DESTINATION_DIR"
echo "Synchronized testing logs from remote cluster" 
