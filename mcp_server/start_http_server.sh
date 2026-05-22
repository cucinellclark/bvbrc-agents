#!/bin/bash
source mcp_env/bin/activate
PORT=$(jq -r '.port' config/config.json) python3 http_server.py
