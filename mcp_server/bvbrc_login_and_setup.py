#!/usr/bin/env python3
"""
Generate mcp_config.json from mcp_example.json template with correct paths.
Authenticate and set KB_AUTH_TOKEN (required).
"""

import json
import os
import requests
from pathlib import Path
from getpass import getpass


def load_config(config_path: str = "config/config.json") -> dict:
    """Load configuration from config.json"""
    config_file = Path(config_path)
    if not config_file.exists():
        return {}
    
    with open(config_file, 'r') as f:
        return json.load(f)


def authenticate(username: str, password: str, authentication_url: str) -> str:
    """
    Authenticate with BV-BRC API using the same method as oauth2_login.
    
    Args:
        username: BV-BRC username
        password: BV-BRC password
        authentication_url: Authentication endpoint URL
        
    Returns:
        Authentication token string
        
    Raises:
        SystemExit: If authentication fails
    """
    print(f"Authenticating with BV-BRC...")
    print(f"  Endpoint: {authentication_url}")
    print(f"  Username: {username}")
    
    try:
        response = requests.post(
            authentication_url,
            headers={
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            data={
                'username': username,
                'password': password
            },
            timeout=30
        )
        
        # Check if authentication was successful
        if response.status_code != 200:
            print(f"Error: Authentication failed (HTTP {response.status_code})")
            print(f"Response: {response.text}")
            return None
        
        # Parse the response to get the token
        # The token should be in the response body
        user_token = response.text.strip()
        
        if not user_token:
            print("Error: No token received from authentication endpoint")
            return None
        
        print(f"✓ Authentication successful!")
        print(f"  Token (first 20 chars): {user_token[:20]}...")
        
        return user_token
        
    except requests.RequestException as e:
        print(f"Error: Authentication request failed: {e}")
        return None
    except Exception as e:
        print(f"Error: Authentication failed: {e}")
        return None


def bvbrc_login_and_setup():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Define paths
    template_path = script_dir / "config" / "mcp_example.json"
    output_path = script_dir  / "mcp_config.json"
    manifest_template_path = script_dir / "config" / "manifest_template.json"
    manifest_output_path = script_dir / "manifest.json"
    mcp_env_path = script_dir / "mcp_env"
    stdio_server_path = script_dir / "stdio_server.py"
    config_path = script_dir / "config" / "config.json"
    
    # Check if template exists
    if not template_path.exists():
        print(f"Error: Template file not found: {template_path}")
        return 1
    
    # Check if manifest template exists
    if not manifest_template_path.exists():
        print(f"Warning: Manifest template file not found: {manifest_template_path}")
    
    # Check if mcp_env exists
    if not mcp_env_path.exists():
        print(f"Warning: mcp_env directory not found: {mcp_env_path}")
        print("You may need to run install.sh first.")
    
    # Check if stdio_server.py exists
    if not stdio_server_path.exists():
        print(f"Error: stdio_server.py not found: {stdio_server_path}")
        return 1
    
    # Read the template
    with open(template_path, 'r') as f:
        config = json.load(f)
    
    # Read the manifest template if it exists
    manifest_config = None
    if manifest_template_path.exists():
        with open(manifest_template_path, 'r') as f:
            manifest_config = json.load(f)
    
    # Replace placeholders with actual paths
    python_path = mcp_env_path / "bin" / "python3"
    
    # Update the config
    config["mcpServers"]["bvbrc-mcp"]["command"] = str(python_path)
    config["mcpServers"]["bvbrc-mcp"]["args"] = [str(stdio_server_path)]
    
    # Authenticate and set KB_AUTH_TOKEN (required)
    print("\n" + "=" * 50)
    print("BV-BVRC Login (required)")
    print("=" * 50)
    
    # Load config to get authentication URL
    app_config = load_config(str(config_path))
    authentication_url = app_config.get("authentication_url", "https://user.patricbrc.org/authenticate")
    
    # Prompt for username and password
    username = input("Username: ").strip()
    if not username:
        print("Error: Username is required")
        return 1
    
    password = getpass("Password: ")
    if not password:
        print("Error: Password is required")
        return 1
    
    # Authenticate
    token = authenticate(username, password, authentication_url)
    if not token:
        print("Error: Authentication failed. Cannot proceed without valid token.")
        return 1
    
    # Set KB_AUTH_TOKEN in the mcp_config.json
    if "env" not in config["mcpServers"]["bvbrc-mcp"]:
        config["mcpServers"]["bvbrc-mcp"]["env"] = {}
    config["mcpServers"]["bvbrc-mcp"]["env"]["KB_AUTH_TOKEN"] = token
    print(f"✓ KB_AUTH_TOKEN set in mcp_config.json")
    
    # Set KB_AUTH_TOKEN in the manifest.json if manifest template exists
    if manifest_config:
        if "server" in manifest_config and "mcp_config" in manifest_config["server"]:
            if "env" not in manifest_config["server"]["mcp_config"]:
                manifest_config["server"]["mcp_config"]["env"] = {}
            manifest_config["server"]["mcp_config"]["env"]["KB_AUTH_TOKEN"] = token
            print(f"✓ KB_AUTH_TOKEN set in manifest.json")
    
    # Write the mcp_config.json output
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Write the manifest.json output if manifest template exists
    if manifest_config:
        with open(manifest_output_path, 'w') as f:
            json.dump(manifest_config, f, indent=2)
    
    print("\n" + "=" * 50)
    print(f"Successfully generated {output_path}")
    print(f"  Python: {python_path}")
    print(f"  Server: {stdio_server_path}")
    print(f"  KB_AUTH_TOKEN: Set ✓")
    
    if manifest_config:
        print(f"\nSuccessfully generated {manifest_output_path}")
        print(f"  KB_AUTH_TOKEN: Set ✓")
    
    return 0


if __name__ == "__main__":
    exit(bvbrc_login_and_setup())

