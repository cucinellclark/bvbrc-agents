#!/usr/bin/env python3
"""
Main entry point for the BVBRC Consolidated MCP Server.

By default, runs in HTTP mode. Use --stdio flag to run in STDIO mode.
"""

import sys

def main():
    # Check for --stdio flag
    if "--stdio" in sys.argv:
        from stdio_server import main as stdio_main
        return stdio_main()
    else:
        from http_server import main as http_main
        return http_main()

if __name__ == "__main__":
    sys.exit(main())

