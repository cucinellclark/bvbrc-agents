module.exports = {
  apps: [
    {
      name: "bvbrc-mcp-server",
      script: "http_server.py",
      interpreter: "/home/ac.cucinell/bvbrc-dev/Copilot/BVBRC-MCP-Servers/bvbrc-mcp-server/mcp_env/bin/python3",
      cwd: "/home/ac.cucinell/bvbrc-dev/Copilot/BVBRC-MCP-Servers/bvbrc-mcp-server",
      autorestart: true,
      output: "/home/ac.cucinell/bvbrc-dev/Copilot/BVBRC-MCP-Servers/bvbrc-mcp-server/logs/http_server.log",
      error: "/home/ac.cucinell/bvbrc-dev/Copilot/BVBRC-MCP-Servers/bvbrc-mcp-server/logs/http_server.error.log"
    }
  ]
};

