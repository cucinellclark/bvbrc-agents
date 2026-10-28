// pm2 config for the alpha production Copilot orchestrator (host elm, account svcbvbrc).
// COPILOT_HOME is the deployment directory: BV-BRC-Copilot-API/, bvbrc-agents/,
// bvbrc-mcp/ and logs/ live there. The gateway's entry is in
// BV-BRC-Copilot-API/copilot.ecosystem.config.js.
//
// Uses the checked-in orchestrator/config/agents.yaml; only the GoWe server
// differs from dev, and GOWE_URL overrides it. The gateway shared secret is
// read from ~svcbvbrc/.copilot_internal_key (chmod 600, same file the gateway
// reads); the orchestrator refuses to start without it or without a GoWe URL.
// pm2 reads this file only when an app is started: after a change,
// `pm2 delete copilot-orchestrator` and start it again, then `pm2 save`.
const H = process.env.COPILOT_HOME || '/disks/p3/copilot-alpha';
const O = `${H}/bvbrc-agents/orchestrator`;

module.exports = {
  apps: [{
    name: 'copilot-orchestrator',
    script: `${O}/orchestrator_env/bin/python`,
    interpreter: 'none',
    cwd: O,
    args: [
      '-m', 'orchestrator',
      // elm only: the gateway on the same host is its only client
      '--host', '127.0.0.1',
      '--port', '9100',
      '--config', `${O}/config/agents.yaml`,
      '--log-level', 'info',
    ],
    env: {
      GOWE_URL: 'http://127.0.0.1:7000',
      // The server writes its own rotating log here: orchestrator-9100.log
      ORCHESTRATOR_LOG_DIR: `${H}/logs`,
      // Uncomment to send model calls through the admission proxy (llm_admission/).
      // LLM_ADMISSION_URL: 'http://127.0.0.1:8005/v1',
    },
    autorestart: true,
    max_restarts: 10,
    restart_delay: 5000,
    kill_timeout: 15000,
    // stdout/stderr: startup errors and uvicorn; the application log is above
    out_file: `${H}/logs/copilot-orchestrator.out.log`,
    error_file: `${H}/logs/copilot-orchestrator.err.log`,
    time: true,
  }],
};
