# llm_admission — OpenAI-compatible admission proxy for BV-BRC Copilot.
#
# Sits on holly between the orchestrator/agents and vLLM on mango.
# Gates /v1/chat/completions through a bounded semaphore (max_in_flight)
# with a bounded wait queue (max_waiting). All other paths pass through.
