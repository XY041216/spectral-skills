# Execution Backend Rules

Spectral Skills use MCP first and scripts fallback second.

## Backend Selection

1. Check `server_health` when an MCP server is declared.
2. Use MCP when available and healthy enough for the task.
3. Use scripts fallback when MCP is unavailable but script preflight passes.
4. Use `agent_guided` only for planning or documentation steps that do not
   produce numeric artifacts.
5. Use `manual_confirmed` only to record explicit user decisions, not computed
   results.

## Execution Records

All execution paths must write an execution record containing backend,
server name, tool chain, fallback status, fallback reason, core version, schema
version, timestamp, Python executable, working directory, plugin root, warnings,
and errors.

## Script Rules

Scripts should:

- accept command-line arguments;
- accept JSON config when needed;
- return JSON output;
- avoid hidden side effects;
- require user or caller provided output directories for file writing;
- write only declared artifacts.

## Output Directory Rule

Tools and scripts must not silently create output in surprising locations. If a
directory is required and not provided, ask for one or return a blocked
preflight result.
