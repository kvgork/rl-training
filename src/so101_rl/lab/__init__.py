"""so101_rl.lab: the minimal AI training connector (plans/mvp-plan.md M5).

Run registry (``runs``), trainer plugins (``trainers``), the job entry point (``job``), the launcher (transient
``systemd --user`` units or detached processes, ``launcher``), the Python API (``api``), the stdio MCP server
(``mcp_server``) and a human CLI (``cli``). Sim only; never imports the safety clamps, the servo core or serial.
"""
