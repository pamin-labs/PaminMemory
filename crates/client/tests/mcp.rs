//! `pamin mcp` passes pamin-kit's conformance suite: valid 2020-12 schemas, no
//! tenant, user or project argument, calls without an identity refused, and
//! every tool named in the MCP skill. The served tool list is snapshotted.

use memory_client::mcp::Memory;

#[test]
fn memory_conforms() {
    let home = tempfile::tempdir().unwrap();
    let memory = Memory::new(
        memory_core::Workspace::at(home.path()),
        "memory".into(),
        Default::default(),
        Vec::new(),
    );
    pamin_kit::conformance!(memory, "../../.agents/skills/pamin-memory-mcp/SKILL.md");
}
