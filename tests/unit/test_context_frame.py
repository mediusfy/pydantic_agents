from pydantic_agents.core.models.core import PipelineState
from pydantic_agents.core.pipeline.context import ContextFrame


def test_context_frame_serialization():
    cf = ContextFrame(
        output="test output",
        modified_files=["foo.py", "bar.py"],
        touched_symbols=["some_function"],
        schema_changes=["CREATE TABLE users"],
        memory_notes=["chose foo instead of bar"],
        command_traces=["all tests passed"],
    )

    dumped = cf.model_dump()
    assert dumped["output"] == "test output"
    assert dumped["modified_files"] == ["foo.py", "bar.py"]

    # Test JSON serialization/deserialization
    js = cf.model_dump_json()
    loaded = ContextFrame.model_validate_json(js)
    assert loaded.output == "test output"
    assert loaded.modified_files == ["foo.py", "bar.py"]


def test_pipeline_state_with_context():
    state = PipelineState(run_id="run_123", stage="init")
    assert state.active_context is None
    assert len(state.context_history) == 0

    cf = ContextFrame(output="Step completed")
    state.active_context = cf
    state.context_history.append(cf)

    dumped = state.model_dump()
    assert dumped["active_context"]["output"] == "Step completed"
    assert len(dumped["context_history"]) == 1
