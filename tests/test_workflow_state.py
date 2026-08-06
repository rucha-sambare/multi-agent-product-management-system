from AI_Product_Manager.workflow_state import WorkflowStore


def test_workflow_store_round_trip(tmp_path):
    store = WorkflowStore(tmp_path / "state.sqlite3")
    run_id = store.create("Example", "com.example")
    store.update(
        run_id, status="needs_review", step="finished", payload={"reviews": 10}
    )
    run = store.get(run_id)
    assert run["status"] == "needs_review"
    assert run["payload"]["reviews"] == 10

