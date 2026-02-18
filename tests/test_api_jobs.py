from casadei.api.jobs import JobManager


def test_create_job():
    mgr = JobManager()
    job_id = mgr.create("product-1", "gen-1")
    state = mgr.get(job_id)
    assert state is not None
    assert state.status == "pending"


def test_job_progress_updates():
    mgr = JobManager()
    job_id = mgr.create("p1", "g1")
    mgr.update_progress(job_id, 0.5, "Running step 1")
    state = mgr.get(job_id)
    assert state.progress == 0.5
    assert state.message == "Running step 1"


def test_job_completion():
    mgr = JobManager()
    job_id = mgr.create("p1", "g1")
    mgr.complete(job_id)
    state = mgr.get(job_id)
    assert state.status == "completed"
    assert state.progress == 1.0


def test_job_failure():
    mgr = JobManager()
    job_id = mgr.create("p1", "g1")
    mgr.fail(job_id, "GPU out of memory")
    state = mgr.get(job_id)
    assert state.status == "failed"
    assert state.error == "GPU out of memory"
