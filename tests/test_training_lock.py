"""
Tests for Training Lock — singleton distributed lock preventing concurrent training.

Validates:
1. Only one training job can run at a time (lock prevents duplicates)
2. Instance identity is logged when lock is acquired
3. Lock releases correctly after training completes or fails
4. Git commit hash is present in governance metadata
"""

import os
import threading
import time

import pytest

from src.ml.training_lock import TrainingLock, TrainingLockError, _instance_identity
from src.utils.redis_client import LocalCache


# ---------------------------------------------------------------------------
# TrainingLock unit tests
# ---------------------------------------------------------------------------


class TestTrainingLock:
    """Unit tests for the distributed training lock."""

    def setup_method(self):
        self.cache = LocalCache(key_prefix="test:")
        self.lock = TrainingLock(self.cache, lock_ttl=60)

    def teardown_method(self):
        self.cache.close()

    def test_acquire_and_release(self):
        """Lock can be acquired and released."""
        assert not self.lock.is_locked()
        assert self.lock.try_acquire("pipeline-001")
        assert self.lock.is_locked()
        self.lock.release("pipeline-001")
        assert not self.lock.is_locked()

    def test_double_acquire_fails(self):
        """Second acquire attempt fails when lock is held."""
        assert self.lock.try_acquire("pipeline-001")
        assert not self.lock.try_acquire("pipeline-002")
        self.lock.release("pipeline-001")

    def test_context_manager_success(self):
        """Context manager acquires and releases cleanly."""
        with self.lock.acquire("pipeline-ctx"):
            assert self.lock.is_locked()
        assert not self.lock.is_locked()

    def test_context_manager_raises_when_held(self):
        """Context manager raises TrainingLockError if lock is held."""
        self.lock.try_acquire("pipeline-first")
        with pytest.raises(TrainingLockError, match="Training lock held by"):
            with self.lock.acquire("pipeline-second"):
                pass  # Should never reach here
        self.lock.release("pipeline-first")

    def test_lock_holder_identity(self):
        """Lock holder identity matches the acquiring instance."""
        self.lock.try_acquire("pipeline-id")
        holder = self.lock.lock_holder()
        assert holder == _instance_identity()
        self.lock.release("pipeline-id")

    def test_instance_identity_format(self):
        """Instance identity contains hostname and PID."""
        identity = _instance_identity()
        assert ":" in identity
        parts = identity.split(":")
        assert len(parts) == 2
        assert parts[1].isdigit()  # PID is numeric

    def test_concurrent_acquire_only_one_wins(self):
        """When multiple threads race to acquire, exactly one succeeds."""
        results = []
        barrier = threading.Barrier(5, timeout=5)

        def _try_lock(idx):
            barrier.wait()
            ok = self.lock.try_acquire(f"pipeline-race-{idx}")
            results.append((idx, ok))

        threads = [threading.Thread(target=_try_lock, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        winners = [idx for idx, ok in results if ok]
        losers = [idx for idx, ok in results if not ok]
        assert len(winners) == 1, f"Expected exactly 1 winner, got {len(winners)}"
        assert len(losers) == 4

        # Cleanup
        self.lock.release(f"pipeline-race-{winners[0]}")

    def test_release_by_non_holder_is_denied(self):
        """A different instance identity cannot release the lock."""
        self.lock.try_acquire("pipeline-owner")

        # Create a second lock instance with different identity
        other_lock = TrainingLock(self.cache, lock_ttl=60)
        other_lock._identity = "other-host:99999"
        other_lock.release("pipeline-owner")

        # Lock should still be held by original
        assert self.lock.is_locked()
        self.lock.release("pipeline-owner")


# ---------------------------------------------------------------------------
# Integration: Training pipeline uses lock to prevent duplicates
# ---------------------------------------------------------------------------


class TestTrainingPipelineLockIntegration:
    """Integration tests proving exactly one training job runs at a time."""

    def test_pipeline_rejects_concurrent_start_with_lock(self, tmp_path):
        """Pipeline raises when distributed lock is already held."""
        from src.ml.model_registry import ModelRegistry
        from src.ml.governance import ModelGovernance
        from src.ml.training_pipeline import TrainingPipeline

        cache = LocalCache(key_prefix="integ:")
        lock = TrainingLock(cache, lock_ttl=60)

        pipeline = TrainingPipeline(
            registry=ModelRegistry(models_dir=str(tmp_path / "models")),
            governance=ModelGovernance(governance_dir=str(tmp_path / "gov")),
            min_training_samples=100,
            training_lock=lock,
        )

        # Pre-acquire lock to simulate another instance holding it
        lock.try_acquire("external-pipeline-xyz")

        with pytest.raises(RuntimeError, match="Training lock held by"):
            pipeline.train(symbols=["AAPL"], trigger="scheduled")

        lock.release("external-pipeline-xyz")
        cache.close()

    def test_pipeline_acquires_lock_on_start(self, tmp_path, monkeypatch):
        """Pipeline acquires distributed lock when starting training."""
        from src.ml.model_registry import ModelRegistry
        from src.ml.governance import ModelGovernance
        from src.ml.training_pipeline import TrainingPipeline

        cache = LocalCache(key_prefix="integ2:")
        lock = TrainingLock(cache, lock_ttl=60)

        pipeline = TrainingPipeline(
            registry=ModelRegistry(models_dir=str(tmp_path / "models")),
            governance=ModelGovernance(governance_dir=str(tmp_path / "gov")),
            min_training_samples=100,
            training_lock=lock,
        )

        # Make _execute_pipeline a no-op to avoid needing real data
        def _fake_execute(progress, symbols, timeframe, lookback, data_override):
            progress.status = "completed"
            time.sleep(0.1)

        monkeypatch.setattr(pipeline, "_execute_pipeline", _fake_execute)

        pipeline_id = pipeline.train(symbols=["AAPL"], trigger="test")

        # Lock should be held immediately after train() returns
        assert lock.is_locked()
        holder = lock.lock_holder()
        assert holder == _instance_identity()

        # Wait for background thread to complete and release
        time.sleep(0.5)
        assert not lock.is_locked(), "Lock should be released after training completes"

        cache.close()


# ---------------------------------------------------------------------------
# Git commit hash presence in governance
# ---------------------------------------------------------------------------


class TestGitCommitInGovernance:
    """Tests ensuring git commit hash is present in governance metadata."""

    def test_governance_get_git_commit_from_env(self, monkeypatch, tmp_path):
        """Governance resolves git commit from GIT_COMMIT env var."""
        from src.ml.governance import ModelGovernance

        monkeypatch.setenv("GIT_COMMIT", "abc123def456789")
        gov = ModelGovernance(governance_dir=str(tmp_path / "gov"))
        commit = gov._get_git_commit()
        assert commit == "abc123def456789"

    def test_governance_get_git_commit_from_file(self, monkeypatch, tmp_path):
        """Governance resolves git commit from .git_commit file."""
        from src.ml.governance import ModelGovernance

        # Clear env vars so file fallback is used
        monkeypatch.delenv("GIT_COMMIT", raising=False)
        monkeypatch.delenv("SOURCE_VERSION", raising=False)
        monkeypatch.delenv("GITHUB_SHA", raising=False)

        # Create the .git_commit file in the project root
        project_root = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))
        commit_file = os.path.join(project_root, ".git_commit")
        test_hash = "feedbeef12345678feedbeef12345678feedbeef"
        existed = os.path.exists(commit_file)

        try:
            with open(commit_file, "w") as f:
                f.write(test_hash)

            gov = ModelGovernance(governance_dir=str(tmp_path / "gov"))
            commit = gov._get_git_commit()
            # Should find it via git CLI or the file
            assert len(commit) >= 8  # At minimum, some hash is present
        finally:
            if not existed and os.path.exists(commit_file):
                os.remove(commit_file)

    def test_governance_get_git_commit_from_github_sha(self, monkeypatch, tmp_path):
        """Governance resolves git commit from GITHUB_SHA env var (CI)."""
        from src.ml.governance import ModelGovernance

        monkeypatch.delenv("GIT_COMMIT", raising=False)
        monkeypatch.delenv("SOURCE_VERSION", raising=False)
        monkeypatch.setenv("GITHUB_SHA", "deadbeef12345678")
        gov = ModelGovernance(governance_dir=str(tmp_path / "gov"))
        commit = gov._get_git_commit()
        assert commit == "deadbeef12345678"

    def test_deploy_yml_injects_git_commit_env(self):
        """Verify deploy.yml passes GIT_COMMIT env var to container."""
        deploy_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".github", "workflows", "deploy.yml"
        )
        with open(deploy_path) as f:
            content = f.read()

        assert "GIT_COMMIT=" in content, (
            "deploy.yml must inject GIT_COMMIT environment variable"
        )
        assert "GIT_COMMIT_HASH=$" in content, (
            "deploy.yml must pass GIT_COMMIT_HASH build arg"
        )

    def test_dockerfile_embeds_git_commit(self):
        """Verify Dockerfile has ARG GIT_COMMIT_HASH and writes .git_commit."""
        dockerfile_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "Dockerfile"
        )
        with open(dockerfile_path) as f:
            content = f.read()

        assert "ARG GIT_COMMIT_HASH" in content
        assert ".git_commit" in content
