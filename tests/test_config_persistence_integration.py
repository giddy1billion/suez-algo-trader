#!/usr/bin/env python3
"""Quick integration test for configuration persistence."""

import tempfile
import os
import time
import gc
from pathlib import Path

from src.config.initializer import initialize_configuration_service, reset_configuration_service
from src.config.repository import ConfigurationRepository


def test_config_persistence():
    """Test that configs persist across simulated restarts."""
    # Create temp db
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    db_url = f'sqlite:///{path}'

    print("\n" + "="*70)
    print("CONFIG PERSISTENCE TEST")
    print("="*70)

    try:
        # --- First Run: Initialize with defaults ---
        print("\n[1] FIRST RUN: Initialize with defaults")
        reset_configuration_service()
        
        service1 = initialize_configuration_service(
            db_url, 
            seed_from_env=False, 
            auto_refresh=False
        )
        print(f"    [OK] Service initialized")
        print(f"    [OK] Cache populated: {len(service1._cache)} categories")
        print(f"    [OK] Total configs: {sum(len(v) for v in service1._cache.values())}")
        
        # Sample values
        trading_interval = service1.get_int("trading", "trading_interval")
        max_leverage = service1.get_float("risk", "max_leverage")
        print(f"    [OK] Sample defaults: trading_interval={trading_interval}, max_leverage={max_leverage}")

        # --- Modify a config ---
        print("\n[2] MODIFY: Update config value in database")
        repo = ConfigurationRepository(database_url=db_url)
        repo.set(
            category="trading",
            key="trading_interval",
            value="300",  # Changed from default 60
            value_type="int",
            changed_by="test",
            change_reason="testing_persistence",
        )
        print(f"    [OK] Updated trading_interval to 300")

        # --- Second Run: Restart and restore from DB ---
        print("\n[3] RESTART: Close service and reinitialize")
        service1_copy = service1
        del service1
        reset_configuration_service()
        gc.collect()
        time.sleep(0.2)
        
        service2 = initialize_configuration_service(
            db_url,
            seed_from_env=False,
            auto_refresh=False
        )
        print(f"    [OK] Service reinitialized (simulated restart)")
        print(f"    [OK] Restored from database")

        # --- Verify persistence ---
        print("\n[4] VERIFY: Check restored values")
        restored_interval = service2.get_int("trading", "trading_interval")
        print(f"    [OK] Restored trading_interval: {restored_interval}")
        
        if restored_interval == 300:
            print(f"\n[SUCCESS] Config persisted correctly!")
        else:
            print(f"\n[FAILURE] Expected 300, got {restored_interval}")
            return False

        # --- Third modification and restart ---
        print("\n[5] MODIFY AGAIN: Update another value")
        repo.set(
            category="risk",
            key="max_leverage",
            value="2.5",
            value_type="float",
            changed_by="test",
            change_reason="testing_persistence_2",
        )
        print(f"    [OK] Updated max_leverage to 2.5")

        print("\n[6] RESTART AGAIN: Final verification")
        del service2
        reset_configuration_service()
        gc.collect()
        time.sleep(0.2)
        
        service3 = initialize_configuration_service(
            db_url,
            seed_from_env=False,
            auto_refresh=False
        )
        
        final_interval = service3.get_int("trading", "trading_interval")
        final_leverage = service3.get_float("risk", "max_leverage")
        
        print(f"    [OK] Final trading_interval: {final_interval}")
        print(f"    [OK] Final max_leverage: {final_leverage}")
        
        if final_interval == 300 and final_leverage == 2.5:
            print(f"\n[SUCCESS] FULL TEST PASSED: All configs persisted correctly!")
            return True
        else:
            print(f"\n[FAILURE] Values don't match")
            return False

    finally:
        # Cleanup
        print("\n[CLEANUP] Removing temp database")
        del service3
        reset_configuration_service()
        gc.collect()
        time.sleep(0.2)
        
        # Try multiple times to cleanup file
        for attempt in range(3):
            try:
                Path(path).unlink(missing_ok=True)
                Path(f"{path}-wal").unlink(missing_ok=True)
                Path(f"{path}-shm").unlink(missing_ok=True)
                print("[OK] Cleanup complete")
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.2)
                    gc.collect()
                else:
                    print(f"[WARN] Cleanup failed: {e}")
                    pass


if __name__ == "__main__":
    success = test_config_persistence()
    exit(0 if success else 1)
