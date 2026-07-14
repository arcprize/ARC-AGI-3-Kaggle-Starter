from __future__ import annotations

import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

from agi_core.storage.memory_cache import drop_cache, get_cache


class SQLiteMemoryCacheTests(unittest.TestCase):
    def test_snapshot_can_be_cloned_from_worker_threads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            with closing(sqlite3.connect(db_path)) as connection:
                connection.execute("CREATE TABLE items(value INTEGER NOT NULL)")
                connection.execute("INSERT INTO items(value) VALUES(7)")
                connection.commit()

            cache = get_cache(db_path)
            self.assertIsNotNone(cache)
            assert cache is not None

            def read_value(_: int) -> int:
                with closing(cache.connect_or_fallback()) as connection:
                    return int(connection.execute("SELECT value FROM items").fetchone()[0])

            try:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    self.assertEqual(list(executor.map(read_value, range(12))), [7] * 12)
            finally:
                drop_cache(db_path)


if __name__ == "__main__":
    unittest.main()
