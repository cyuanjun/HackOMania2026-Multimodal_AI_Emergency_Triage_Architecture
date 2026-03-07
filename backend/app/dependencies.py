from app.repositories.seed_store import SeedStore


class StoreProxy:
    def __init__(self) -> None:
        self._store: SeedStore | None = None

    def _get_store(self) -> SeedStore:
        if self._store is None:
            self._store = SeedStore()
        return self._store

    def __getattr__(self, name: str):
        return getattr(self._get_store(), name)


store = StoreProxy()
