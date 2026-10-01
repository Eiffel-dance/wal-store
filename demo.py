from app import WalStore
s=WalStore('store.wal'); s.set('account',100); print('commit',s.commit(),s.state)
