import json
from pathlib import Path
class WalStore:
    def __init__(self,path): self.path=Path(path); self.state={}; self.commit_seq=0; self.recover()
    def _append(self,row):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open("a",encoding="utf-8") as f: f.write(json.dumps(row,sort_keys=True)+"
")
    def set(self,key,value): self._append({"op":"set","key":key,"value":value,"seq":self.commit_seq+1})
    def delete(self,key): self._append({"op":"delete","key":key,"seq":self.commit_seq+1})
    def commit(self): self.commit_seq+=1; self._append({"op":"commit","seq":self.commit_seq}); self.recover(); return self.commit_seq
    def recover(self):
        candidate={}; pending=[]; last=0
        if self.path.exists():
            for row in (json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines()):
                if row["op"]=="commit":
                    if row["seq"]<=last: raise ValueError("non-monotonic commit")
                    for p in pending: candidate.pop(p["key"],None) if p["op"]=="delete" else candidate.__setitem__(p["key"],p["value"])
                    pending=[]; last=row["seq"]
                else: pending.append(row)
        self.state=candidate; self.commit_seq=last
