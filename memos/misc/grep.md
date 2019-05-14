Grep something from some paths
```bash
grep -n -H -R --exclude-dir={.git} --exclude={*.py,*.md} "Something" *
```
