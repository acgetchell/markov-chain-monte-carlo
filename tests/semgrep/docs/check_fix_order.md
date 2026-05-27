Bad command ordering:

```bash
just fix
just check
```

Bad scoped command ordering:

```bash
just python-fix
just python-check
```

Good command ordering:

```bash
just check
just fix
just python-check
just python-fix
```
