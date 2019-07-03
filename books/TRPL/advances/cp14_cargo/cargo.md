### two profile
```bash
cargo build
cargo build --release
```

### customize profile
the default settings
```toml
[profile.dev]
opt-level = 0

[profile.release]
opt-level = 3
```


### generate docs
see `docs-demo` for example

build the docs
```bash
cargo doc
```

build the docs and open in a browser
```bash
cargo doc --open
```