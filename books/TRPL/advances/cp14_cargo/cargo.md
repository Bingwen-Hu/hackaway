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


### publish your crate
Firstly, you have to register in crate.io through github. Then, you log in cargo locally. After you have setup your license and description, you can publish your crate by following command.
```bash
cargo publish
```

### yank your crate
Yank your crate means new projects could not use it as dependence.
```bash
cargo yank --vers 1.10.1
```
undo yanking
```bash
cargo yank --vers 1.10.1 --undo
```
