## Organize your Rust code in workspace

1. create a directory for workspace
2. create a `Cargo.toml` for it, here is an example

    ```toml
    [workspace]

    members = [
        "adder", 
        "add-one",
    ]
    ```
3. you can add as many as library you like
    ```bash
    # create a binary package
    cargo new adder
    # create a library package
    cargo new add-one --lib
    ```

4. you can run `cargo build` now! A new `Cargo.lock` and `target` directory will be generated.

## Run specified package
```bash
cargo run -p adder
```