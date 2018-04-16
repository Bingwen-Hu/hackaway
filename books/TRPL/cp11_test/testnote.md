### prevent test in parallel
`cargo test -- test-threads=1`

### nocapture normal output
`cargo test -- --nocapture`

### run certain test
`cargo test some_Test`

### ignore test and test ignored
```
#[test]
#[ignore]
fn ...
```
ignore test
`cargo test`
run only ignored test
`cargo test -- --ignored`