fn connect() {}
// inside module
mod server;

// Rule of module
// If a module named foo has no submodules, you should put the declarations for foo in a file named foo.rs.
// If a module named foo does have submodules, you should put the declarations for foo in a file named foo/mod.rs.