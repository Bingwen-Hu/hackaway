### File I/O

    File access mode flags, can be retrieved by fcntl F_GETFL
    O_RDONLY  0  ==> flags = fcntl(fd, F_GETFL)
    O_WRONLY  1      accessMode = flags & O_ACCMODE
    O_RDWR    2      accessMode == O_RDONLY ? O_RDONLY :
                     accessMode == O_WRONLY ? O_WRONLY : O_RDWR

    File creation flags, can't be retrieved or changed
    O_CLOEXEC    Set the close-on-exec flag
    O_CREAT      Create file if it doen't exist
    O_DIRECTORY  Fail if pathname is not a directory
    O_EXCL       With O_CREAT: create file exclusively
    O_LARGEFILE  Used on 32-bit systems to open large files
    O_NOCTTY     Don't let pathname become controlling terminal
    O_NOFOLLOW   Don't dereference symbolic links
    O_TRUNC      Truncate existing file to zero length

    Open file status flags, can be retrieved and changed by fcntl
    O_APPEND     Write are always appended to end of file
    		 im: even invoke lseek!
    O_ASYNC      Generate a signal when I/O is possible
    O_DIRECT     File I/O bypassed buffer cache
    O_DSYNC      Provide synchronized I/O data integrity
    O_NOATIME    Don't update file last access time on read()
    		 im: useful when do some backup or indexing
    O_NONBLOCK   Open in nonblocking mode
    O_SYNC       Makefile writes synchronous
    ==
    ==> flags & O_SYNC to see whether the flags bit is set!
    
### Some notes

    fcntl is a very fancy function! it can be used to:
    - retrieve access mode of the file using F_GETFL
    - set the flags status of the file using F_SETTL
    - dup fd! using fcntl(oldfd, F_DUPFD, startfd)

    pread and pwrite can be used to achieve higher performance
    so as preadv and pwritev.

    using mkstemp to create a temporary file
    char template[] = "/tmp/somestringXXXXXX";
    fd = mkstemp(template);
    //some code ...
    unlink(template); 

### Program

    extern char etext, edata, end; // in C
    Note that the uninit-data-segment is allocated space in run time!
    etext       the boundary address of text-segment
    edata       the boundary address of init-data-segment
    end         the boundary address of uninit-data-segment, so as the beginning
    	        address of heap

    extern char **environ; // in C
    environ     contains environment variable in the format home=/home/mory
                every key=value pair is a string (end with NULL)
    relative functions:
    getenv, putenv, setenv, unsetenv /* not standard */ clearenv
    
    
### Memory

    base function:
    brk        set the program break to the location specified by the param.
    sbrk       increase the number of heap according to the param
    malloc family:
    malloc     accept a size_t and return the pointer of the block
    calloc     accept size_t and num to alocate a series of blocks
    memalign   align the memory block
    alloca     allocate memory on stack, should not be call with free useful
               when calling setjmps.
    free       free the memory allocate by the malloc family except alloca

### Users and Groups

    the password file : /etc/passwd
    relative functions: getpwnam, getpwuid
    struct passwd:
    username: pw_name
    password: pw_passwd
    user ID : pw_uid
    Group ID: pw_gid
    Comments: pw_gecos
    home Dir: pw_dir
    login Sh: pw_shell
    Note: return value: pwd==NULL & errno==0 ==> Not found

    the group file : /etc/group
    relative functions: getgrnam, getgrgid
    struct group:
    grp name: gr_name
    password: gr_passwd
    group ID: gr_gid
    membeLst: gr_mem

    the shadow file : /etc/shadow
    relative functions: getspnam
    struct spwd:
    username: sp_namp
    encrypwd: sp_pwdp
    lastchg : sp_lstchg
    mindays : sp_min
    maxdays : sp_max
    warndays: sp_warn
    inactive: sp_inact
    expired : sp_expire
    reserved: sp_flag

    other functions:
    crypt: encrypted password
    getpass: ask user to input password
    getpwent/getgrent/getspent: beginning iterate the specified file
    setpwent/setgrent/setspent: set the pointer to the beginning
    endpwent/endgrent/endspent: close the iteration
    
### Process credentials

    real-user-ID indicates to whom the file/program belongs to
    effective-UID determines the permission when program runs
    saved-user-ID initially same as effective-UID

    NOTE: saved-user-ID is designed for use with set-user-id programs
    a example program is provide in /hackaway/memos/C/proc-cred.c
    to see the result. first compile it normally and the result is identical
    then run the following commands you can see the change.
    linux:~> sudo gcc proc-cred.c -o proc-cred // make owner root
    linux:~> sudo chmod u+s proc-cred // set-user-ID
    
    file-system-id is always the same as EUID except calling specific sys-call
    functions:
    getuid/geteuid/getgid/getegid
    setuid/setgid/seteuid/setugid/setreuid/setregid
    linux-specific-functions: setresuid/setresgid/setfsuid/setfsgid
    
