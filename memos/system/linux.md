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
    
### some notes

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
