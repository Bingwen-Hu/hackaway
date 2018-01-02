### 21 century Clang tool chain

- gcc, clang
- gdb
- valgrind
- gprof
- make
- pkg-config
- doxygen


### basic
```
#include <math>
```

should compile with
```
-lm
```


include external head files
```
-I/usr/local/include
```

link external libraries
```
-L/usr/local/lib
```

searching tool you have to know
```
find /usr -name 'libuseful*'
```


using pkg-config
```
pkg-config --libs gsl
pkg-config --cflags gsl
```