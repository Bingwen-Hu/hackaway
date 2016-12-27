// scala basic
// warning: Lisp is most beautiful!


// ==================== Variable ====================
var mutable = "string"
val immutable = "Mory"
mutable = mutable.drop(3)
immutable = immutable.drop(3) // error!

// the value of a variable can be change but not the type!
var imInt: Int = 4
imInt = 10
imInt = "wrong!"

// every thing has a value
val happy = false
val x = if (happy) "I am happy" else "so sad" // x = "so sad"

// type convertion
val imInt: Int = 4
val imDouble: Double = imInt
val imInt2: Int = imDouble.toInt

// ==================== Function ====================
// every parameter must have a type but the return type is not necessary
// unless the function is recursive
def abs(x: Double) = if (x >= 0) x else -x
def fac(n: Int) = {
  var r = 1
  for (i <- 1 to n) r = r * i
  r
}

// a recursive version of fac
def facr(n: Int): Int = if (n <= 0) 1 else n * facr(n-1)

// special function we call it Procedure
// Procedure has no return value. we just need its side-effect
def box(s: String) {
  var border = "-" * s.length + "--\n"
  print(border + "|" + s + "|\n" + border)
}

// ==================== Control ====================
// the Python-like for
// Note that the i need not declare
for (i <- 1 to 10) yield i
for (i <- (0 to 10).reverse) yield i

// look at this two, what makes them different?
for (c <- "hello"; i <- 0 to 1) yield (c + i).toChar
for (i <- 0 to 1; c <- "hello") yield (c + i).toChar
// hieflmlmop
// Vector(h, e, l, l, o, i, f, m, m, p)
// first: h => ascii-Int then plus 1 => char so i
// second: why the result is vector?

// Guard is supported!
// Note that UNTIL does not contain the last index, TO does
for (i <- 1 until 10 if i * i < 10) yield i

// C-style while
var n = 10
var r = 0
while (n > 0 ) {
  r += n
  n -=1
}

// ==================== Array ====================
// Buffer is mutable and Array is not
// always use like this
import scala.collection.mutable.ArrayBuffer
val b = ArrayBuffer[Int]() // () means init value (1) is ok!
b += 2
b += 3
nb = b.toArray // so get an Array

// ==================== Class hierarchy ====================
// Any
// ├── AnyRef
// │   ├── Classes       |
// │   ├── Collections   |── Null ──── Nothing (All)
// │   └── String        |
// └── AnyVal
//     ├── Boolean
//     ├── Char
//     └── Numeric-Type
//
// Nothing is the type of return
// Null is the type of an empty String


// ==================== Pattern Match ====================
val y = 1
val x = 2

val right = x > y match {
  case true => x
  case false => y
}

// the name sa can be anything valid -- Just represent the status!!
val status = "ok"
val message = status match {
  case sa if sa != null => println(s"Received '$sa'")
  case other => println("what a error")
}

// ==================== String interpolation ====================


