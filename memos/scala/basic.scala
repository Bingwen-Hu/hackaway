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
// in fact we have seen it last segment

val approx = 355/113f
println("Pi, using 355/113, is about " + approx + ".")

// using string interpolation
// using braces when necessary 
println(s"Pi, using 355/113 is about $approx")

val fruit = "apple"
println(s"how do you like ${fruit}s")

// it seems that f-literal is larger than s-literal
println(f"how do you like $fruit%.3s")
println(f"Pi, using 355/113 is about $approx%.3f")


// ==================== Regular Expressions ====================
// Regular in Scala is very different from others!

val input = "Enjoying this apple 3.1415926 times today"
val pattern = """.* apple ([\d.]+) times .*""".r // look there is a 'r' in tail
val pattern(amountText) = input                  // very strange usage!
val amount = amountText.toDouble

// ==================== Type operations ====================

5.asInstanceOf[Long] // convert!
4.getClass           // show!
4.isInstanceOf[Int]  // boolean!
'A'.hashcode         // useful in hashbase functions
20.toDouble          // convert!
20.toString          // number to string
List(1, 2, 3.0).toVector

// ==================== Tuple & Map ====================
val info = (5, "Korben", true)
val name = info._2

// another way using '->'

val red = "red" -> "0xFF0000"
val reversed = red._2 -> red._1

val m = Map("AAPL" -> 597, "MSFT" -> 40)
val n = m - "AAPL" + ("GOOG" -> 123)

// mutable and immutable
// collection.immutable.List -> collection.mutable.Buffer
// collection.immutable.Set -> collection.mutable.Set
// collection.immutable.Map -> collection.mutable.Map

// ==================== Function ====================
// basic
def max(x: Int, y: Int): Int = if (x >= y) x else y

// Type parameter 
def identity[A](a: A): A = a

// nest functions
// the underscore indicate func is a function
def addfunction(x: Int) = {
  def func(added: Int) = added + x
  func _
}
// the same as 
def addfunction(x: Int) = {
  def func(added: Int) = added + x
  func(_)
}

// default and name parameters
def greet(prefix: String, name: String) = s"$prefix $name"
greet(prefix = "boy", name = "mory")

def greet(prefix: String = "Boy", name: String) = s"$prefix $name"
greet(name = "Mory")

// vararg parameters
def isum(xs: Int*) = {
  var total = 0
  for (x <- xs) total += x
  total
}
isum(1, 2, 3, 5)

// parameter groups and curry function
// largerThan7 curry one parameter and become a partially function
// wait for another parameter to get in
def max(x: Int)(y: Int) = if (x >= y) x else y
def largerThan7: Int => Int = max(7) _


// high order functions
def safeStringOps(s: String, f:String => String) = {
  if (s != null) f(s) else s
}
safeStringOps("Good", _.reverse)

// function literals with placeholder
val doubler : Int => Int = (x: Int) => x * 2
val doubler2: Int => Int = _ * 2


// By-name parameter
// by-name parameter is confusing but is useful when we want a parameter can
// accept either a value or a function
// when a value is passed to a by-name parameter, nothing special
// when a fucntion is passed than every time the function is referred, it called

def doubles(x: => Int) = { // special syntax
  println("Now doubling " + x)
  x * 2
}

def f(i: Int) = { println(s"Hello from f($i)"); i}



// finally part of function: high-order with literal

def safeStringOps(s: String, f:String => String) = {
  if (s != null) f(s) else s
}
val uuid = java.util.UUID.randomUUID.toString
val timeUUID = safeStringOps(uuid, {s =>
  val now = System.currentTimeMillis
  val timed = s.take(24) + now
  timed.toUpperCase
})

// using parameter group we can simplify
val timeUUID = safeStringOps(uuid) {s =>
  val now = System.currentTimeMillis
  val timed = s.take(24) + now
  timed.toUpperCase
}

// another example using by-name parameter
// show the beauty of scala
def timer[A](f: => A): A = {
  def now = System.currentTimeMillis
  val start = now; val a = f; val end = now
  println(s"Excuted in ${end - start} ms")
  a
}

val veryRandomAmount  = timer {
  util.Random.setSeed(System.currentTimeMillis)
  for (i <- 1 to 100000) util.Random.nextDouble
  util.Random.nextDouble
}
