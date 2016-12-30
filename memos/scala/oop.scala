/**
  * Object-orented in Scala
  */

// a basic class
class User {
  val name: String = "Yubaba"
  def greet: String = s"Hello from $name"
  override def toString = s"User($name)"
}

// a parameter class
// the parameter with var|val is considered as member of the class
class User(val name: String) {
  def greet: String = s"Hello from $name"
  override def toString = s"User($name)"
}

// some usage
val users = List(new User("Mory"), new User("Anna"), new User("Sky"))
val sizes = users map (_.name.size)
val sorted = users sortBy (_.name)
val Anna = users find (_.name contains "a")
val greet = Anna map (_.greet) getOrElse "hi"

// inheritance and polymorphism
class A {
  def hi = "Hello from " + getClass.getName
  override def toString = getClass.getName
}
class B extends A
class C extends B {override def hi = "hi C -> " + super.hi}

val hiA = new A().hi
val hiB = new B().hi
val hiC = new C().hi

// becaus class A is base of B and B of C, so
val a: A = new B // is OK, but a.hi invoke the one in class B (of course!)
val b: B = new A // is not OK

// let us combine the things above!
class Car (val make: String, var reversed: Boolean) {
  def reserve(r: Boolean): Unit = {reversed = r}
}
class Lotus(val color: String, reserved: Boolean) extends Car("Lotus", reserved)

val l = new Lotus("silver", false)
println(s"Requested a ${l.color} ${l.make}")

// class can have default values of course
// also type parameters

// only because it extends the Traversable it gains a lot of functions
class Singular[A](element: A) extends Traversable[A] {
  def foreach[B](f: A => B) = f(element)
}
val p = new Singular("Planes")
p foreach println
val name: String = p.head


// Abstract Class
abstract class Car {
  val year: Int
  val automatic: Boolean = true
  def color: String
}

new Car() // go wrong! abstract class could not be instantiated

class RedMini(val year: Int) extends Car {
  def color = "Red"
}

val m: Car = new RedMini(2015)

// here is the real magic
class Mini(val year: Int, val color: String) extends Car
val redMini: Car = new Mini(2015, "RED")
println(s"Got a ${redMini.color} Mini")

// as you see, we pass a value to a function color


// ==================== Anonymous Class ====================
// let me explain
// first define an anonymous class Listener
// class Listening contains a var as class Listener init-value is null
// after calling f: register, the var listener is assigend
// now, it is ready to send message!
abstract class Listener {def trigger}
class Listening {
  var listener: Listener = null
  def register(l: Listener) {listener = l}
  def sendNotification() { listener.trigger}
}
val notification = new Listening()
notification.register(new Listener {
  def trigger {println(s"Trigger at ${new java.util.Date}")}
})
notification.sendNotification

// ==================== Overload methods ====================
// Note: overloaded methods needs return type
class Printer(msg: String) {
  def print(s: String): Unit = println(s"$msg: $s")
  def print(l: Seq[String]): Unit = print(l.mkString(", ")) 
}

new Printer("Today's Report").print("Foggy" :: "Rainy" :: "hot" :: Nil)

// ==================== Apply method and lazy values ====================
// in short apply function is a shortcut
class Multiplier(factor: Int) {
  def apply(input: Int) = input * factor
}

val tripleMe = new Multiplier(3)
val tripled = tripleMe.apply(10)
val tripled2 = tripleMe(10)


// lazy variable is useful when we don't want to instantiate all
class RandomPoint {
  val x = {println("creating x"); util.Random.nextInt}
  lazy val y = {println("now y"); util.Random.nextInt}
}

val p = new RandomPoint()
println(s"Location is ${p.x}, ${p.y}")
